import numpy as np
import torch
from tqdm import tqdm
from memory import SlidingWindow
import time

from neptune.utils import stringify_unsupported
from datasets import CLASSES
# from utils import tensor_linspace

def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def train(params, MemoryClass, model, train_loader, criterion, optimizer, use_le=True, feature_transform=None, run=None):

    run[f'adaptive_lr'].append(get_lr(optimizer))

    model.train()
    correct = 0
    train_loss = 0
    running_loss = 0.0
    it = 0
    total_samples_cnt = 0
    old_input, olf_target = None, None

    pbar = tqdm(train_loader, unit="audios", unit_scale=train_loader.batch_size)
    for batch_idx, (data, target) in enumerate(pbar):
        data = torch.unsqueeze(data, 1)

        if params['use_cuda']:
            data, target = data.cuda(), target.cuda()

        if feature_transform is not None:
            data = feature_transform(data)

        pred_sum = torch.zeros(data.shape[0], model.output_size, device=data.device)
        # implement sliding window and sum logits over windows
        memory = MemoryClass(data, **MemoryClass.kwargs)
        n_steps = len(memory)

        # init old input
        if old_input is None:
            old_input = torch.zeros_like(data[:, 0])
            old_target = torch.zeros_like(target)
        # run model for a few steps without training
        # if 'settling_steps' in params and params['settling_steps'] > 0:
        #     input_trans = tensor_linspace(old_input, data[:, 0], params['settling_steps']).unsqueeze(-1)
        #     target_trans = tensor_linspace(old_target, target, params['settling_steps'])
        #     with torch.no_grad():
        #         for i in range(params['settling_steps']):
        #             model(input_trans[:, i], target_trans[:, i], beta=params['beta'])

        for input in memory:
            optimizer.zero_grad()
            if use_le:
                raise NotImplementedError("LE not implemented for this model")
                # with torch.no_grad():
                #     for _ in range(params['n_updates']):
                #         # calling the model automatically populates the gradients
                #         output = model(input, target, beta=params['beta'])
                #         loss = loss_fn(output, target, reduction='sum')
                #         # # log exemplary memory output aka input
                #         # if run is not None and batch_idx == 0:
                #         #     for i in range(input.shape[1]):
                #         #         run[f"dynamics/memory_{i}"].append(input[0, i].detach().cpu().numpy())
            else:
                output = model(input)
                loss = criterion(output, target)

                loss.backward()
            optimizer.step()

            # average over steps
            pred_sum += output / n_steps
            train_loss += loss.item() / n_steps

            running_loss += loss.item()
            it += 1

        old_input = data[:, -1]
        old_target = target

        pred = pred_sum.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        # statistics
        target = target.cpu().numpy().astype(np.float32)
        total_samples_cnt += len(target)

        run[f'train_running_loss'].log(running_loss/it)

        # update the progress bar
        pbar.set_postfix({
            'train running loss': "%.05f" % (running_loss / it),
            'train running acc': "%.02f%%" % (100*correct/total_samples_cnt)
        })

    accuracy = correct/total_samples_cnt
    epoch_loss = running_loss / it
    run[f'train_acc'].log(100*accuracy)
    run[f'train_loss'].log(epoch_loss)

def test(params, MemoryClass, model, test_loader, criterion, run=None):
    model.eval()
    correct = 0
    running_loss = 0.0
    it = 0
    total_samples_cnt = 0

    pbar = tqdm(test_loader, unit="audios", unit_scale=test_loader.batch_size)
    for batch_i, (data, target) in enumerate(pbar):
        data = torch.unsqueeze(data, 1)

        if params['use_cuda']:
            data, target = data.cuda(), target.cuda()

        pred_sum = torch.zeros(data.shape[0], model.output_size, device=data.device)
        memory = MemoryClass(data, **MemoryClass.kwargs)
        n_steps = len(memory)

        # # run model for a few steps without training
        # if 'settling_steps' in params and params['settling_steps'] > 0:
        #     # init old input
        #     if old_input is None:
        #         old_input = torch.zeros_like(data[:, 0])
        #     input_trans = tensor_linspace(old_input, data[:, 0], params['settling_steps']).unsqueeze(-1)
        #     with torch.no_grad():
        #         for i in range(params['settling_steps']):
        #             model(input_trans[:, i])

        for input in memory:
            out = model(input)
            loss = criterion(out, target)
            _, pred_label = torch.max(out, 1)
            pred_sum += out / n_steps

            running_loss += loss.item()
            it += 1

        pred = pred_sum.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        # statistics
        target = target.cpu().numpy().astype(np.float32)
        total_samples_cnt += len(target)

        run[f'val_running_loss'].log(running_loss/it)

        # update the progress bar
        pbar.set_postfix({
            'val running loss': "%.05f" % (running_loss / it),
            'val running acc': "%.02f%%" % (100*correct/total_samples_cnt)
        })

    accuracy = correct/total_samples_cnt
    epoch_loss = running_loss / it
    run[f'val_acc'].log(100*accuracy)
    run[f'val_loss'].log(epoch_loss)

    return epoch_loss

def bens_run(args, params, name, memory, model, train_dataloader, valid_dataloader, criterion, optimizer, lr_scheduler):

    # import neptune and connect to neptune.ai
    try:
        try:
            if params['with_neptune']:
                import neptune
                run = neptune.init_run(
                    project=f"generalized-latent-equilibrium/{name}",
                    # capture_hardware_metrics=False,
                    # capture_stdout=False,
                    # capture_stderr=False,
                    # https://docs.neptune.ai/logging/system_metrics/
                )
                print(f"Starting {name} experiment...")
            else:
                print(f"Starting {name} experiment without observer...")
                run = None
        except ModuleNotFoundError:
            print(f"Neptune not found, starting {name} experiment without observer...")
            run = None

        # log parameters to neptune
        if run is not None:
            run["parameters"] = stringify_unsupported(params)

        params['name'] = name

        print(f"training on Google speech commands ({len(CLASSES)} classes)...")
        since = time.time()
        for epoch in range(0, args.max_epochs):
            if args.lr_scheduler == 'step':
                lr_scheduler.step()

            train(params, memory, model, train_dataloader, criterion, optimizer, use_le=params['use_le'], run=run)
            epoch_loss = test(params, memory, model, valid_dataloader, criterion, run=run)

            if args.lr_scheduler == 'plateau':
                lr_scheduler.step(metrics=epoch_loss)

            time_elapsed = time.time() - since
            time_str = 'total time elapsed: {:.0f}h {:.0f}m {:.0f}s '.format(time_elapsed // 3600, time_elapsed % 3600 // 60, time_elapsed % 60)
        print("finished")

    except KeyboardInterrupt:
        print('Interrupted')