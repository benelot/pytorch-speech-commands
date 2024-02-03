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


def train(params, MemoryClass, model, train_loader, criterion, optimizer, epoch, fn_out, use_le=True, feature_transform=None, run=None):

    run[f'adaptive_lr'].append(get_lr(optimizer))

    model.train()
    correct = 0
    train_loss = 0
    running_loss = 0.0
    it = 0
    total_samples_cnt = 0
    old_input, old_target = None, None

    pbar = tqdm(train_loader, unit="audios", unit_scale=train_loader.batch_size)
    for batch_idx, (data, target) in enumerate(pbar):
        batch_loss = 0
        # data = data.float() # TODO: will change acc
        data = data.unsqueeze(1) # TODO: Wrap into a transform

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
            batch_loss += loss.item() / n_steps / input.shape[0]
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

        if batch_idx % params['log_interval'] == 0:
            print('Train Epoch: {}({}) [{}/{} ({:.0f}%)]\tBatch Loss: {:.6f}'.format(
                epoch, batch_idx, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), batch_loss))

        if fn_out is not None and batch_idx % params['checkpoint_interval'] == 0:
            torch.save(model.state_dict(), fn_out.format(postfix=f'_{epoch}_{batch_idx}'))

        # statistics
        target = target.cpu().numpy().astype(np.float32)
        total_samples_cnt += len(target)

        run[f'train_running_loss'].log(running_loss/it)

        # update the progress bar
        pbar.set_postfix({
            'train running loss': "%.05f" % (running_loss / it),
            'train running acc': "%.02f%%" % (100*correct/total_samples_cnt)
        })

    # average loss over dataset
    train_acc = 100. * correct / total_samples_cnt
    epoch_loss = running_loss / it

    if run is not None:
        # TODO: log train_loss vs. epoch_loss
        run[f"train_loss"].append(epoch_loss)
        run[f"train_acc"].append(train_acc)

    return train_loss, train_acc

def test(params, MemoryClass, model, test_loader, criterion, feature_transform=None, epoch=0, prefix='valid', lr_scheduler=None, run=None):
    model.eval()
    correct = 0
    test_loss = 0
    old_input = None
    running_loss = 0.0
    it = 0
    total_samples_cnt = 0

    pbar = tqdm(test_loader, unit="audios", unit_scale=test_loader.batch_size)
    if True: # TODO: dummy for torch.no_grad()
        for batch_i, (data, target) in enumerate(pbar):
            # data = data.float() # TODO: will change acc
            data = data.unsqueeze(1)

            if params['use_cuda']:
                data, target = data.cuda(), target.cuda()

            if feature_transform is not None:
                data = feature_transform(data)

            pred_sum = torch.zeros(data.shape[0], model.output_size, device=data.device)
            memory = MemoryClass(data, **MemoryClass.kwargs)
            n_steps = len(memory)

            # run model for a few steps without training
            # if 'settling_steps' in params and params['settling_steps'] > 0:
            #     # init old input
            #     if old_input is None:
            #         old_input = torch.zeros_like(data[:, 0])
            #     input_trans = tensor_linspace(old_input, data[:, 0], params['settling_steps']).unsqueeze(-1)
            #     with torch.no_grad():
            #         for i in range(params['settling_steps']):
            #             model(input_trans[:, i])

            for input in memory:
                if params['use_le']:
                    raise NotImplementedError("LE not implemented for this model")
                    # for _ in range(params['n_updates']):
                        # output = model(input)
                        # # input_list.append(input[0].detach().cpu().numpy())
                        # # preds_activation_list.append(output[0].detach().cpu().numpy())
                        # # average over steps
                        # pred_sum += output / n_steps
                        # test_loss += loss_fn(output, target, reduction='sum').item() / n_steps
                else:
                    out = model(input)
                    loss = criterion(out, target)
                    _, pred_label = torch.max(out, 1)
                    pred_sum += out / n_steps

                    running_loss += loss.item()
                    it += 1

                old_input = data[:, -1]

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

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / total_samples_cnt
    epoch_loss = running_loss / it

    if lr_scheduler is not None:
        lr_scheduler.step(epoch_loss)
        if run is not None:
            run['adaptive_lr'].append(lr_scheduler.optimizer.param_groups[0]['lr'])

    if run is not None:
        run[f"val_acc"].append(test_acc)
        run[f"val_loss"].append(epoch_loss)
    print('Evaluate on', prefix, 'set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_acc))

    return test_loss, test_acc

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
            # if args.lr_scheduler == 'step':
            #     lr_scheduler.step()

            train(params, memory, model, train_dataloader, criterion, optimizer, epoch=epoch, fn_out=None, use_le=params['use_le'], run=run)
            test(params, memory, model, valid_dataloader, criterion, run=run)

            # if args.lr_scheduler == 'plateau':
            #     lr_scheduler.step(metrics=epoch_loss)

            time_elapsed = time.time() - since
            time_str = 'total time elapsed: {:.0f}h {:.0f}m {:.0f}s '.format(time_elapsed // 3600, time_elapsed % 3600 // 60, time_elapsed % 60)
        print("finished")

    except KeyboardInterrupt:
        print('Interrupted')