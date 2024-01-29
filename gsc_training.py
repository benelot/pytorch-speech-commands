import numpy as np
import torch
from tqdm import tqdm
from memory import SlidingWindow
import time

from neptune.utils import stringify_unsupported
from datasets import CLASSES

def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def train(model, train_dataloader, criterion, optimizer, use_gpu, run):

    run[f'adaptive_lr'].append(get_lr(optimizer))

    model.train()  # Set model to training mode
    running_loss = 0.0
    it = 0
    correct_windows_cnt = 0
    total_windows_cnt = 0
    correct_samples_cnt = 0
    total_samples_cnt = 0

    pbar = tqdm(train_dataloader, unit="audios", unit_scale=train_dataloader.batch_size)
    for inputs, targets in pbar:
        inputs = torch.unsqueeze(inputs, 1)

        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()

        prediction_probabilities = []

        memory = SlidingWindow(inputs, model.input_size)
        for input in memory:

            outputs = model(input)
            loss = criterion(outputs, targets)
            _, pred_label = torch.max(outputs, 1)
            prediction_probabilities.append(np.expand_dims(outputs.detach().cpu().numpy(), axis=-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct_windows_cnt += (pred_label == targets).sum()
            total_windows_cnt += targets.shape[0]
            it += 1

        # statistics
        predictions = np.argmax(np.mean(np.concatenate(prediction_probabilities, axis=-1), axis=-1), axis=-1)
        target = targets.cpu().numpy().astype(np.float32)
        correct_samples_cnt += sum(predictions == target)
        total_samples_cnt += len(target)

        run[f'train_running_loss'].log(running_loss/it)

        # update the progress bar
        pbar.set_postfix({
            'train running loss': "%.05f" % (running_loss / it),
            'train running acc': "%.02f%%" % (100*correct_samples_cnt/total_samples_cnt)
        })

    accuracy = correct_samples_cnt/total_samples_cnt
    epoch_loss = running_loss / it
    run[f'train_acc'].log(100*accuracy)
    run[f'train_loss'].log(epoch_loss)

def validate(model, valid_dataloader, criterion, use_gpu, run):

    model.eval()  # Set model to evaluate mode

    running_loss = 0.0
    it = 0
    correct_windows_cnt = 0
    total_windows_cnt = 0
    correct_samples_cnt = 0
    total_samples_cnt = 0

    pbar = tqdm(valid_dataloader, unit="audios", unit_scale=valid_dataloader.batch_size)
    for inputs, targets in pbar:
        inputs = torch.unsqueeze(inputs, 1)

        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()

        prediction_probabilities = []
        memory = SlidingWindow(inputs, model.input_size)
        for input in memory:

            out = model(input)
            loss = criterion(out, targets)
            _, pred_label = torch.max(out, 1)
            prediction_probabilities.append(np.expand_dims(out.detach().cpu().numpy(), axis=-1))

            running_loss += loss.item()
            correct_windows_cnt += (pred_label == targets).sum()
            total_windows_cnt += targets.shape[0]
            it += 1

        # statistics
        predictions = np.argmax(np.mean(np.concatenate(prediction_probabilities, axis=-1), axis=-1), axis=-1)
        target = targets.cpu().numpy().astype(np.float32)
        correct_samples_cnt += sum(predictions == target)
        total_samples_cnt += len(target)

        run[f'val_running_loss'].log(running_loss/it)

        # update the progress bar
        pbar.set_postfix({
            'val running loss': "%.05f" % (running_loss / it),
            'val running acc': "%.02f%%" % (100*correct_samples_cnt/total_samples_cnt)
        })

    accuracy = correct_samples_cnt/total_samples_cnt
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

            train(model, train_dataloader, criterion, optimizer, params['use_cuda'], run)
            epoch_loss = validate(model, valid_dataloader, criterion, params['use_cuda'], run)

            if args.lr_scheduler == 'plateau':
                lr_scheduler.step(metrics=epoch_loss)

            time_elapsed = time.time() - since
            time_str = 'total time elapsed: {:.0f}h {:.0f}m {:.0f}s '.format(time_elapsed // 3600, time_elapsed % 3600 // 60, time_elapsed % 60)
        print("finished")

    except KeyboardInterrupt:
        print('Interrupted')