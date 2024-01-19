#!/usr/bin/env python
# %%
"""Train a CNN for Google speech commands."""

# adapted from: https://github.com/tugstugi/pytorch-speech-commands

import argparse
import time

import sys
sys.path.append("..")

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from torchvision.transforms import *

# import models
# from data.gsc.gsc_dataset import *
# from data.gsc import *
from tqdm import *

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

import torchvision
from torchvision.transforms import *

import models
from datasets import *
from transforms import *
from mixup import *
from baseline_models import LinearBase, MLPBase

import neptune



# %%

from neptune.utils import stringify_unsupported
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

params = {
    # NN parameters
    "model_type": "cnn",
    # gsc parameters
    # other parameters
    "use_cuda": True,
    "with_neptune": True,
}

# parse parameters from command line which often change
parser = argparse.ArgumentParser(description='Train a GLE network on the MNIST1D dataset.')
parser.add_argument('--no_neptune', action='store_true', default=False, help='Do not use neptune.')
parser.add_argument('--seed', type=int, default=12, help='Random seed.')
parser.add_argument("--train-dataset", type=str, default='datasets/speech_commands/train', help='path of train dataset')
parser.add_argument("--valid-dataset", type=str, default='datasets/speech_commands/valid', help='path of validation dataset')
parser.add_argument("--background-noise", type=str, default='datasets/speech_commands/train/_background_noise_', help='path of background noise')
parser.add_argument("--comment", type=str, default='', help='comment in tensorboard title')
parser.add_argument("--batch-size", type=int, default=128, help='batch size')
parser.add_argument("--dataload-workers-nums", type=int, default=6, help='number of workers for dataloader')
parser.add_argument("--weight-decay", type=float, default=0, help='weight decay')
parser.add_argument("--optim", choices=['sgd', 'adam'], default='adam', help='choices of optimization algorithms')
parser.add_argument("--learning-rate", type=float, default=5e-5, help='learning rate for optimization')
parser.add_argument("--lr-scheduler", choices=['plateau', 'step'], default='plateau', help='method to adjust learning rate')
parser.add_argument("--lr-scheduler-patience", type=int, default=5, help='lr scheduler plateau: Number of epochs with no improvement after which learning rate will be reduced')
parser.add_argument("--lr-scheduler-step-size", type=int, default=50, help='lr scheduler step: number of epochs of learning rate decay.')
parser.add_argument("--lr-scheduler-gamma", type=float, default=0.5, help='learning rate is multiplied by the gamma to decrease it')
parser.add_argument("--max-epochs", type=int, default=150, help='max number of epochs')
parser.add_argument("--resume", type=str, help='checkpoint file to resume')
parser.add_argument("--input", choices=['mel32','mel40'], default='mel32', help='input of NN')
parser.add_argument('--mixup', action='store_true', help='use mixup')
args = parser.parse_known_args()[0]

params['optim'] = args.optim
print("Using optimizer: {}".format(params['optim']))

params['with_neptune'] = not args.no_neptune
print("Using neptune: {}".format(params['with_neptune']))

params['seed'] = args.seed
print("Using seed: {}".format(params['seed']))

params['epochs'] = args.max_epochs
print("Using max epochs: {}".format(params['epochs']))

params['batch_size'] = args.batch_size
print("Using batch size: {}".format(params['batch_size']))

params['lr'] = args.learning_rate
print("Using learning rate: {}".format(params['lr']))

params['lr_scheduler'] = args.lr_scheduler
print("Using lr scheduler: {}".format(params['lr_scheduler']))

params['lr_scheduler_patience'] = args.lr_scheduler_patience
print("Using lr scheduler patience: {}".format(params['lr_scheduler_patience']))

params['lr_scheduler_step_size'] = args.lr_scheduler_step_size
print("Using lr scheduler step size: {}".format(params['lr_scheduler_step_size']))

params['lr_scheduler_gamma'] = args.lr_scheduler_gamma
print("Using lr scheduler gamma: {}".format(params['lr_scheduler_gamma']))

# import neptune and connect to neptune.ai
name = "130-gsc-mel-classification"

torch.manual_seed(params["seed"])

class SlidingWindow:

    def __init__(self, data, window_size, dilation=1):
        self.data = data
        self.dilation = dilation
        self.window_size = window_size
        self.window_idx = 0

    def __len__(self):
        return self.data.shape[-1] - self.dilation * (self.window_size - 1)

    def __getitem__(self, idx):
        if self.window_idx >= len(self):
            raise StopIteration

        window = torch.squeeze(self.data[:, :, :, self.window_idx:self.window_idx + self.dilation * self.window_size:self.dilation])

        self.window_idx += 1
        return window


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


    use_gpu = torch.cuda.is_available()
    print('use_gpu', use_gpu)
    if use_gpu:
        torch.backends.cudnn.benchmark = True

    n_mels = 32
    if args.input == 'mel40':
        n_mels = 40

    data_aug_transform = Compose([ChangeAmplitude(), ChangeSpeedAndPitchAudio(), FixAudioLength(), ToSTFT(), StretchAudioOnSTFT(), TimeshiftAudioOnSTFT(), FixSTFTDimension()])
    bg_dataset = BackgroundNoiseDataset(args.background_noise, data_aug_transform)
    add_bg_noise = AddBackgroundNoiseOnSTFT(bg_dataset)
    train_feature_transform = Compose([ToMelSpectrogramFromSTFT(n_mels=n_mels), DeleteSTFT(), ToTensor('mel_spectrogram', 'input')])
    train_dataset = SpeechCommandsDataset(args.train_dataset,
                                    Compose([LoadAudio(),
                                            data_aug_transform,
                                            add_bg_noise,
                                            train_feature_transform]))

    valid_feature_transform = Compose([ToMelSpectrogram(n_mels=n_mels), ToTensor('mel_spectrogram', 'input')])
    valid_dataset = SpeechCommandsDataset(args.valid_dataset,
                                    Compose([LoadAudio(),
                                            FixAudioLength(),
                                            valid_feature_transform]))

    weights = train_dataset.make_weights_for_balanced_classes()
    sampler = WeightedRandomSampler(weights, len(weights))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler,
                                pin_memory=use_gpu, num_workers=args.dataload_workers_nums)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                pin_memory=use_gpu, num_workers=args.dataload_workers_nums)


    # a name used to save checkpoints etc.
    full_name = '%s_%s_%s_bs%d_lr%.1e' % ("MLPBase", args.optim, args.lr_scheduler, args.batch_size, args.learning_rate)
    if args.comment:
        full_name = '%s_%s' % (full_name, args.comment)

    # model = LinearBase(n_mels * 32, len(CLASSES))
    kernel_size = 16
    model = MLPBase(n_mels * kernel_size, len(CLASSES))
    model.input_size = kernel_size

    if use_gpu:
        model = model.cuda()

    criterion = torch.nn.CrossEntropyLoss()

    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    start_timestamp = int(time.time()*1000)
    start_epoch = 0
    best_accuracy = 0
    best_loss = 1e100
    global_step = 0

    if args.resume:
        print("resuming a checkpoint '%s'" % args.resume)
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        model.float()
        optimizer.load_state_dict(checkpoint['optimizer'])

        best_accuracy = checkpoint.get('accuracy', best_accuracy)
        best_loss = checkpoint.get('loss', best_loss)
        start_epoch = checkpoint.get('epoch', start_epoch)
        global_step = checkpoint.get('step', global_step)

        del checkpoint  # reduce memory

    if args.lr_scheduler == 'plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.lr_scheduler_patience, factor=args.lr_scheduler_gamma)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_scheduler_step_size, gamma=args.lr_scheduler_gamma, last_epoch=start_epoch-1)

    def get_lr():
        return optimizer.param_groups[0]['lr']


    def train_temporal_style(epoch):
        global global_step

        print("epoch %3d with lr=%.02e" % (epoch, get_lr()))
        phase = 'train'
        run[f'{phase}/learning_rate'].append(get_lr())
        # mlflow.log_metric('%s/learning_rate' % phase,  get_lr(), step=epoch)

        model.train()  # Set model to training mode

        running_loss = 0.0
        it = 0
        correct_windows_cnt = 0
        total_windows_cnt = 0
        correct_samples_cnt = 0
        total_samples_cnt = 0

        pbar = tqdm(train_dataloader, unit="audios", unit_scale=train_dataloader.batch_size)
        for batch in pbar:
            inputs = batch['input']
            inputs = torch.unsqueeze(inputs, 1)
            targets = batch['target']

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
            global_step += 1

            run[f'{phase}/loss'].log(running_loss/it)
            # mlflow.log_metric('%s/loss' % phase, running_loss/it, step=global_step)

            # update the progress bar
            pbar.set_postfix({
                'loss': "%.05f" % (running_loss / it),
                'acc': "%.02f%%" % (100*correct_samples_cnt/total_samples_cnt)
            })

        accuracy = correct_samples_cnt/total_samples_cnt
        epoch_loss = running_loss / it
        run[f'{phase}/accuracy'].log(100*accuracy)
        # mlflow.log_metric('%s/accuracy' % phase, 100*accuracy, step=epoch)
        run[f'{phase}/epoch_loss'].log(epoch_loss)
        # mlflow.log_metric('%s/epoch_loss' % phase, epoch_loss, step=epoch)

    def validate_temporal_style(epoch):
        global best_accuracy, best_loss, global_step

        phase = 'valid'
        model.eval()  # Set model to evaluate mode

        running_loss = 0.0
        it = 0
        correct_windows_cnt = 0
        total_windows_cnt = 0
        correct_samples_cnt = 0
        total_samples_cnt = 0

        pbar = tqdm(valid_dataloader, unit="audios", unit_scale=valid_dataloader.batch_size)
        for batch in pbar:
            inputs = batch['input']
            inputs = torch.unsqueeze(inputs, 1)
            targets = batch['target']

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
            global_step += 1

            run[f'{phase}/loss'].log(running_loss/it)
            # mlflow.log_metric('%s/loss' % phase, loss.item(), step=global_step)

            # update the progress bar
            pbar.set_postfix({
                'loss': "%.05f" % (running_loss / it),
                'acc': "%.02f%%" % (100*correct_samples_cnt/total_samples_cnt)
            })

        accuracy = correct_samples_cnt/total_samples_cnt
        epoch_loss = running_loss / it
        run[f'{phase}/accuracy'].log(100*accuracy)
        # mlflow.log_metric('%s/accuracy' % phase, 100*accuracy, step=epoch)
        run[f'{phase}/epoch_loss'].log(epoch_loss)
        # mlflow.log_metric('%s/epoch_loss' % phase, epoch_loss, step=epoch)

        checkpoint = {
            'epoch': epoch,
            'step': global_step,
            'state_dict': model.state_dict(),
            'loss': epoch_loss,
            'accuracy': accuracy,
            'optimizer' : optimizer.state_dict(),
        }

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(checkpoint, 'checkpoints/best-loss-speech-commands-checkpoint-%s.pth' % full_name)
            torch.save(model, 'checkpoints/%d-%s-best-loss.pth' % (start_timestamp, full_name))
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(checkpoint, 'checkpoints/best-acc-speech-commands-checkpoint-%s.pth' % full_name)
            torch.save(model, 'checkpoints/%d-%s-best-acc.pth' % (start_timestamp, full_name))

        torch.save(checkpoint, 'checkpoints/last-speech-commands-checkpoint.pth')
        del checkpoint  # reduce memory

        return epoch_loss

    print(f"training on Google speech commands ({len(CLASSES)} classes)...")
    since = time.time()
    for epoch in range(start_epoch, args.max_epochs):
        if args.lr_scheduler == 'step':
            lr_scheduler.step()

        train_temporal_style(epoch)
        epoch_loss = validate_temporal_style(epoch)

        if args.lr_scheduler == 'plateau':
            lr_scheduler.step(metrics=epoch_loss)

        time_elapsed = time.time() - since
        time_str = 'total time elapsed: {:.0f}h {:.0f}m {:.0f}s '.format(time_elapsed // 3600, time_elapsed % 3600 // 60, time_elapsed % 60)
        print("%s, best accuracy: %.02f%%, best loss %f" % (time_str, 100*best_accuracy, best_loss))
    print("finished")
    # %%
except KeyboardInterrupt:
    print('Interrupted')