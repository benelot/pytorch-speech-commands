import torch

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