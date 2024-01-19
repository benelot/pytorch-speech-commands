# The MNIST-1D dataset | 2020
# Sam Greydanus

import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearBase(nn.Module):
  def __init__(self, input_size, output_size):
    super(LinearBase, self).__init__()
    self.linear = nn.Linear(input_size, output_size)
    print("Initialized LinearBase model with {} parameters".format(self.count_params()))

  def count_params(self):
    return sum([p.view(-1).shape[0] for p in self.parameters()])

  def forward(self, x):
    x = x.view(x.shape[0], -1) # flatten the input
    return self.linear(x)

class MLPBase(nn.Module):
  def __init__(self, input_size, output_size, hidden_size=100):
    super(MLPBase, self).__init__()
    self.input_size = input_size
    self.linear1 = nn.Linear(input_size, hidden_size)
    self.linear2 = nn.Linear(hidden_size, hidden_size)
    self.linear3 = nn.Linear(hidden_size, output_size)
    print("Initialized MLPBase model with {} parameters".format(self.count_params()))

  def count_params(self):
    return sum([p.view(-1).shape[0] for p in self.parameters()])

  def forward(self, x):
    x = x.reshape(x.shape[0], -1) # flatten the input
    h = self.linear1(x).relu()
    # h = h + self.linear2(h).relu() # residual connection
    h = self.linear2(h).relu() # no residual connection # TODO: the results are better without the residual connection :D
    return self.linear3(h)

class ConvBase(nn.Module):
    def __init__(self, output_size, channels=25, linear_in=125):
        super(ConvBase, self).__init__()
        self.conv1 = nn.Conv1d(1, channels, 5, stride=2, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(channels, channels, 3, stride=2, padding=1)
        self.linear = nn.Linear(linear_in, output_size) # flattened channels -> 10 (assumes input has dim 50)
        print("Initialized ConvBase model with {} parameters".format(self.count_params()))

    def count_params(self):
        return sum([p.view(-1).shape[0] for p in self.parameters()])

    def forward(self, x, verbose=False): # the print statements are for debugging
        x = x.view(-1,1,x.shape[-1]) # reshape to (batch, channels, time)
        h1 = self.conv1(x).relu()
        h2 = self.conv2(h1).relu()
        h3 = self.conv3(h2).relu()
        h3 = h3.view(h3.shape[0], -1) # flatten the conv features
        return self.linear(h3) # a linear classifier goes on top

class TemporalConvBase(nn.Module):
    def __init__(self, input_size=5, output_size=10, channels=25):
        super(TemporalConvBase,self).__init__()
        self.conv1d = nn.Conv1d(1, channels, input_size)
        self.lin = nn.Linear(channels, output_size)
        print("Initialized TemporalConvBase model with {} parameters".format(self.count_params()))

        self.input_size = input_size

    def count_params(self):
        return sum([p.view(-1).shape[0] for p in self.parameters()])

    def forward(self, x):
        x = x.view(-1,1,x.shape[-1]) # reshape to (batch, channels, time)
        x = self.conv1d(x).relu()
        x = x.view(x.shape[0], -1) # flatten the conv features
        return self.lin(x)

class GRUBase(torch.nn.Module):
  def __init__(self, input_size, output_size, hidden_size=6, time_steps=40, bidirectional=True):
    super(GRUBase, self).__init__()
    self.gru = torch.nn.GRU(input_size=input_size, hidden_size=hidden_size,
                            batch_first=True, bidirectional=bidirectional)
    flat_size = 2*hidden_size*time_steps if bidirectional else hidden_size*time_steps
    self.linear = torch.nn.Linear(flat_size, output_size)
    self.hidden_size = hidden_size
    self.bidirectional = bidirectional
    print("Initialized GRUBase model with {} parameters".format(self.count_params()))

  def count_params(self):
    return sum([p.view(-1).shape[0] for p in self.parameters()])

  def forward(self, x, h0=None): # assumes seq has [batch, time]
    x = x.view(*x.shape[:2], 1) # add a spatial dimension
    k = 2 if self.bidirectional else 1
    h0 = 0*torch.randn(k, x.shape[0], self.hidden_size) if h0 is None else h0
    h0 = h0.to(x.device) # GPU support

    output, hn = self.gru(x, h0)
    output = output.reshape(output.shape[0],-1) # [batch, time*hidden_size]
    return self.linear(output) # a linear classifier goes on top

class LSTMBase(torch.nn.Module):
  def __init__(self, input_size, output_size, hidden_size=6, time_steps=40, bidirectional=True):
    super(LSTMBase, self).__init__()
    self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            batch_first=True, bidirectional=bidirectional)
    flat_size = 2*hidden_size*time_steps if bidirectional else hidden_size*time_steps
    self.linear = torch.nn.Linear(flat_size, output_size)
    self.hidden_size = hidden_size
    self.bidirectional = bidirectional
    print("Initialized LSTMBase model with {} parameters".format(self.count_params()))

  def count_params(self):
    return sum([p.view(-1).shape[0] for p in self.parameters()])

  def forward(self, x, h0=None, c0=None): # assumes seq has [batch, time]
    x = x.view(*x.shape[:2], 1) # add a spatial dimension
    k = 2 if self.bidirectional else 1
    h0 = 0*torch.randn(k, x.shape[0], self.hidden_size) if h0 is None else h0
    c0 = 0*torch.randn(k, x.shape[0], self.hidden_size) if c0 is None else c0
    h0 = h0.to(x.device) # GPU support
    c0 = c0.to(x.device) # GPU support

    output, (hn, cn) = self.lstm(x, (h0, c0))
    output = output.reshape(output.shape[0],-1) # [batch, time*hidden_size]
    return self.linear(output) # a linear classifier goes on top