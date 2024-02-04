"""
Main file continaing models to be tested, compared and explored in the 
main notebook.

Creation date: 01/02/2024
Last_modification: 01/02/2024
By: Mehdi EL KANSOULI 
"""

import numpy as np 
import torch 
import torch.nn as nn 


class WeightLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias_gain=None, init='he'):
        # bias_gain=None -> no biases, else train biases with he or normal initialization with std=bias_gain
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias_gain = bias_gain
        self.init=init
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        if init == 'he':
            self.bias.data.fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.matmul(self.weight.t()) * np.sqrt(2/self.in_features)
        if self.bias_gain is not None:
            x = x + self.bias_gain * self.bias
        return x


class FunctionLayer(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x)
    

class SpikyModel(nn.Module):
    
    def __init__(self, freq, input_size, hidden_size, output_size, bias_gain=1):
        super(SpikyModel, self).__init__()

        # define custom acitvation function
        sincos = lambda x: torch.sin(x) + torch.cos(x)
        act = lambda x: torch.relu(x) + sincos(freq * x) / (freq)
        self.custom_activation = FunctionLayer(act)

        # define each layer
        self.layer1 = WeightLayer(input_size, hidden_size, bias_gain=bias_gain)
        self.layer2 = WeightLayer(hidden_size, output_size, bias_gain=bias_gain)

    def forward(self, x):
        x = self.layer1(x)
        x = self.custom_activation(x)
        x = self.layer2(x)

        return x
    
    def inference(self, x):
        """
        Custom function that extracts the signal from the network 
        without noise. 
        """
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)

        return x


class OneHiddenRelu(nn.Module):
    """
    One hidden layer ReLU network. 
    """
    def __init__(self, input, hidden_size):
        super(OneHiddenRelu, self).__init__()
        self.hidden_layer = nn.Linear(input, hidden_size)  # One hidden layer with 10 neurons
        self.output_layer = nn.Linear(hidden_size, 1)  # Output layer

    def forward(self, x):
        x = torch.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x
    

class DeepRelu(nn.Module):
    def __init__(self, input, hidden_size):
        super(DeepRelu, self).__init__()
        self.hidden_layer_1 = nn.Linear(input, hidden_size)  
        self.hidden_layer_2 = nn.Linear(hidden_size, hidden_size) 
        self.hidden_layer_3 = nn.Linear(hidden_size, hidden_size) 
        self.output_layer = nn.Linear(hidden_size, 1)  

    def forward(self, x):
        x = torch.relu(self.hidden_layer_1(x))
        x = torch.relu(self.hidden_layer_2(x))
        x = torch.relu(self.hidden_layer_3(x))
        x = self.output_layer(x)
        return x