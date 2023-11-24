import torch
import torch.nn as nn
import torch.quantization

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation_function="relu", sigma_magnitude=255.0, classification=False):
        super().__init__()
        self.activation_function = activation_function
        if activation_function == "custom":
            activation_function = "relu"
        self.rnn = nn.RNNCell(input_size, hidden_size, nonlinearity=activation_function)
        self.linear = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
        self.sigma_magnitude = sigma_magnitude
        self.classification = classification

    def forward(self, x):
        y = torch.zeros(x.shape[1], self.hidden_size)
        if self.activation_function == "tanh":
            x = x/self.sigma_magnitude
        for i in range(x.shape[0]):
            y = self.rnn(x[i], y)
            if self.activation_function == "tanh":
                y = y*self.sigma_magnitude
            elif self.activation_function == "custom":
                y = torch.clamp(y, min=0, max=self.sigma_magnitude)
                y = torch.sqrt(y/self.sigma_magnitude) * self.sigma_magnitude

        y = self.linear(y)
        if self.classification:
            y = torch.softmax(y, -1)
        return y
