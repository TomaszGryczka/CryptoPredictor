import torch
import torch.nn as nn


class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        # create LSTM model
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=0.2)
        self.first_fully_connected_layer = nn.Linear(hidden_size, 128)
        self.second_fully_connected_layer = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # initialize lstm layers
        initial_hidden_state = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        initial_cell_state = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # forward propagate lstm
        _, (final_hidden_state, _) = self.lstm(x, (initial_hidden_state, initial_cell_state))
        final_hidden_state = final_hidden_state.view(-1, self.hidden_size)
        out = self.relu(final_hidden_state)
        out = self.first_fully_connected_layer(out)
        out = self.relu(out)
        out = self.second_fully_connected_layer(out)
        return out
