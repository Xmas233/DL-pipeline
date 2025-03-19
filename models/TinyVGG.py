"""
Contains Pytorhch model code to instantiate a TinyVGG model.
"""
import torch
import torch.nn as nn

class TinyVGG(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_shape,
                      hidden_units,
                      3,
                      1,
                      0),
            nn.ReLU(),
            nn.Conv2d(hidden_units,
                      hidden_units,
                      3,
                      1,
                      0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units,
                      hidden_units,
                      3,
                      1,
                      0),
            nn.ReLU(),
            nn.Conv2d(hidden_units,
                      hidden_units,
                      3,
                      1,
                      0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units *13 * 13, output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x