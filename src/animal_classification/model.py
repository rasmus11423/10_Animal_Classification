import torch
import torch.nn as nn


class AnimalClassifier(nn.Module):
    def __init__(self):
        super(AnimalClassifier, self).__init__()

        # 1 channel, 48x48 pixels
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),  # BatchNorm after convolution
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),  # Dropout for regularization

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),  # BatchNorm after convolution
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),  # Dropout for regularization
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 128),
            nn.LayerNorm(128),  # LayerNorm for fully connected layer
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.sequential(x)


if __name__ == "__main__":
    model = AnimalClassifier()
    print(model)
    x = torch.randn(1, 1, 48, 48)  # Single batch
    print(model(x).shape)