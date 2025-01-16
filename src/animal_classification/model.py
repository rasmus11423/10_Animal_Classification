import torch
import torch.nn as nn

class Animal_classifier(nn.Module):
    def __init__(self):
        super(Animal_classifier, self).__init__()
        
        # 1 channel, 48x48 pixels
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64*10*10, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.sequential(x)


if __name__ == '__main__':
    model = Animal_classifier()
    print(model)
    x = torch.randn(1, 1, 48, 48)
    print(model(x).shape)