import torch
import torch.nn as nn
import torch.nn.functional as F

class StartingNetwork(torch.nn.Module):
    """
    Basic CNN model
    """

    def __init__(self):
        super().__init__()
        '''
        Layers:
        - Conv2d:
        '''
        self.conv1 = nn.Conv2d(3, 8, kernel_size=5, padding=2) # 3x600x600
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(16*300*300, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # print("shape after conv layers and pooling:", x.shape)
        x = torch.reshape(x, (-1, 16*300*300))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # no relu at the end
        return x
