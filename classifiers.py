import torch
import torch.nn as nn

class CNN_classifier(nn.Module):
    """
    Used then for computed inception score with a classifier
    corresponding to the data
    """
    def __init__(self):
        super(CNN_classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        if not self.training:
            x = x.view(x.shape[0], 1, 28, 28)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if not self.training: #we need probability for inception score
            x = nn.functional.softmax(x, dim=1)
        return x

