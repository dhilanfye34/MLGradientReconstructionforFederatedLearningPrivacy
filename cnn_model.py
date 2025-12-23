import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallCNN(nn.Module):
    def __init__(self, pretrained=False):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1) 
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        if pretrained:
            self.load_pretrained_weights()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 14x14
        x = self.pool(F.relu(self.conv2(x)))  # 7x7
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def load_pretrained_weights(self):
        # Loads pretrained MNIST weights if available
        weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mnist_pretrained.pth")
        print("Loading weights from:", weights_path)
        try:
            if os.path.exists(weights_path):
                checkpoint = torch.load(weights_path, map_location=torch.device("cpu"), weights_only=False)
                self.load_state_dict(checkpoint)
                print("Successfully loaded pretrained weights")
            else:
                print(f"Pretrained weights not found at {weights_path}")
        except Exception as e:
            print(f"Failed to load pretrained weights: {e}")

def load_from_state_dict(state):
    m = SmallCNN(pretrained=False)
    m.load_state_dict(state)
    return m
