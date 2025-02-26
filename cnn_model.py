import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallCNN(nn.Module):
    def __init__(self, pretrained=False):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 32 * 32, 128)  # Adjust for CIFAR-10 image size (32x32)
        self.fc2 = nn.Linear(128, 10)  # CIFAR-10 has 10 classes

        if pretrained:
            self.load_pretrained_weights()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def load_pretrained_weights(self):
        try:
            state_dict = torch.load("cifar10_pretrained.pth", map_location=torch.device("cpu"))
            self.load_state_dict(state_dict)
            print("✅ Loaded pretrained CIFAR-10 weights.")
        except Exception as e:
            print(f"⚠️ Failed to load pretrained weights: {e}")

