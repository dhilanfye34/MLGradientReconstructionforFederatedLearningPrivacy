import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallCNN(nn.Module):
    def __init__(self, pretrained=False):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 10)

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
        """Loads pretrained CIFAR-10 weights if available."""
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "MLGradientReconstructionforFederatedLearningPrivacy"))
        weights_path = os.path.join(BASE_DIR, "cifar10_pretrained.pth")
        print("Loading weights from:", weights_path)
        try:
            # Directly load the checkpoint using torch.load
            checkpoint = torch.load(weights_path, map_location=torch.device("cpu"), weights_only=False)
            self.load_state_dict(checkpoint, strict=False)
            print("✅ Successfully loaded pretrained weights!")
        except Exception as e:
            print(f"❌ Failed to load pretrained weights: {e}")
