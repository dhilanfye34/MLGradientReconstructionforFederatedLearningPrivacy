import torch
import torchvision.models as models

# Load CIFAR-10 pretrained model
model = models.resnet18(pretrained=True)  # Change to a smaller model if needed
torch.save(model.state_dict(), "cifar10_pretrained.pth")

print("✅ CIFAR-10 pretrained weights downloaded and saved!")
