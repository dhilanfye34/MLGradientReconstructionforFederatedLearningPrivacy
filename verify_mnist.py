import torch
from torchvision import datasets, transforms
from cnn_model import SmallCNN

# Load model
model = SmallCNN(pretrained=True)
model.eval()

# Load one MNIST test sample
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
mnist = datasets.MNIST(root="./dataset_assets", train=False, download=True, transform=transform)
image, label = mnist[0]
image = image.unsqueeze(0)  # Add batch dimension

# Run prediction
with torch.no_grad():
    output = model(image)
    pred = torch.argmax(output, dim=1)
    print(f"✅ Model Prediction: {pred.item()} | True Label: {label}")
