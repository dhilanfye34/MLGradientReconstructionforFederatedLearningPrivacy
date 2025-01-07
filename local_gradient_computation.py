# File: local_gradient_computation.py
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from inversefed.nn.models import construct_model

# Load ResNet18 model
model, _ = construct_model("ResNet18", num_classes=1000)
model.eval()

# Load a small dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dataset = datasets.FakeData(transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

# Perform gradient computation
for data, target in dataloader:
    data, target = data.to('cpu'), target.to('cpu')
    output = model(data)
    loss = nn.CrossEntropyLoss()(output, target)
    gradients = torch.autograd.grad(loss, model.parameters(), create_graph=True)

    # Save gradients locally
    torch.save(gradients, "local_gradients.pt")
    break  # Compute for one batch
