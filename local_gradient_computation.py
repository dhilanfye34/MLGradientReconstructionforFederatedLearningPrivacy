import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim

def compute_gradients():
    # Load a small dataset (simulated or downloaded)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = torchvision.datasets.FakeData(transform=transform, size=10)  # Use FakeData for testing
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

    # Define a simple model
    model = nn.Sequential(
        nn.Linear(3 * 32 * 32, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    model = model.to("cpu")  # Ensure compatibility with Raspberry Pi

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Iterate over the dataset and compute gradients
    for inputs, labels in dataloader:
        inputs = inputs.view(inputs.size(0), -1)  # Flatten inputs
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # Save gradients to a file
        gradients = [param.grad.clone() for param in model.parameters()]
        torch.save(gradients, "gradients.pt")
        print("Gradients saved to gradients.pt")
        break  # Stop after one batch for demonstration

if __name__ == "__main__":
    compute_gradients()
