import socket
import pickle
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torchvision.models import ResNet18_Weights
from comb_methods import combined_gradient_matching
from inversefed import utils, consts
from PIL import Image
from torchvision import transforms
import numpy as np

# Step 1: System Setup
setup = utils.system_startup()

# Load normalization constants for ImageNet
dm = torch.as_tensor(consts.imagenet_mean, **setup)[:, None, None]
ds = torch.as_tensor(consts.imagenet_std, **setup)[:, None, None]

# **Helper function to plot images**
def plot(tensor, title, save_path=None):
    tensor = tensor.clone().detach()
    tensor.mul_(ds).add_(dm).clamp_(0, 1)
    tensor_to_plot = tensor[0].permute(1, 2, 0).cpu()
    plt.imshow(tensor_to_plot)
    plt.title(title)
    if save_path:
        save_image(tensor, save_path)
    plt.show()

# **Function to send gradients to Raspberry Pi and receive processed gradients**
def send_to_raspberry_pi(gradients, server_ip="192.168.4.171", port=12345):
    """
    Sends gradients to Raspberry Pi, keeps the connection open for continuous training.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect((server_ip, port))

        # Log input gradients before sending
        print("📊 Input Gradients Before Sending:")
        for i, grad in enumerate(gradients):
            print(f"Gradient {i}: {grad.shape}")

        # Serialize gradients
        serialized_gradients = pickle.dumps(gradients)
        data_size = len(serialized_gradients)
        client_socket.sendall(data_size.to_bytes(8, "big"))  # Send size first

        # Send in chunks
        chunk_size = 4096
        for i in range(0, data_size, chunk_size):
            client_socket.sendall(serialized_gradients[i:i + chunk_size])

        print("✅ Finished sending gradients. Waiting for response...")

        # Receive processed gradient size
        size_data = client_socket.recv(8)
        processed_size = int.from_bytes(size_data, "big")

        # Receive processed gradients
        data = b""
        while len(data) < processed_size:
            chunk = client_socket.recv(min(chunk_size, processed_size - len(data)))
            if not chunk:
                raise ConnectionError("Connection lost while receiving processed gradients.")
            data += chunk

    processed_gradients = pickle.loads(data)
    print(f"✅ Received processed gradients: {[pg.shape for pg in processed_gradients]}")
    return [torch.tensor(g, requires_grad=False) for g in processed_gradients]

# **Main training function**
def run_training():
    """
    Automates the training process:
    - Computes initial gradients
    - Sends to Raspberry Pi
    - Runs 100 training iterations
    - Sends updated gradients back
    - Repeats indefinitely
    """
    model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.to(**setup)
    model.eval()

    # Load the input image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open("images/11794_ResNet18_ImageNet_input.png").convert("RGB")
    ground_truth = transform(image).unsqueeze(0).to(**setup)

    # Target label
    label = torch.tensor([243], device=setup['device'])  # German Shepherd class
    plot(ground_truth, f"Ground Truth (Label: {label})", "11794_input_image.png")

    while True:  # Infinite loop to keep training
        print("🔄 Starting new training cycle...")

        # Compute initial gradients
        model.zero_grad()
        target_loss = torch.nn.functional.cross_entropy(model(ground_truth), label)
        pred = model(ground_truth).softmax(dim=1)
        print(f"Model Prediction Probabilities: {pred[0][:10]}")
        print(f"Loss Value: {target_loss.item()}")
        
        input_gradient = torch.autograd.grad(target_loss, model.parameters())
        input_gradient = [grad.detach().numpy() for grad in input_gradient]

        # Send gradients to Raspberry Pi and receive processed gradients
        processed_gradients = send_to_raspberry_pi(input_gradient)

        # Run 100 iterations of training
        print("🚀 Running 100 iterations of training...")
        dummy_data, dummy_label = combined_gradient_matching(
            model=model,
            origin_grad=processed_gradients, 
            use_tv=True
        )

        # Save and visualize reconstructed image
        plot(dummy_data, "Reconstructed (Combined)", "11794_Combined_output.png")
        print("✅ Reconstructed image saved successfully. Restarting process...\n")

# **Run the training process**
if __name__ == "__main__":
    run_training()
