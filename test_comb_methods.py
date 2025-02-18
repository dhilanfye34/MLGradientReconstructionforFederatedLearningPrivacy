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
import time

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
def send_to_raspberry_pi(client_socket, image_tensor, label):
    """ Sends image tensor and label to Raspberry Pi for local training. """
    # Serialize image tensor & label
    data = pickle.dumps((image_tensor.numpy(), label.numpy()))
    data_size = len(data)
    print(f"📤 Sending {data_size} bytes of image data...")

    # Send data size first
    client_socket.sendall(data_size.to_bytes(8, "big"))

    # Send data in chunks
    sent_bytes = 0
    chunk_size = 4096
    for i in range(0, data_size, chunk_size):
        client_socket.sendall(data[i:i+chunk_size])
        sent_bytes += min(4096, data_size - sent_bytes)
        print(f"✅ Sent {sent_bytes}/{data_size} bytes...")

    print("✅ Finished sending image. Waiting for model updates...")

    # Receive processed model size
    size_data = client_socket.recv(8)
    model_size = int.from_bytes(size_data, "big")

    # Receive the updated model
    model_data = b""
    while len(model_data) < model_size:
        chunk = client_socket.recv(min(4096, model_size - len(model_data)))
        if not chunk:
            print("⚠️ Connection lost while receiving model update.")
            return None
        model_data += chunk

    updated_model_weights = pickle.loads(model_data)
    print("✅ Received updated model weights.")

    return updated_model_weights

# **Main training function**
def run_training():
    """ Sends images to Raspberry Pi for local training and updates local model with received weights. """
    model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.to(**setup)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open("images/11794_ResNet18_ImageNet_input.png").convert("RGB")
    ground_truth = transform(image).unsqueeze(0).to(**setup)

    label = torch.tensor([243], device=setup['device'])
    plot(ground_truth, f"Ground Truth (Label: {label})", "11794_input_image.png")

    while True:  # ✅ Loop indefinitely
        try:
            # **Persistent connection to Raspberry Pi**
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
                client_socket.connect(("192.168.4.171", 12345))
                print("🔗 Connected to Raspberry Pi server.")

                while True:  # ✅ Inner loop for continuous updates
                    print("🔄 Sending image to Raspberry Pi for training...")

                    updated_model_weights = send_to_raspberry_pi(client_socket, ground_truth, label)
                    if updated_model_weights is None:
                        print("⚠️ Connection lost, restarting...")
                        break  # If connection is lost, restart

                    # ✅ Load the updated model weights
                    model.load_state_dict(updated_model_weights)
                    print("✅ Model updated with new weights. Restarting process...\n")

                    time.sleep(5)  # ✅ Small delay before sending the next round

        except Exception as e:
            print(f"❌ Error: {e}. Restarting...")
            time.sleep(5)  # ✅ Prevent rapid restart spam

# **Run the training process**
if __name__ == "__main__":
    run_training()
