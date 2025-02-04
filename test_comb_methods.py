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
def send_to_raspberry_pi(ground_truth, label, model, server_ip="192.168.4.171", port=12345):
    """ Keeps sending gradients continuously without dropping connection. """
    MAX_RETRIES = 3
    retries = 0

    while retries < MAX_RETRIES:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
                client_socket.connect((server_ip, port))
                print("🔗 Connected to Raspberry Pi server.")

                while True:  # ✅ Keep sending gradients continuously
                    # Compute new gradients before sending
                    model.zero_grad()
                    target_loss = torch.nn.functional.cross_entropy(model(ground_truth), label)
                    input_gradient = torch.autograd.grad(target_loss, model.parameters())
                    input_gradient = [grad.detach().numpy() for grad in input_gradient]

                    # Serialize and send gradients
                    serialized_gradients = pickle.dumps(input_gradient)
                    data_size = len(serialized_gradients)

                    print(f"📤 Sending {data_size} bytes of gradients...")

                    # Send data size first
                    client_socket.sendall(data_size.to_bytes(8, "big"))

                    # ✅ Send in explicit chunks & log each one
                    sent_bytes = 0
                    chunk_size = 4096
                    for i in range(0, data_size, chunk_size):
                        chunk = serialized_gradients[i:i+chunk_size]
                        client_socket.sendall(chunk)
                        sent_bytes += len(chunk)
                        print(f"✅ Sent {sent_bytes}/{data_size} bytes...")  # Debugging

                    print("✅ Finished sending gradients. Waiting for processed gradients...")

                    # Receive processed gradient size
                    size_data = client_socket.recv(8)
                    if not size_data:
                        print("⚠️ Connection lost while waiting for processed gradients. Retrying...")
                        break

                    processed_size = int.from_bytes(size_data, "big")

                    # Ensure full response is received
                    data = b""
                    while len(data) < processed_size:
                        chunk = client_socket.recv(min(4096, processed_size - len(data)))
                        if not chunk:
                            print("⚠️ Connection lost while receiving. Retrying...")
                            break
                        data += chunk
                        print(f"✅ Received {len(data)}/{processed_size} bytes...")  # Debugging

                    if len(data) < processed_size:
                        print("⚠️ Incomplete data received. Retrying next batch...")
                        continue

                    processed_gradients = pickle.loads(data)
                    print(f"✅ Received processed gradients: {[pg.shape for pg in processed_gradients]}")

                    return [torch.tensor(g, requires_grad=False) for g in processed_gradients]

        except (socket.error, ConnectionError) as e:
            retries += 1
            print(f"⚠️ Connection failed ({e}). Retrying {retries}/{MAX_RETRIES}...")
            time.sleep(3)

    raise ConnectionError("❌ Failed to connect to Raspberry Pi after multiple attempts.")

# **Main training function**
def run_training():
    """ Starts training and keeps it running indefinitely. """
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

    # **Now it immediately starts sending gradients**
    send_to_raspberry_pi(ground_truth, label, model)

# **Run the training process**
if __name__ == "__main__":
    run_training()
