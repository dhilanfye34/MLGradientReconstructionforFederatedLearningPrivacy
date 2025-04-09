import os
import torch
import torchvision
import torchvision.transforms as transforms
import socket
import pickle
from torchvision.utils import save_image

# -----------------------------
# Prepare output folder
# -----------------------------
os.makedirs("results", exist_ok=True)

# -----------------------------
# Load CIFAR-10 Dataset
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
cifar10_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
image, label = cifar10_dataset[0]  # Use the first image for this example
image = image.unsqueeze(0)  # Add a batch dimension
label = torch.tensor([label], dtype=torch.long)  # Ensure label is a tensor

# -----------------------------
# Function to send image & label to the Pi
# -----------------------------
def send_to_raspberry_pi(image_tensor, label_tensor):
    """
    Sends the image and label to the Pi, receives a reconstructed image tensor.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect(("192.168.4.171", 12345))  # <-- Replace with Pi IP if needed
        print("🔗 Connected to Raspberry Pi server.")

        # Serialize the image and label
        data = pickle.dumps((image_tensor.numpy(), label_tensor.numpy()))
        data_size = len(data)
        print(f"📤 Sending {data_size} bytes of CIFAR-10 image data...")
        client_socket.sendall(data_size.to_bytes(8, "big"))

        # Send data in chunks
        chunk_size = 4096
        for i in range(0, data_size, chunk_size):
            client_socket.sendall(data[i:i+chunk_size])

        print("✅ Finished sending image. Waiting for reconstructed image...")

        # Receive size of the incoming data
        size_data = client_socket.recv(8)
        if not size_data:
            print("⚠️ Failed to receive image size.")
            return None

        image_size = int.from_bytes(size_data, "big")
        print(f"📦 Expecting {image_size} bytes of reconstructed image...")

        # Receive image data
        image_data = b""
        while len(image_data) < image_size:
            chunk = client_socket.recv(min(chunk_size, image_size - len(image_data)))
            if not chunk:
                print("⚠️ Connection lost during image reception.")
                return None
            image_data += chunk

        print("✅ Reconstructed image received.")
        return pickle.loads(image_data)

# -----------------------------
# Main function
# -----------------------------
if __name__ == "__main__":
    reconstructed = send_to_raspberry_pi(image, label)

    if reconstructed is not None:
        # Un-normalize before saving
        mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        reconstructed_img = (reconstructed * std + mean).clamp(0, 1)

        save_path = "results/reconstructed_from_board.png"
        save_image(reconstructed_img, save_path)
        print(f"🖼️ Reconstructed image saved to: {save_path}")
    else:
        print("❌ No image reconstructed.")
