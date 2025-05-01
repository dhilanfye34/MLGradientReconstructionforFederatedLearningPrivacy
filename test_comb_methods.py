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
# Load MNIST Dataset
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
image, label = mnist_dataset[0]
image = image.unsqueeze(0)
label = torch.tensor([label], dtype=torch.long)

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
        print(f"📤 Sending {data_size} bytes of MNIST image data...")
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
    print(f"👀 Sending label: {label.item()}")
    from matplotlib import pyplot as plt

# Save and show original input image
    unnorm = (image * 0.5 + 0.5).clamp(0, 1)  # Undo normalization
    save_image(unnorm, "results/original_sent_image.png")

    plt.imshow(unnorm.squeeze().numpy(), cmap="gray")
    plt.title(f"Original Image (Label: {label.item()})")
    plt.axis("off")
    plt.show()
    reconstructed = send_to_raspberry_pi(image, label)

    if reconstructed is not None:
        # Un-normalize before saving
        mean = torch.tensor([0.5]).view(1, 1, 1, 1)
        std = torch.tensor([0.5]).view(1, 1, 1, 1)
        reconstructed_img = (reconstructed * std + mean).clamp(0, 1)

        save_path = "results/reconstructed_from_board.png"
        save_image(reconstructed_img, save_path)
        print(f"🖼️ Reconstructed image saved to: {save_path}")
    else:
        print("❌ No image reconstructed.")
