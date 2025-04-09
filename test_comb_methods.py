import torch
import torchvision
import torchvision.transforms as transforms
from cnn_model import SmallCNN  # Make sure cnn_model.py is in your PYTHONPATH
import socket
import pickle
import time

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
def send_to_raspberry_pi(client_socket, image_tensor, label):
    """
    Serializes the image tensor and label, sends them to the Raspberry Pi server,
    and then receives updated model weights.
    """
    # Serialize image and label as a tuple
    data = pickle.dumps((image_tensor.numpy(), label.numpy()))
    data_size = len(data)
    print(f"📤 Sending {data_size} bytes of CIFAR-10 image data...")
    
    # First send the size of the data
    client_socket.sendall(data_size.to_bytes(8, "big"))
    
    # Then send the data itself in chunks
    chunk_size = 4096
    for i in range(0, data_size, chunk_size):
        client_socket.sendall(data[i:i+chunk_size])
        # Optionally log progress:
        # print(f"✅ Sent {min(i+chunk_size, data_size)}/{data_size} bytes")
    
    print("✅ Finished sending image. Waiting for model updates...")
    
    # Receive the size of the updated model weights
    size_data = client_socket.recv(8)
    if not size_data:
        print("⚠️ Did not receive updated model size.")
        return None
    model_size = int.from_bytes(size_data, "big")
    
    # Now receive the model data
    received_data = b""
    while len(received_data) < model_size:
        chunk = client_socket.recv(min(chunk_size, model_size - len(received_data)))
        if not chunk:
            print("⚠️ Connection lost while receiving model update.")
            return None
        received_data += chunk
    
    updated_weights = pickle.loads(received_data)
    print("✅ Received updated model weights.")
    return updated_weights

# -----------------------------
# Main training function (client side)
# -----------------------------
def run_training():
    model = SmallCNN(pretrained=True)
    model.eval()
    
    final_weights = None

    try:
        while True:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
                client_socket.connect(("192.168.4.171", 12345))
                print("🔗 Connected to Raspberry Pi server.")

                while True:
                    print("🔄 Sending CIFAR-10 image to Raspberry Pi for training...")
                    updated_weights = send_to_raspberry_pi(client_socket, image, label)
                    
                    if updated_weights is None:
                        print("⚠️ Did not receive update. Restarting connection...")
                        break
                    
                    # Update the local model with the new weights
                    model.load_state_dict(updated_weights)
                    model.eval()
                    final_weights = updated_weights  # Save the latest weights
                    print("✅ Model updated with new weights. Restarting process...\n")
                    
                    time.sleep(5)
    except KeyboardInterrupt:
        print("⏹️ Training interrupted by user. Final model weights are returned.")
        return final_weights

if __name__ == "__main__":
    weights = run_training()
    if weights is not None:
        print("Final model weights:")
        for key in weights:
            print(key, weights[key].shape)

