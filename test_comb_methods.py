import torch
import torchvision
import torchvision.transforms as transforms
from cnn_model import SmallCNN  # Import SmallCNN model
import socket
import pickle
import time

# **Load CIFAR-10 Dataset**
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

cifar10_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
image, label = cifar10_dataset[0]  # Get the first image & label
image = image.unsqueeze(0)  # Add batch dimension
label = torch.tensor([label], dtype=torch.long)  # Ensure proper label format

# **Function to send image & label to Raspberry Pi**
def send_to_raspberry_pi(client_socket, image_tensor, label):
    """ Sends image tensor and label to Raspberry Pi for local training. """
    data = pickle.dumps((image_tensor.numpy(), label.numpy()))
    data_size = len(data)

    client_socket.sendall(data_size.to_bytes(8, "big"))

    chunk_size = 4096
    for i in range(0, data_size, chunk_size):
        client_socket.sendall(data[i:i+chunk_size])
    
    print("✅ Finished sending CIFAR-10 image. Waiting for model updates...")

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
    model = SmallCNN(pretrained=True)  # ✅ Load with pretrained CIFAR-10 weights
    model.eval()

    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
                client_socket.connect(("192.168.4.171", 12345))
                print("🔗 Connected to Raspberry Pi server.")

                while True:
                    print("🔄 Sending CIFAR-10 image to Raspberry Pi for training...")
                    updated_model_weights = send_to_raspberry_pi(client_socket, image, label)
                    
                    if updated_model_weights is None:
                        print("⚠️ Connection lost, skipping update...")
                        continue  

                    model.load_state_dict(updated_model_weights)
                    model.eval()  # ✅ Set model to evaluation mode after update
                    print("✅ Model updated with new weights. Restarting process...\n")

                    time.sleep(5)

        except Exception as e:
            print(f"❌ Error: {e}. Restarting in 5 seconds...")
            time.sleep(5)

if __name__ == "__main__":
    run_training()
