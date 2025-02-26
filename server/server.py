import os
import torch
import torch.nn as nn
import torch.optim as optim
import socket
import pickle
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cnn_model import SmallCNN

HOST = "0.0.0.0"
PORT = 12345
BUFFER_SIZE = 4096
LEARNING_RATE = 0.01

# Correct path to pretrained weights
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Moves up one level
WEIGHTS_PATH = os.path.join(BASE_DIR, "cifar10_pretrained.pth")

# **Initialize model with pretrained CIFAR-10 weights**
model = SmallCNN(pretrained=True)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

def train_on_edge_device(image, label):
    global model, optimizer

    # Forward pass
    output = model(image)
    loss = criterion(output, label)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("✅ Training step complete. Returning updated weights.")

    return model.state_dict()  # Send updated model weights

def start_server():
    """Starts the Raspberry Pi server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        print(f"🚀 Server listening on {HOST}:{PORT}")

        while True:
            conn, addr = server_socket.accept()
            print(f"🔗 Connected to client at {addr}")

            try:
                while True:
                    size_data = conn.recv(8)
                    if not size_data:
                        print("❌ Connection lost.")
                        break  

                    data_size = int.from_bytes(size_data, "big")
                    data = b""
                    while len(data) < data_size:
                        chunk = conn.recv(min(BUFFER_SIZE, data_size - len(data)))
                        if not chunk:
                            break
                        data += chunk

                    image, label = pickle.loads(data)
                    print("📥 Received CIFAR-10 image. Starting training...")

                    updated_weights = train_on_edge_device(torch.tensor(image), torch.tensor(label))

                    serialized_response = pickle.dumps(updated_weights)
                    conn.sendall(len(serialized_response).to_bytes(8, "big"))
                    conn.sendall(serialized_response)

                    print("✅ Updated weights sent back.")

            except Exception as e:
                print(f"❌ Error: {e}")
                break  

if __name__ == "__main__":
    start_server()
