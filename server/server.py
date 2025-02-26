import torch
import torch.nn as nn
import torch.optim as optim
import socket
import pickle
from cnn_model import SmallCNN  # Import updated model

HOST = "0.0.0.0"
PORT = 12345
BUFFER_SIZE = 4096
LEARNING_RATE = 0.01

# ✅ Initialize model (ensure cnn_model.py allows pretrained loading)
model = SmallCNN(pretrained=True)
model.train()  # Set model to training mode
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

def train_on_edge_device(image, label):
    global model, optimizer

    # Convert image and label to tensors
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add batch dim
    label = torch.tensor(label, dtype=torch.long).unsqueeze(0)  # Add batch dim

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

                    # Deserialize image + label
                    image, label = pickle.loads(data)
                    print("📥 Received CIFAR-10 image. Starting training...")

                    # Train and get updated weights
                    updated_weights = train_on_edge_device(image, label)

                    # Serialize and send back weights
                    serialized_response = pickle.dumps(updated_weights)
                    conn.sendall(len(serialized_response).to_bytes(8, "big"))
                    conn.sendall(serialized_response)

                    print("✅ Updated weights sent back.")

            except Exception as e:
                print(f"❌ Error: {e}")
                break  

if __name__ == "__main__":
    start_server()
