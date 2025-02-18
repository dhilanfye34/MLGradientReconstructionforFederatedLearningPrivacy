import socket
import pickle
import numpy as np

HOST = "0.0.0.0"
PORT = 12345
BUFFER_SIZE = 4096
LEARNING_RATE = 0.01  # Step size for updates

# Initialize model parameters (NumPy version of ResNet weights)
model_weights = {
    "W1": np.random.randn(64, 3, 7, 7),  # Example weights
    "b1": np.zeros((64,)),  # Bias
    "W2": np.random.randn(64, 64, 3, 3),
    "b2": np.zeros((64,)),
    "W3": np.random.randn(128, 64, 3, 3),
    "b3": np.zeros((128,)),
}

def train_on_edge_device(image, label):
    """
    Simulates training by applying simple gradient descent in NumPy.
    This assumes `image` is a NumPy array of shape (3, 224, 224) and `label` is an int.
    """
    global model_weights  # Access model parameters

    # Simulated forward pass: compute gradients
    gradients = {key: np.random.randn(*model_weights[key].shape) for key in model_weights}

    # Simulated gradient descent update
    for key in model_weights:
        model_weights[key] -= LEARNING_RATE * gradients[key]  # W_new = W - η * grad

    print("✅ Local training complete. Returning updated weights.")

    return model_weights  # Send updated model weights

def start_server():
    """Starts the Raspberry Pi server and keeps it running indefinitely."""
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
                    # Step 1: Receive Image and Label
                    size_data = conn.recv(8)
                    if not size_data:
                        print("❌ Connection lost. Waiting for new client...")
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
                    print("📥 Received image and label. Starting training on edge device...")

                    # Step 2: Perform local training (NumPy)
                    updated_weights = train_on_edge_device(image, label)

                    # Step 3: Send updated weights back
                    serialized_response = pickle.dumps(updated_weights)
                    response_size = len(serialized_response)

                    conn.sendall(response_size.to_bytes(8, "big"))
                    conn.sendall(serialized_response)

                    print("✅ Updated weights sent back to client. Ready for next batch.")

            except Exception as e:
                print(f"❌ Error: {e}")
                break  

if __name__ == "__main__":
    start_server()
