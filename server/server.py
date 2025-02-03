import socket
import pickle
import numpy as np

HOST = "0.0.0.0"  # Listen on all interfaces
PORT = 12345
BUFFER_SIZE = 4096  # Safe chunk size for data transfer

def compute_gradients_numpy():
    """
    Simulate gradient computation with shapes matching the client's model.
    """
    expected_shapes = [
        (64, 3, 7, 7),  # First Conv Layer Weights
        (64,),           # First Conv Layer Bias
        (128, 64, 3, 3), # Second Conv Layer Weights
        (128,)           # Second Conv Layer Bias
    ]
    
    gradients = [np.random.randn(*shape).astype(np.float32) for shape in expected_shapes]
    
    print(f"🧮 Generated gradients with expected shapes:")
    for i, g in enumerate(gradients):
        print(f"Layer {i}: {g.shape}")
    
    return gradients

def start_server():
    """Starts the Raspberry Pi server to receive and process gradients."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        print(f"🚀 Server listening on {HOST}:{PORT}")

        while True:
            conn, addr = server_socket.accept()
            print(f"🔗 Connected to client at {addr}")

            try:
                # Step 1: Receive data size
                size_data = conn.recv(8)
                if not size_data:
                    print("❌ Error: No data size received.")
                    continue
                data_size = int.from_bytes(size_data, "big")
                print(f"📩 Expecting {data_size} bytes from client...")

                # Step 2: Receive the full data in chunks
                data = b""
                while len(data) < data_size:
                    chunk = conn.recv(min(BUFFER_SIZE, data_size - len(data)))
                    if not chunk:
                        print("⚠️ Error: Connection lost while receiving data.")
                        break
                    data += chunk
                    print(f"✅ Received {len(data)}/{data_size} bytes...")

                if len(data) < data_size:
                    print(f"⚠️ Error: Incomplete data received ({len(data)} bytes instead of {data_size}).")
                    continue

                # Step 3: Deserialize gradients
                gradients = pickle.loads(data)
                print(f"📊 Received {len(gradients)} gradients.")

                # Log received gradient shapes
                for i, g in enumerate(gradients):
                    print(f"Client Gradient {i}: Shape = {g.shape}")

                # Step 4: Process gradients (simulate computation)
                processed_gradients = compute_gradients_numpy()

                # Step 5: Validate shapes before sending back
                for i, (client_grad, processed_grad) in enumerate(zip(gradients, processed_gradients)):
                    if client_grad.shape != processed_grad.shape:
                        print(f"⚠️ Mismatch: Client {client_grad.shape} != Processed {processed_grad.shape}")
                    else:
                        print(f"✅ Matched: {client_grad.shape}")

                # If there's a mismatch, return the client gradients instead
                processed_gradients = [
                    client_grad if client_grad.shape != processed_grad.shape else processed_grad
                    for client_grad, processed_grad in zip(gradients, processed_gradients)
                ]

                # Step 6: Serialize and send processed gradients
                serialized_response = pickle.dumps(processed_gradients)
                response_size = len(serialized_response)

                conn.sendall(response_size.to_bytes(8, "big"))
                print(f"📤 Sending {response_size} bytes of processed gradients back to client...")

                sent_bytes = 0
                for i in range(0, response_size, BUFFER_SIZE):
                    conn.sendall(serialized_response[i:i+BUFFER_SIZE])
                    sent_bytes += min(BUFFER_SIZE, response_size - sent_bytes)
                    print(f"✅ Sent {sent_bytes}/{response_size} bytes...")

                print("✅ Processed gradients sent successfully!")

            except Exception as e:
                print(f"❌ Error: {e}")

            finally:
                conn.close()
                print("🔌 Connection closed.\n")

if __name__ == "__main__":
    start_server()
