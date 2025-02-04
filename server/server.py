import socket
import pickle
import numpy as np
import time

HOST = "0.0.0.0"  # Listen on all interfaces
PORT = 12345
BUFFER_SIZE = 4096  # Safe chunk size for data transfer

import numpy as np

def compute_gradients_numpy(received_gradients):
    """
    Processes incoming gradients using NumPy and returns updated gradients.
    - Ensures shapes remain the same.
    - Simulates an ML update by applying a minor scaling factor.
    """
    processed_gradients = []
    
    for grad in received_gradients:
        grad_np = np.array(grad, dtype=np.float32)  # Ensure float32 for efficiency

        # Simulate an update (like a weight update in backpropagation)
        grad_np *= 0.9  # Scale down by 10% to simulate gradient descent

        processed_gradients.append(grad_np)

    return processed_gradients

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
            conn.settimeout(5)  # ✅ Set a timeout of 5 seconds for receiving data

            try:
                while True:
                    # Step 1: Receive data size
                    size_data = conn.recv(8)
                    if not size_data:
                        print("❌ Connection lost. Waiting for new client...")
                        break  

                    data_size = int.from_bytes(size_data, "big")
                    print(f"📩 Expecting {data_size} bytes from client...")

                    # Step 2: Receive the full data in chunks
                    data = b""
                    while len(data) < data_size:
                        chunk = conn.recv(min(BUFFER_SIZE, data_size - len(data)))
                        if not chunk:
                            print("⚠️ Connection lost while receiving data. Retrying...")
                            break
                        data += chunk
                        print(f"✅ Received {len(data)}/{data_size} bytes...")

                    if len(data) < data_size:
                        print(f"⚠️ Error: Incomplete data received ({len(data)} bytes instead of {data_size}).")
                        break

                    # Step 3: Deserialize gradients
                    gradients = pickle.loads(data)
                    print(f"📊 Received {len(gradients)} gradients.")

                    # Step 4: Process gradients
                    processed_gradients = compute_gradients_numpy(gradients)

                    # Step 5: Serialize processed gradients
                    serialized_response = pickle.dumps(processed_gradients)
                    response_size = len(serialized_response)

                    # Step 6: Send the size of the processed gradients first
                    conn.sendall(response_size.to_bytes(8, "big"))

                    # Step 7: Send processed gradients in chunks
                    sent_bytes = 0
                    for i in range(0, response_size, BUFFER_SIZE):
                        conn.sendall(serialized_response[i:i+BUFFER_SIZE])
                        sent_bytes += min(BUFFER_SIZE, response_size - sent_bytes)
                        print(f"✅ Sent {sent_bytes}/{response_size} bytes...")

                    print("✅ Processed gradients sent successfully!")

            except socket.timeout:
                print("⚠️ Timeout: No data received for 5 seconds. Reconnecting...")

            except Exception as e:
                print(f"❌ Error: {e}")
                break  

            print("🔌 Client disconnected. Waiting for new connection...\n")

if __name__ == "__main__":
    start_server()
