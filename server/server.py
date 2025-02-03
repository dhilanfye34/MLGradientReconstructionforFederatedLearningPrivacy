import socket
import pickle
import numpy as np

HOST = "0.0.0.0"  # Listen on all interfaces
PORT = 12345
BUFFER_SIZE = 4096  # Safe chunk size for data transfer

def compute_gradients_numpy():
    """Simulate gradient computation for layers in a neural network using NumPy."""
    conv1_weights = np.random.randn(64, 3, 7, 7)  
    conv1_bias = np.random.randn(64)              
    conv2_weights = np.random.randn(128, 64, 3, 3)  
    conv2_bias = np.random.randn(128)             

    gradients = [conv1_weights, conv1_bias, conv2_weights, conv2_bias]
    return gradients

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

            while True:  # Keep receiving and processing gradients in a loop
                try:
                    # Step 1: Receive data size
                    size_data = conn.recv(8)
                    if not size_data:
                        print("❌ Connection lost. Waiting for new client...")
                        break  # Exit loop, wait for new client

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
                        break

                    # Step 3: Deserialize gradients
                    gradients = pickle.loads(data)
                    print(f"📊 Received {len(gradients)} gradients.")

                    # Step 4: Process gradients
                    processed_gradients = compute_gradients_numpy()

                    # Step 5: Serialize processed gradients
                    serialized_response = pickle.dumps(processed_gradients)
                    response_size = len(serialized_response)

                    # Step 6: Send the size of the processed gradients first
                    conn.sendall(response_size.to_bytes(8, "big"))
                    print(f"📤 Sending {response_size} bytes of processed gradients back to client...")

                    # Step 7: Send the processed gradients in chunks
                    sent_bytes = 0
                    for i in range(0, response_size, BUFFER_SIZE):
                        conn.sendall(serialized_response[i:i+BUFFER_SIZE])
                        sent_bytes += min(BUFFER_SIZE, response_size - sent_bytes)
                        print(f"✅ Sent {sent_bytes}/{response_size} bytes...")

                    print("✅ Processed gradients sent successfully!")

                except Exception as e:
                    print(f"❌ Error: {e}")
                    break  # Restart the loop if an error occurs

            print("🔌 Client disconnected. Waiting for new connection...\n")

if __name__ == "__main__":
    start_server()
