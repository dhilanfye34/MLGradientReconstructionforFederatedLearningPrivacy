import socket
import pickle
import numpy as np

HOST = "0.0.0.0"
PORT = 12345
BUFFER_SIZE = 4096

def compute_gradients_numpy(received_gradients):
    """ Simulates ML updates by modifying received gradients. """
    processed_gradients = []
    for grad in received_gradients:
        grad_np = np.array(grad, dtype=np.float32)
        grad_np *= 0.9  # Simulates gradient update
        processed_gradients.append(grad_np)
    return processed_gradients

def start_server():
    """ Keeps the server running without dropping connections. """
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
                    # Receive data size
                    size_data = conn.recv(8)
                    if not size_data:
                        print("⚠️ No data received. Waiting for next batch...")
                        continue  # Instead of breaking, wait for more data

                    data_size = int.from_bytes(size_data, "big")
                    print(f"📩 Expecting {data_size} bytes...")

                    # Receive data
                    data = b""
                    while len(data) < data_size:
                        chunk = conn.recv(min(BUFFER_SIZE, data_size - len(data)))
                        if not chunk:
                            print("⚠️ Connection lost while receiving. Waiting for next batch...")
                            continue
                        data += chunk

                    if len(data) < data_size:
                        print("⚠️ Incomplete data received. Waiting for next batch...")
                        continue

                    # Deserialize gradients
                    gradients = pickle.loads(data)
                    print(f"📊 Received {len(gradients)} gradients.")

                    # Process gradients
                    processed_gradients = compute_gradients_numpy(gradients)

                    # Send back processed gradients
                    serialized_response = pickle.dumps(processed_gradients)
                    response_size = len(serialized_response)

                    conn.sendall(response_size.to_bytes(8, "big"))
                    conn.sendall(serialized_response)
                    print(f"✅ Sent processed gradients back. Waiting for next batch...")

            except Exception as e:
                print(f"❌ Server Error: {e}")
                continue  # Restart loop to keep the server running

if __name__ == "__main__":
    start_server()
