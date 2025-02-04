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
                while True:  # ✅ Keep processing new gradients without disconnecting
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

                    gradients = pickle.loads(data)
                    processed_gradients = compute_gradients_numpy(gradients)

                    serialized_response = pickle.dumps(processed_gradients)
                    response_size = len(serialized_response)

                    conn.sendall(response_size.to_bytes(8, "big"))
                    conn.sendall(serialized_response)

                    print("✅ Processed gradients sent successfully!")
                    print("🔄 Ready for next batch of gradients...")  # ✅ Keep looping

            except Exception as e:
                print(f"❌ Error: {e}")
                break  

if __name__ == "__main__":
    start_server()
