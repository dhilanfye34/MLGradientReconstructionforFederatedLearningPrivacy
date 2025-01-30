import socket
import pickle
import numpy as np

HOST = "0.0.0.0"
PORT = 12345

def process_gradients(gradients):
    """
    Fake function that processes received gradients.
    """
    print("Processing received gradients...")
    return [g * 0.9 for g in gradients]  # Example modification

def start_server():
    """
    Start the Raspberry Pi server.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((HOST, PORT))
        server_socket.listen()

        print(f"Server listening on {HOST}:{PORT}")

        while True:
            conn, addr = server_socket.accept()
            print(f"Connected to client at {addr}")

            try:
                # Receive data size first
                size_data = conn.recv(8)
                if not size_data:
                    print("Error: No data size received.")
                    continue

                data_size = int.from_bytes(size_data, "big")

                # Receive the full data in chunks
                data = b""
                chunk_size = 4096
                while len(data) < data_size:
                    chunk = conn.recv(min(chunk_size, data_size - len(data)))
                    if not chunk:
                        print("Error: Connection lost while receiving data.")
                        break
                    data += chunk

                # Deserialize the received data
                gradients = pickle.loads(data)
                print(f"Received {len(gradients)} gradients.")

                # Process gradients
                processed_gradients = process_gradients(gradients)

                # Serialize processed gradients
                serialized_response = pickle.dumps(processed_gradients)
                response_size = len(serialized_response)

                # Send processed gradients size first
                conn.sendall(response_size.to_bytes(8, "big"))

                # Send in chunks
                for i in range(0, response_size, chunk_size):
                    conn.sendall(serialized_response[i:i + chunk_size])

                print("Processed gradients sent back.")

            except Exception as e:
                print(f"Error: {e}")

            finally:
                conn.close()
                print("Connection closed.")

if __name__ == "__main__":
    start_server()
