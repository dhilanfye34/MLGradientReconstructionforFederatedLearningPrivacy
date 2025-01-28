import socket
import pickle
from local_gradient_computation import compute_gradients_numpy  # Import the gradient computation function

HOST = "0.0.0.0"  # Server host (accept connections from any IP)
PORT = 12345      # Port for communication

def start_server():
    """
    Start the server to listen for client connections and handle requests.
    """
    # Set up the socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)  # Listen for a single connection

        print(f"Server listening on {HOST}:{PORT}")

        while True:
            # Accept a connection from a client
            client_socket, client_address = server_socket.accept()
            with client_socket:
                print(f"Connected to client at {client_address}")

                # Receive data from the client
                data = client_socket.recv(4096)  # Adjust buffer size as needed
                if not data:
                    break
                request = pickle.loads(data)  # Deserialize the client's request

                print(f"Received request: {request}")

                # Compute gradients using the imported function
                gradients = compute_gradients_numpy()

                # Serialize and send the computed gradients back to the client
                response = pickle.dumps(gradients)
                client_socket.sendall(response)
                print(f"Sent gradients back to client at {client_address}")

if __name__ == "__main__":
    start_server()
