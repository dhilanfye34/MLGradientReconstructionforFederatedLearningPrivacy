import socket
import numpy as np

# Configuration
SERVER_HOST = "your.server.ip.address"  # Replace with the Raspberry Pi's IP address
SERVER_PORT = 65432                    # Port to connect to (must match the server)

def generate_dummy_gradients():
    """
    Simulates generating gradients on the client-side (MacBook or other device).
    Replace this with the actual gradient generation logic.
    """
    print("Generating dummy gradients...")
    dummy_gradients = np.random.rand(10).astype(np.float32)  # Example: 10 random gradient values
    return dummy_gradients.tobytes()

def send_and_receive_gradients():
    # Create a TCP socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        print(f"Connecting to server at {SERVER_HOST}:{SERVER_PORT}...")
        client_socket.connect((SERVER_HOST, SERVER_PORT))
        
        # Generate dummy gradients
        gradients = generate_dummy_gradients()
        
        # Send gradients to the server
        print("Sending gradients to the server...")
        client_socket.sendall(gradients)
        
        # Receive processed gradients from the server
        print("Waiting for processed gradients...")
        data = client_socket.recv(4096)
        
        # Convert the received data back to NumPy array
        processed_gradients = np.frombuffer(data, dtype=np.float32)
        print("Received processed gradients:", processed_gradients)

if __name__ == "__main__":
    send_and_receive_gradients()
