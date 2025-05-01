import os
import sys
import socket
import pickle
import torch
import torch.nn.functional as F
from torch.autograd import grad
from torchvision.utils import save_image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cnn_model import SmallCNN
from comb_methods import combined_gradient_matching  # <- Call the reconstruction

HOST = "0.0.0.0"
PORT = 12345
BUFFER_SIZE = 4096

model = SmallCNN(pretrained=True)
model.eval()

def reconstruct_image(image_tensor, label_tensor):
    image_tensor = image_tensor.to(torch.float32)
    label_tensor = label_tensor.to(torch.long)

    # Compute original gradients from real image
    image_tensor.requires_grad = True
    output = model(image_tensor)
    loss = F.cross_entropy(output, label_tensor)
    origin_grad = grad(loss, model.parameters(), create_graph=False)

    print("✅ Computed gradients from incoming image.")
    dummy_data, _ = combined_gradient_matching(model, origin_grad, label_tensor, switch_iteration=50)
    return dummy_data

def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        print(f"🚀 Server listening on {HOST}:{PORT}")

        while True:
            conn, addr = server_socket.accept()
            print(f"🔗 Connected to client at {addr}")

            try:
                size_data = conn.recv(8)
                data_size = int.from_bytes(size_data, "big")

                data = b""
                while len(data) < data_size:
                    data += conn.recv(min(BUFFER_SIZE, data_size - len(data)))

                image, label = pickle.loads(data)
                image_tensor = torch.tensor(image)
                label_tensor = torch.tensor(label)

                print("📥 Received image and label. Starting reconstruction...")
                dummy_data = reconstruct_image(image_tensor, label_tensor)

                # Serialize reconstructed image to send back
                serialized_image = pickle.dumps(dummy_data.cpu().detach())
                conn.sendall(len(serialized_image).to_bytes(8, "big"))
                conn.sendall(serialized_image)
                print("📤 Sent reconstructed image back to client.")

            except Exception as e:
                print(f"❌ Error: {e}")

if __name__ == "__main__":
    start_server()
