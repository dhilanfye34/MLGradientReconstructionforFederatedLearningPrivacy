import os, sys, socket, pickle, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cnn_model import SmallCNN
from comb_methods import combined_gradient_matching

HOST, PORT = "0.0.0.0", 12346

def recv_pkl(conn):
    size = int.from_bytes(conn.recv(8), "big")
    buf = b""
    while len(buf) < size:
        chunk = conn.recv(min(4096, size - len(buf)))
        if not chunk: raise EOFError("socket closed")
        buf += chunk
    return pickle.loads(buf)

def main():
    print(f"[attack] listening on {HOST}:{PORT}")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT)); s.listen(1)
        conn, addr = s.accept()
        print(f"[attack] got gradient from {addr}")
        msg = recv_pkl(conn)

    grads = [g.to(torch.float32) for g in msg["grads"]]
    label = int(msg["label"])  # single-label case

    model = SmallCNN(pretrained=True).eval()
    print("[attack] starting reconstruction…")
    dummy, _ = combined_gradient_matching(model, grads, label, switch_iteration=500)
    print("[attack] done. Check results/ for images.")

if __name__ == "__main__":
    main()