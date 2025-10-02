#!/usr/bin/env python3
import os, sys, socket, pickle, torch

HOST, PORT = "0.0.0.0", 12346  # listen on server_port + 1

def recv_pkl(conn):
    size_bytes = conn.recv(8)
    if not size_bytes:
        raise EOFError("socket closed before size header")
    size = int.from_bytes(size_bytes, "big")
    buf = b""
    while len(buf) < size:
        chunk = conn.recv(min(4096, size - len(buf)))
        if not chunk:
            raise EOFError("socket closed mid-payload")
        buf += chunk
    return pickle.loads(buf)

def main():
    print(f"[attack/label] listening on {HOST}:{PORT}")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT)); s.listen(1)
        conn, addr = s.accept()
        print(f"[attack/label] got payload from {addr}")
        msg = recv_pkl(conn)
        conn.close()

    # Expecting: {"fc2_bias_grad": tensor/list length 10, "note": "..."}
    g = torch.tensor(msg["fc2_bias_grad"], dtype=torch.float32)  # shape [10]
    inferred = int(torch.argmin(g).item())

    print("[attack/label] fc2.bias grad:", g.tolist())
    print(f"[attack/label] inferred label = {inferred}")

    os.makedirs("results", exist_ok=True)
    with open("results/label_leak_payload.pkl", "wb") as f:
        pickle.dump(msg, f)
    print("[attack/label] saved payload to results/label_leak_payload.pkl")

if __name__ == "__main__":
    main()
