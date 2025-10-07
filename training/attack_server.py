#!/usr/bin/env python3
import os, sys, socket, pickle, torch

HOST, PORT = "0.0.0.0", 12346  # listen on server_port + 1

def recv_pkl(conn):
    size_bytes = conn.recv(8) # Read the 8-byte size header
    if not size_bytes:
        raise EOFError("socket closed before size header")

    size = int.from_bytes(size_bytes, "big") # Decode size and then read exactly that many bytes
    buf = b""
    while len(buf) < size:
        # Read in chunks until we have the full payload
        chunk = conn.recv(min(4096, size - len(buf)))
        if not chunk:
            # If we didn't get a full chunk, throw an error
            raise EOFError("socket closed mid-payload")
        buf += chunk
    # Turn bytes back into a Python object (dict with our gradient)    
    return pickle.loads(buf)

def main():
    # Start a TCP server that accepts exactly one connection/payload
    print(f"[attack/label] listening on {HOST}:{PORT}")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # Allow quick restart if we stop/start the script
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Bind and listen (blocking accept below)
        s.bind((HOST, PORT))
        s.listen(1)

        # Wait for the leaking client to connect and send the payload
        conn, addr = s.accept()
        print(f"[attack/label] got payload from {addr}")

        # Receive the pickled dict from the client
        msg = recv_pkl(conn)

        # We don't expect more than one payload, so close the connection
        conn.close()

    # Convert to a torch tensor (handles list or tensor values)
    val = msg["fc2_bias_grad"]
    g = val.detach().float() if torch.is_tensor(val) else torch.tensor(val, dtype=torch.float32)  # shape [10]

    # For cross-entropy + softmax, with batch size = 1:
    # The most negative entry in the bias gradient typically corresponds to the true label.
    inferred = int(torch.argmin(g).item())

    # Log the raw 10 numbers and our inferred digit
    print("[attack/label] fc2.bias grad:", g.tolist())
    print(f"[attack/label] inferred label = {inferred}")

    # Save the raw payload to disk so you can show it later
    os.makedirs("results", exist_ok=True)
    with open("results/label_leak_payload.pkl", "wb") as f:
        pickle.dump(msg, f)
    print("[attack/label] saved payload to results/label_leak_payload.pkl")

if __name__ == "__main__":
    main()