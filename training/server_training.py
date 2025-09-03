#!/usr/bin/env python3
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import socket, threading, argparse, torch
from cnn_model import SmallCNN
from training.utils import send_pkl, recv_pkl, average_state_dicts
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------- helpers for nice logs ----------
def _count_params(state):
    return sum(t.numel() for t in state.values())

def _l2_delta(prev, curr):
    return torch.sqrt(sum((curr[k] - prev[k]).float().pow(2).sum() for k in curr))

def eval_acc(state, loader, device):
    model = SmallCNN(pretrained=False)
    model.load_state_dict(state)
    model.to(device).eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

def client_handler(idx, sock, addr, barrier, recv_list):
    print(f"[client {idx}] connected from {addr}", flush=True)
    while True:
        # main thread prints round info; here we just do the send/recv
        barrier.wait()  # wait until server says 'broadcast now'
        send_pkl(sock, GLOBAL_STATE)
        updated = recv_pkl(sock)
        recv_list[idx] = updated
        print(f"[client {idx}] received local update ({_count_params(updated):,} params)", flush=True)
        barrier.wait()  # tell server 'this client is done for the round'

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=12345)
    p.add_argument('--devices', type=int, default=2)
    p.add_argument('--rounds', type=int, default=20)
    args = p.parse_args()

    # --- listen & accept N devices
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('', args.port))
    server.listen(args.devices)
    print(f"[server] listening on 0.0.0.0:{args.port} — waiting for {args.devices} device(s)...", flush=True)

    socks, addrs = [], []
    for i in range(args.devices):
        conn, addr = server.accept()
        socks.append(conn); addrs.append(addr)
        print(f"[server] device {i} accepted from {addr}", flush=True)

    print(f"[server] all {args.devices} devices connected. starting federated rounds.\n", flush=True)

    # --- datasets for quick validation
    tx = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    val_loader = DataLoader(datasets.MNIST('./data', train=False, download=True, transform=tx),
                            batch_size=256, shuffle=False)

    # --- global model init
    global GLOBAL_STATE
    GLOBAL_STATE = SmallCNN(pretrained=True).state_dict()  # start from your pretrained 97% ckpt
    print(f"[server] initial global model ready ({_count_params(GLOBAL_STATE):,} params)\n", flush=True)

    barrier = threading.Barrier(args.devices + 1)  # +1 for main thread
    updates = [None] * args.devices

    for idx, (sock, addr) in enumerate(zip(socks, addrs)):
        threading.Thread(target=client_handler,
                         args=(idx, sock, addr, barrier, updates),
                         daemon=True).start()

    device = 'cpu'  # evaluation
    for r in range(args.rounds):
        print(f"──────────────────────────────── Round {r:02d} ────────────────────────────────", flush=True)
        t0 = time.time()
        prev_state = {k: v.clone() for k, v in GLOBAL_STATE.items()}

        print(f"[server] broadcasting W_k to {args.devices} client(s)...", flush=True)
        barrier.wait()   # release clients → each will send update back

        print("[server] waiting for client updates...", flush=True)
        barrier.wait()   # all clients finished

        print("[server] aggregating (FedAvg)...", flush=True)
        GLOBAL_STATE = average_state_dicts(updates)
        torch.save(GLOBAL_STATE, f'training/round_{r}.pth')

        dW = _l2_delta(prev_state, GLOBAL_STATE).item()
        acc = eval_acc(GLOBAL_STATE, val_loader, device=device)
        dt = time.time() - t0
        print(f"[server] round {r:02d} complete | ΔW L2 = {dW:.2f} | val acc = {acc:.2%} | time = {dt:.1f}s\n", flush=True)

    print("[server] training done. checkpoints saved as training/round_*.pth", flush=True)

if __name__ == '__main__':
    main()
