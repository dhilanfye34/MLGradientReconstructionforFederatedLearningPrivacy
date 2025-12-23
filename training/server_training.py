#!/usr/bin/env python3
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
import socket, threading, argparse, torch
from cnn_model import SmallCNN
from training.utils import send_pkl, recv_pkl, average_state_dicts
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True) 

def _count_params(state):
    return sum(t.numel() for t in state.values())

def _l2_delta(prev, curr):
    return torch.sqrt(sum((curr[k] - prev[k]).float().pow(2).sum() for k in curr)) # measure how much the weights changed

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
        barrier.wait() 
        send_pkl(sock, GLOBAL_STATE) # send the global model to the client
        updated = recv_pkl(sock) # get back the client's updated weights
        recv_list[idx] = updated
        print(f"[client {idx}] received local update ({_count_params(updated):,} params)", flush=True)
        barrier.wait()  

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=12345)
    p.add_argument('--devices', type=int, default=2)
    p.add_argument('--rounds', type=int, default=20)
    args = p.parse_args()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('', args.port)) 
    server.listen(args.devices)
    print(f"[server] listening on 0.0.0.0:{args.port} - waiting for {args.devices} device(s)...", flush=True)

    socks, addrs = [], []
    for i in range(args.devices):
        conn, addr = server.accept() 
        socks.append(conn); addrs.append(addr)
        print(f"[server] device {i} accepted from {addr}", flush=True)

    print(f"[server] all {args.devices} devices connected. starting federated rounds.\n", flush=True)

    tx = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]) # image norm
    val_loader = DataLoader(datasets.MNIST('./data', train=False, download=True, transform=tx), batch_size=256, shuffle=False)

    global GLOBAL_STATE 
    GLOBAL_STATE = SmallCNN(pretrained=True).state_dict() 
    print(f"[server] initial global model ready ({_count_params(GLOBAL_STATE):,} params)\n", flush=True)

    barrier = threading.Barrier(args.devices + 1) 
    updates = [None] * args.devices

    for idx, (sock, addr) in enumerate(zip(socks, addrs)): # threads for each client
        threading.Thread(target=client_handler, args=(idx, sock, addr, barrier, updates), daemon=True).start()

    device = 'cpu'  
    # main federated learning loop
    for r in range(args.rounds):
        print(f"──────────────────────────────── Round {r:02d} ────────────────────────────────", flush=True)
        t0 = time.time()
        prev_state = {k: v.clone() for k, v in GLOBAL_STATE.items()} # keep a copy of the old weights so we can see how much they change

        print(f"[server] broadcasting W_k to {args.devices} client(s)...", flush=True)
        barrier.wait()  
        print("[server] waiting for client updates...", flush=True)
        barrier.wait()  
        print("[server] aggregating for FedAvg...", flush=True)

        GLOBAL_STATE = average_state_dicts(updates) # combine all client updates into one global model
        torch.save(GLOBAL_STATE, os.path.join(CKPT_DIR, f"round_{r}.pth")) 
        
        dW = _l2_delta(prev_state, GLOBAL_STATE).item() # calculate the L2 norm of the weight changes
        acc = eval_acc(GLOBAL_STATE, val_loader, device=device) 
        dt = time.time() - t0 # time
        
        print(f"[server] round {r:02d} complete | change in W L2 = {dW:.2f} | val acc = {acc:.2%} | time = {dt:.1f}s\n", flush=True)
    print(f"[server] training done. checkpoints saved in {CKPT_DIR}", flush=True)

if __name__ == '__main__':
    main()
