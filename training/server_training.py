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

# helpers for nice logs and eval
def _count_params(state):
    return sum(t.numel() for t in state.values()) # count the number of parameters in the model

def _l2_delta(prev, curr):
    # calculate how much the weights changed between rounds
    return torch.sqrt(sum((curr[k] - prev[k]).float().pow(2).sum() for k in curr)) # calculate the L2 norm of the weight changes

def eval_acc(state, loader, device):
    model = SmallCNN(pretrained=False)
    model.load_state_dict(state)
    model.to(device).eval() # evaluate the model
    correct = total = 0 # initialize the correct and total counts
    with torch.no_grad(): # disable gradient computation
        for x, y in loader: # iterate through the test set
            x, y = x.to(device), y.to(device) # move the data to the device
            pred = model(x).argmax(1) # get the predicted class
            correct += (pred == y).sum().item() # count the number of correct predictions
            total += y.size(0) # count the number of total predictions
    return correct / total # return the accuracy

def client_handler(idx, sock, addr, barrier, recv_list): # each client gets its own thread to handle communication
    print(f"[client {idx}] connected from {addr}", flush=True) # print the client connection information
    while True:
        barrier.wait()  # wait until server says 'broadcast now'
        send_pkl(sock, GLOBAL_STATE) # send the current global model to the client
        updated = recv_pkl(sock) # wait for the client to train and send back updated weights
        recv_list[idx] = updated # store the updated weights
        print(f"[client {idx}] received local update ({_count_params(updated):,} params)", flush=True)
        barrier.wait()  # tell server 'this client is done for the round'

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=12345)
    p.add_argument('--devices', type=int, default=2)
    p.add_argument('--rounds', type=int, default=20)
    args = p.parse_args()

    # listen & accept N devices
    # set up the server socket to listen for clients
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('', args.port))
    server.listen(args.devices)
    print(f"[server] listening on 0.0.0.0:{args.port} — waiting for {args.devices} device(s)...", flush=True)

    # wait for all the clients to connect before we start
    socks, addrs = [], []
    for i in range(args.devices):
        conn, addr = server.accept()
        socks.append(conn); addrs.append(addr)
        print(f"[server] device {i} accepted from {addr}", flush=True)

    print(f"[server] all {args.devices} devices connected. starting federated rounds.\n", flush=True)

    # datasets for quick validation
    tx = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    val_loader = DataLoader(datasets.MNIST('./data', train=False, download=True, transform=tx), batch_size=256, shuffle=False)

    global GLOBAL_STATE # global model init
    GLOBAL_STATE = SmallCNN(pretrained=True).state_dict() # start with a pretrained model that already works pretty well
    print(f"[server] initial global model ready ({_count_params(GLOBAL_STATE):,} params)\n", flush=True)

    # barrier helps coordinate all the threads so they stay in sync
    barrier = threading.Barrier(args.devices + 1)  # +1 for main thread
    updates = [None] * args.devices

    # spin up a thread for each client connection
    for idx, (sock, addr) in enumerate(zip(socks, addrs)):
        threading.Thread(target=client_handler,
                         args=(idx, sock, addr, barrier, updates),
                         daemon=True).start()

    device = 'cpu'  # evaluation
    # main federated learning loop
    for r in range(args.rounds):
        print(f"──────────────────────────────── Round {r:02d} ────────────────────────────────", flush=True)
        t0 = time.time()
        prev_state = {k: v.clone() for k, v in GLOBAL_STATE.items()} # keep a copy of the old weights so we can see how much they change

        print(f"[server] broadcasting W_k to {args.devices} client(s)...", flush=True)
        barrier.wait()   # release clients -> each will send update back
        print("[server] waiting for client updates...", flush=True)
        barrier.wait()   # all clients finished
        print("[server] aggregating (FedAvg)...", flush=True)

        GLOBAL_STATE = average_state_dicts(updates)
        torch.save(GLOBAL_STATE, os.path.join(CKPT_DIR, f"round_{r}.pth")) # save a checkpoint in case we want to analyze it later
        
        dW = _l2_delta(prev_state, GLOBAL_STATE).item() # calculate the L2 norm of the weight changes
        acc = eval_acc(GLOBAL_STATE, val_loader, device=device) # evaluate the accuracy
        dt = time.time() - t0 # calculate the time taken
        
        print(f"[server] round {r:02d} complete | ΔW L2 = {dW:.2f} | val acc = {acc:.2%} | time = {dt:.1f}s\n", flush=True) # print the results
    print(f"[server] training done. checkpoints saved in {CKPT_DIR}", flush=True) # print the checkpoint directory

if __name__ == '__main__':
    main()
