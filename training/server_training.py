#!/usr/bin/env python3
import socket, threading, pickle, argparse, torch
from cnn_model import SmallCNN
from training.utils import send_pkl, recv_pkl, average_state_dicts
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def eval_acc(state, loader, device):
    model = SmallCNN(pretrained=False)
    model.load_state_dict(state)
    model.to(device).eval()
    correct = total = 0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

def client_handler(sock, barrier, recv_list, idx):
    while True:
        barrier.wait()               # wait for server to broadcast
        send_pkl(sock, GLOBAL_STATE)
        updated = recv_pkl(sock)
        recv_list[idx] = updated
        barrier.wait()               # tell server round finished

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=12345)
    p.add_argument('--devices', type=int, default=2)
    p.add_argument('--rounds',  type=int, default=20)
    args = p.parse_args()

    # --- listen & accept N devices
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('', args.port)); server.listen(args.devices)
    socks = [server.accept()[0] for _ in range(args.devices)]
    print(f'[✔] {args.devices} devices connected')

    # --- datasets for quick validation
    tx = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.5,),(0.5,))])
    val_loader = DataLoader(
        datasets.MNIST('./data', train=False, download=True, transform=tx),
        batch_size=256, shuffle=False)

    # --- global model init
    global GLOBAL_STATE
    GLOBAL_STATE = SmallCNN(pretrained=True).state_dict()   # start from your pretrained 97% ckpt

    barrier = threading.Barrier(args.devices + 1)           # +1 for main thread
    updates  = [None]*args.devices
    for idx,sock in enumerate(socks):
        threading.Thread(target=client_handler,
                         args=(sock,barrier,updates,idx),
                         daemon=True).start()

    for r in range(args.rounds):
        barrier.wait()                # release clients → they send local updates
        barrier.wait()                # wait until all updates are back

        GLOBAL_STATE = average_state_dicts(updates)
        torch.save(GLOBAL_STATE, f'training/round_{r}.pth')

        acc = eval_acc(GLOBAL_STATE, val_loader, device='cpu')
        print(f'▶ Round {r:02d}: val-acc = {acc:.3%}')

if __name__ == '__main__':
    main()
