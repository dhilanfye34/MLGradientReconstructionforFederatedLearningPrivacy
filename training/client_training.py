#!/usr/bin/env python3
import argparse, socket, pickle, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from cnn_model import SmallCNN
from training.utils import recv_pkl, send_pkl

def local_epoch(model, loader, device):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    for x,y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--host', default='laptop.local')
    p.add_argument('--port', type=int, default=12345)
    p.add_argument('--shard', type=int, default=0)       # device id
    p.add_argument('--total_shards', type=int, default=1)
    args = p.parse_args()

    # simple deterministic split: idx % total_shards == shard
    tx = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.5,),(0.5,))])
    all_ds  = datasets.MNIST('./data', train=True,  download=True, transform=tx)
    shard_ds = torch.utils.data.Subset(all_ds,
                [i for i in range(len(all_ds)) if i % args.total_shards == args.shard])
    loader = DataLoader(shard_ds, batch_size=64, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((args.host, args.port))
        print('[✔] Connected to server')
        while True:
            state_dict = recv_pkl(s)              # global weights
            model = SmallCNN(load_from_state_dict=state_dict) \
                    if hasattr(SmallCNN, 'load_from_state_dict') else SmallCNN(pretrained=False)
            model.load_state_dict(state_dict)
            model.to(device)

            local_epoch(model, loader, device)    # 1 epoch
            send_pkl(s, model.cpu().state_dict()) # send updated weights back
