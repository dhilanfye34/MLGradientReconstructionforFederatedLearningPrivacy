#!/usr/bin/env python3
import sys, os, time, pickle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse, socket, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from cnn_model import SmallCNN
from training.utils import recv_pkl, send_pkl

def local_epoch(model, loader, device, log_every=200):
    # just running one epoch of training on the client's local data
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    total, batches = 0.0, 0
    t0 = time.time()
    for b, (x, y) in enumerate(loader, 1):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        total += loss.item()
        batches = b
        # print out progress every once in a while so we know it's working
        if b % log_every == 0:
            print(f"[client] batch {b}: loss={loss.item():.4f}", flush=True)
    avg = (total / max(1, batches))
    dt = time.time() - t0
    print(f"[client] local epoch done | batches={batches} | avg loss={avg:.4f} | time={dt:.1f}s", flush=True)

def _count_params(state):
    # helper to count how many params we're dealing with
    return sum(t.numel() for t in state.values())

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--host', default='laptop.local')
    p.add_argument('--port', type=int, default=12345)
    p.add_argument('--shard', type=int, default=0)       # device id
    p.add_argument('--total_shards', type=int, default=1)
    args = p.parse_args()

    # dataset shard, each client only gets a slice of the data
    # doing some basic normalization here
    tx = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    all_ds = datasets.MNIST('./data', train=True, download=True, transform=tx)
    # figure out which samples belong to this client based on shard id
    shard_idx = [i for i in range(len(all_ds)) if i % args.total_shards == args.shard]
    shard_ds = torch.utils.data.Subset(all_ds, shard_idx)
    loader = DataLoader(shard_ds, batch_size=64, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[client] starting | host={args.host}:{args.port} | shard={args.shard}/{args.total_shards} "
          f"| shard_size={len(shard_ds)} | device={device}", flush=True)

    # open up a socket connection to the server
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print(f"[client] connecting to server…", flush=True)
        s.connect((args.host, args.port))
        print(f"[client] connected", flush=True)

        # keep running rounds until the server tells us to stop
        while True:
            print("[client] waiting for global weights…", flush=True)
            try:
                state_dict = recv_pkl(s)  # global weights
            except EOFError:
                # server's done, we're done
                print("[client] server closed connection (EOF). exiting.", flush=True)
                return
            except Exception as e:
                print(f"[client] error receiving global weights: {e}. exiting.", flush=True)
                return
            # this is the attack part - if we're in leak mode, send gradient info
            if os.environ.get("LEAK_LABEL_ONLY") == "1":
                try:
                    # need to rebuild the model with the weights we just got
                    model = SmallCNN(pretrained=False)
                    model.load_state_dict(state_dict)
                    model.to(device).train()

                    # grab just one sample so we can compute gradients for it
                    x, y = next(iter(loader))
                    x0, y0 = x[:1].to(device), y[:1].to(device)

                    # do a forward pass and compute loss
                    model.zero_grad()
                    logits = model(x0)
                    loss = F.cross_entropy(logits, y0)

                    # now we need to get the gradients and find the bias gradient
                    # the trick is that the bias gradient directly leaks the label
                    grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)
                    name_list = [n for n, _ in model.named_parameters()]
                    grads_by_name = {name_list[i]: g.detach().cpu() for i, g in enumerate(grads)}

                    if "fc2.bias" not in grads_by_name:
                        raise RuntimeError("fc2.bias gradient not found (check model param names)")

                    fc2_bias_grad = grads_by_name["fc2.bias"]  # length=10

                    # package it up to send to the attack server
                    payload = {
                        "fc2_bias_grad": fc2_bias_grad,   # attacker will do argmin to get true label
                        "note": "label leakage demo"
                    }
                    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)

                    # send to attack server (runs on next port up)
                    atk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    atk.settimeout(3.0)
                    atk.connect((args.host, args.port + 1))
                    atk.sendall(len(data).to_bytes(8, "big"))
                    atk.sendall(data)
                    atk.close()
                    print("[client] leaked last-layer bias gradient (label-only) to attack server", flush=True)
                except Exception as e:
                    print(f"[client] label-only leak failed: {e}", flush=True)
                finally:
                    # turn off the leak flag after we're done
                    os.environ["LEAK_LABEL_ONLY"] = "0"

            print(f"[client] received W_k ({_count_params(state_dict):,} params). training 1 local epoch…", flush=True)

            # load up the global model and get ready to train
            model = SmallCNN(pretrained=False)
            model.load_state_dict(state_dict)
            model.to(device)

            # do the actual training on our local data
            local_epoch(model, loader, device)               # 1 epoch
            out_state = model.cpu().state_dict()
            # send the updated weights back to the server
            print(f"[client] sending updated weights back ({_count_params(out_state):,} params)…", flush=True)
            send_pkl(s, out_state)
            print("[client] update sent.", flush=True)

if __name__ == '__main__':
    main()
