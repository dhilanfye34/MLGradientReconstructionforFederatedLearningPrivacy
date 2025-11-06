#!/usr/bin/env python3
import os, sys, socket, pickle, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch.nn.functional as F
from torch.autograd import grad
from pathlib import Path
from cnn_model import SmallCNN
from torchvision.utils import save_image

HOST, PORT = "0.0.0.0", 12346
OUTDIR = Path(__file__).parent / "results"
OUTDIR.mkdir(parents=True, exist_ok=True)

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

def total_variation(x):
    # x: [1,1,28,28]
    tv_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    tv_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return tv_h + tv_w

def deep_leakage_from_gradients(model, grads_by_name, iters=6000, log_every=200, save_every=1000, device="cpu"):
    """
    DLG with named gradients + mild priors.
    grads_by_name: dict {param_name: grad_tensor}
    """
    model.train()  # IMPORTANT: match single-example grad behavior
    for p in model.parameters():
        p.requires_grad_(True)

    # Build params and targets **in the same name order**
    name_param_pairs = [(n, p) for n, p in model.named_parameters()]
    tgt = []
    params = []
    for n, p in name_param_pairs:
        if n not in grads_by_name:
            raise RuntimeError(f"Missing grad for param: {n}")
        params.append(p)
        tgt.append(grads_by_name[n].to(device).detach().float())
    assert len(params) == len(tgt)

    # Dummy variables we will optimize
    dummy_img = torch.randn(1, 1, 28, 28, device=device, requires_grad=True)   # normalized space
    dummy_lbl_logits = torch.zeros(1, 10, device=device, requires_grad=True)

    opt = torch.optim.Adam([dummy_img, dummy_lbl_logits], lr=0.1)

    # Priors
    lambda_tv = 1e-4
    lambda_l2 = 1e-6

    for it in range(1, iters + 1):
        opt.zero_grad()

        # Forward with soft label on dummy image
        pred = model(dummy_img)
        pseudo_y = F.softmax(dummy_lbl_logits, dim=-1)
        ce = torch.sum(-pseudo_y * F.log_softmax(pred, dim=-1))  # CE with soft labels

        # Gradients of CE wrt model params (first-order graph ON)
        dummy_grads = grad(ce, params, create_graph=True)

        # Gradient-matching objective
        match = sum((dg - tg).pow(2).sum() for dg, tg in zip(dummy_grads, tgt))

        # Add tiny image priors to make the image less noisy
        tv = total_variation(dummy_img)
        l2 = (dummy_img ** 2).mean()
        loss = match + lambda_tv * tv + lambda_l2 * l2

        loss.backward()
        opt.step()

        if it % log_every == 0:
            with torch.no_grad():
                print(f"it={it:05d} | match={match.item():.6f} | tv={tv.item():.6f} | pred={pred.argmax(1).item()}")

        if it % save_every == 0:
            with torch.no_grad():
                # Invert normalization for viewing (training used mean=0.5 std=0.5)
                vis = (dummy_img * 0.5 + 0.5).clamp(0, 1).cpu()
                save_image(vis, OUTDIR / f"dlg_iter_{it}.png")
                torch.save(dummy_img.detach().cpu(), OUTDIR / f"dlg_iter_{it}.pt")
                print(f"🖼️ saved {OUTDIR / f'dlg_iter_{it}.png'}")

    with torch.no_grad():
        vis = (dummy_img * 0.5 + 0.5).clamp(0, 1).cpu()
        save_image(vis, OUTDIR / "dlg_final_image.png")
        torch.save(dummy_img.detach().cpu(), OUTDIR / "dlg_final_image.pt")
        print(f"✅ final PNG saved at {OUTDIR / 'dlg_final_image.png'}")

def main():
    print(f"[attack/DLG] listening on {HOST}:{PORT}")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT)); s.listen(1)
        conn, addr = s.accept()
        print(f"[attack/DLG] got payload from {addr}")
        msg = recv_pkl(conn)
        conn.close()

    grads_by_name = msg["grads_by_name"]
    state_dict = msg["state_dict"]

    device = "cpu"  # switch to "cuda" if you want
    model = SmallCNN(pretrained=False)
    # Ensure tensors
    state_dict = {k: (v if isinstance(v, torch.Tensor) else torch.tensor(v)) for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)

    print("[attack/DLG] starting reconstruction…")
    deep_leakage_from_gradients(model, grads_by_name, iters=6000, log_every=200, save_every=1000, device=device)
    print("[attack/DLG] done.")

if __name__ == "__main__":
    main()
