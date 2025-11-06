#!/usr/bin/env python3
import os, socket, pickle, torch
import torch.nn.functional as F
from torch.autograd import grad
from pathlib import Path
from cnn_model import SmallCNN

HOST, PORT = "0.0.0.0", 12346  # must match client host:(port+1)
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

def deep_leakage_from_gradients(model, target_grads, iters=800, log_every=50, save_every=100, device="cpu"):
    """
    Gradient-matching (DLG) for a single MNIST image (1x1x28x28).
    target_grads: list[Tensor], same order as model.parameters()
    """
    model.eval()

    # Dummy image & soft label to optimize
    dummy_img = torch.randn(1, 1, 28, 28, device=device, requires_grad=True)
    dummy_lbl_logits = torch.zeros(1, 10, device=device, requires_grad=True)

    opt = torch.optim.Adam([dummy_img, dummy_lbl_logits], lr=0.1)
    tgt = [g.to(device).float() for g in target_grads]

    for it in range(1, iters + 1):
        opt.zero_grad()

        pred = model(dummy_img)                                  # forward on dummy
        pseudo_y = F.softmax(dummy_lbl_logits, dim=-1)           # soft label
        loss = torch.sum(-pseudo_y * F.log_softmax(pred, dim=-1))  # CE with soft label

        dummy_grads = grad(loss, model.parameters(), create_graph=False)

        grad_diff = sum((dg - tg).pow(2).sum() for dg, tg in zip(dummy_grads, tgt))
        grad_diff.backward()
        opt.step()

        if it % log_every == 0:
            with torch.no_grad():
                pred_lbl = pred.argmax(dim=1).item()
                print(f"🔁 it={it:04d} | grad_diff={grad_diff.item():.6f} | pred≈{pred_lbl}")

        if it % save_every == 0:
            with torch.no_grad():
                # unnormalize to [0,1] for viewing since training used Normalize(0.5,0.5)
                img_vis = (dummy_img * 0.5 + 0.5).clamp(0, 1).cpu()
                path = OUTDIR / f"dlg_iter_{it}.pt"
                torch.save(img_vis, path)
                print(f"💾 saved dummy image tensor at {path}")

    with torch.no_grad():
        final_img = (dummy_img * 0.5 + 0.5).clamp(0, 1).cpu()
        pred_lbl = F.softmax(dummy_lbl_logits, dim=-1).argmax(dim=-1).item()
    torch.save(final_img, OUTDIR / "dlg_final_image.pt")
    print(f"🖼️ final image saved at {OUTDIR / 'dlg_final_image.pt'} | pred label ≈ {pred_lbl}")
    return final_img

def main():
    print(f"[attack/DLG] listening on {HOST}:{PORT}")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT)); s.listen(1)
        conn, addr = s.accept()
        print(f"[attack/DLG] got payload from {addr}")
        msg = recv_pkl(conn)
        conn.close()

    grads = msg["grads"]                                   # list of tensors (CPU)
    state_dict = msg["state_dict"]                         # model weights from client round
    true_label = msg.get("label", None)

    device = "cpu"  # use "cuda" if available and desired
    model = SmallCNN(pretrained=False)
    # Ensure tensors
    state_dict = {k: (v if isinstance(v, torch.Tensor) else torch.tensor(v)) for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)

    print("[attack/DLG] starting reconstruction…")
    deep_leakage_from_gradients(model, grads, iters=800, log_every=50, save_every=100, device=device)
    print("[attack/DLG] done.")

if __name__ == "__main__":
    main()
