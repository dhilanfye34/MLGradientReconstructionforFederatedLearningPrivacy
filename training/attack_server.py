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

DEBUG = True  # extra logging

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
    tv_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    tv_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return tv_h + tv_w

def cosine_layer_loss_weighted(dummy_grads, tgt_grads, eps=1e-12):
    """Mean cosine distance per layer, weighted by target grad norm."""
    losses, weights = [], []
    
    for dg, tg in zip(dummy_grads, tgt_grads):
        dg_f = dg.reshape(-1)  # flatten dummy grad
        tg_f = tg.reshape(-1)  # flatten target grad
        cos = (dg_f * tg_f).sum() / (dg_f.norm() * tg_f.norm() + eps)  # cosine similarity
        losses.append(1.0 - cos)  # convert to distance
        weights.append(tg_f.norm().detach() + eps)  # weight by target norm
    
    w = torch.stack(weights)
    L = torch.stack(losses)
    return (L * (w / w.sum())).sum()  # weighted average

def deep_leakage_from_gradients(model, grads_by_name, fixed_label, *, iters=20000, log_every=1000, save_every=2000, device="cpu"):
    model.train()  
    for p in model.parameters():
        p.requires_grad_(True)  

    params, tgt = [], []
    
    for n, p in model.named_parameters():
        if n not in grads_by_name:
            raise RuntimeError(f"[DLG] Missing grad for param: {n}")
        g = grads_by_name[n]
        if not torch.is_tensor(g):
            g = torch.tensor(g)  
        params.append(p)
        tgt.append(g.to(device).detach().float())  

    if DEBUG:
        print(f"[DLG/debug] received {len(tgt)} layer-grads")
        bad_shapes = []
        for (n, p), g in zip(model.named_parameters(), tgt):
            if p.shape != g.shape:
                bad_shapes.append((n, tuple(p.shape), tuple(g.shape)))
        if bad_shapes:
            print("[DLG/debug] SHAPE MISMATCHES:", bad_shapes)
            raise RuntimeError("Gradient/param shape mismatch—stop")
        if "fc2.bias" in grads_by_name:
            gb = grads_by_name["fc2.bias"].reshape(-1).float()
            print(f"[DLG/debug] fc2.bias grad stats: min={gb.min().item():.6f} "
                  f"max={gb.max().item():.6f} argmin(label?)={int(torch.argmin(gb))}")
        total_norm = torch.sqrt(sum(g.pow(2).sum() for g in tgt)).item()
        print(f"[DLG/debug] total target-grad L2 norm: {total_norm:.6f}")

    dummy_img = torch.randn(1, 1, 28, 28, device=device, requires_grad=True)  
    target_y = torch.tensor([int(fixed_label)], device=device)  

    opt = torch.optim.Adam([dummy_img], lr=0.05)  
    lambda_tv = 2e-4   # small TV keeps edges but avoids smearing
    lambda_l2 = 1e-6

    best_match = float("inf") 
    best_png = OUTDIR / "dlg_best.png"

    for it in range(1, iters + 1):
        opt.zero_grad()  

        pred = model(dummy_img)  # forward pass
        ce = F.cross_entropy(pred, target_y) 
        dummy_grads = grad(ce, params, create_graph=True) # get gradients

        match = cosine_layer_loss_weighted(dummy_grads, tgt)  # weighted cosine
        tv = total_variation(dummy_img)  
        l2 = (dummy_img ** 2).mean()  # l2 regularization
        loss = match + lambda_tv * tv + lambda_l2 * l2  # total loss

        loss.backward()  # backprop
        opt.step()  # update image

        with torch.no_grad():
            dummy_img.clamp_(-1.0, 1.0)

        if it % log_every == 0:
            with torch.no_grad():
                print(f"it={it:05d} | match(cos)={match.item():.6f} | tv={tv.item():.6f} "
                      f"| pred={pred.argmax(1).item()} | y*={int(target_y.item())}")  # progress update

        if match.item() < best_match:
            best_match = match.item()
            
            with torch.no_grad():
                vis = (dummy_img * 0.5 + 0.5).clamp(0, 1).cpu()  # denormalize for display
                save_image(vis, best_png)

        if it % save_every == 0:
            with torch.no_grad():
                vis = (dummy_img * 0.5 + 0.5).clamp(0, 1).cpu()     
                out_png = OUTDIR / f"dlg_iter_{it}.png"
                
                save_image(vis, out_png)  
                torch.save(dummy_img.detach().cpu(), OUTDIR / f"dlg_iter_{it}.pt")  
                print(f"🖼️ saved {out_png}")

    with torch.no_grad():
        vis = (dummy_img * 0.5 + 0.5).clamp(0, 1).cpu()  # final image
        save_image(vis, OUTDIR / "dlg_final_image.png")  
        torch.save(dummy_img.detach().cpu(), OUTDIR / "dlg_final_image.pt")  
        print(f"✅ final PNG saved at {OUTDIR / 'dlg_final_image.png'} | best snapshot: {best_png} (match={best_match:.6f})")

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
    maybe_label = msg.get("label", None)  

    device = "cpu"
    model = SmallCNN(pretrained=False)  
    state_dict = {k: (v if isinstance(v, torch.Tensor) else torch.tensor(v)) for k, v in state_dict.items()}  
    model.load_state_dict(state_dict)  
    model.to(device)  

    model_names = [n for n, _ in model.named_parameters()]
    missing = [n for n in model_names if n not in grads_by_name]
    extra = [n for n in grads_by_name.keys() if n not in set(model_names)]
    if DEBUG:
        print(f"[DLG/debug] model has {len(model_names)} params; payload has {len(grads_by_name)} grads") # print the number of params and grads
        if missing:
            print("[DLG/debug] MISSING grads for:", missing[:10], ("..." if len(missing) > 10 else "")) # print first 10 missing grads
        if extra:
            print("[DLG/debug] EXTRA grads in payload:", extra[:10], ("..." if len(extra) > 10 else "")) # print first 10 extra grads
        if maybe_label is not None:
            print(f"[DLG/debug] payload says label={maybe_label}") # print the label

    if maybe_label is None:
        if "fc2.bias" not in grads_by_name:
            raise RuntimeError("No label provided and cannot infer (fc2.bias missing).")
        gb = grads_by_name["fc2.bias"].reshape(-1).float()  # get bias grad
        fixed_label = int(torch.argmin(gb).item())  # infer label from min
        if DEBUG:
            print(f"[DLG/debug] inferred label via fc2.bias argmin -> {fixed_label}")
    else:
        fixed_label = int(maybe_label)  # use provided label

    print("[attack/DLG] starting reconstruction…")
    deep_leakage_from_gradients(model, grads_by_name, fixed_label, iters=20000, log_every=1000, save_every=2000, device=device)
    print("[attack/DLG] done.")

if __name__ == "__main__":
    main()
