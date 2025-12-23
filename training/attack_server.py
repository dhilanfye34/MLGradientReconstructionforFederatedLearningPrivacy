import os, sys, socket, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch.nn.functional as F
from torch.autograd import grad
from pathlib import Path
from cnn_model import SmallCNN
from training.utils import recv_pkl
from torchvision.utils import save_image

HOST, PORT = "0.0.0.0", 12346
OUTDIR = Path(__file__).parent / "results"
OUTDIR.mkdir(parents=True, exist_ok=True)

DEBUG = True # extra debugging

def total_variation(x):
    tv_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    tv_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return tv_h + tv_w

def cosine_layer_loss_weighted(dummy_grads, tgt_grads, eps=1e-12):
    losses, weights = [], []
    for dg, tg in zip(dummy_grads, tgt_grads):
        dg_f = dg.reshape(-1)  
        tg_f = tg.reshape(-1)  
        cos = (dg_f * tg_f).sum() / (dg_f.norm() * tg_f.norm() + eps)  # calc cos similarity between our fake gradients and the real ones
        losses.append(1.0 - cos)  
        weights.append(tg_f.norm().detach() + eps)  
    
    w = torch.stack(weights)
    L = torch.stack(losses)
    return (L * (w / w.sum())).sum()  # weighted average

def deep_leakage_from_gradients(model, grads_by_name, label_counts, leak_batch_size, *, msg=None, iters=20000, log_every=1000, save_every=2000, device="cpu"):
    model.train()  
    for p in model.parameters():
        p.requires_grad_(True)  # make sure we can calc gradients for the model parameters

    params, tgt = [], []
    
    for n, p in model.named_parameters():
        if n not in grads_by_name:
            raise RuntimeError(f"[DLG] Missing grad for param: {n}")
        g = grads_by_name[n]
        if not torch.is_tensor(g):
            g = torch.tensor(g)  
        params.append(p)
        tgt.append(g.to(device).detach().float())  # store the target gradients we want to match

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
    # use true labels if available, otherwise reconstruct from label counts
    if msg is not None and "true_labels" in msg:
        labels_list = msg["true_labels"]
        print(f"[DLG] Using true label list from payload: {labels_list}")
    else:
        labels_list = []
        for label, count in label_counts.items():
            labels_list.extend([int(label)] * count) 

    while len(labels_list) < leak_batch_size:
        labels_list.append(labels_list[-1] if labels_list else 0) 
    labels_list = labels_list[:leak_batch_size]

    target_y = torch.tensor(labels_list, device=device).long() # target labels for the batch
    print(f"[DLG] Reconstructing batch of size {leak_batch_size} with labels: {labels_list}")

    dummy_img = torch.randn(leak_batch_size, 1, 28, 28, device=device, requires_grad=True)  
    
    opt = torch.optim.Adam([dummy_img], lr=0.05) # optimizer to change the dummy image
    lambda_tv = 2e-4 # small tv keeps edges but avoids smearing
    lambda_l2 = 1e-6

    best_match = float("inf") 
    best_png = OUTDIR / "dlg_best.png"

    for it in range(1, iters + 1):
        opt.zero_grad()  

        pred = model(dummy_img)  # pass fake image through the model
        
        ce = F.cross_entropy(pred, target_y, reduction='sum') 
        dummy_grads = grad(ce, params, create_graph=True) # calc gradients for fake image
        match = cosine_layer_loss_weighted(dummy_grads, tgt) # compare fake gradients to real ones
        tv = total_variation(dummy_img) 
        l2 = (dummy_img ** 2).mean()  
        loss = match + lambda_tv * tv + lambda_l2 * l2  # combine all losses
        loss.backward()  # calc how to change the dummy image to minimize loss
        opt.step()  # update the dummy image

        with torch.no_grad():
            dummy_img.clamp_(-1.0, 1.0) # keep pixel values in valid range

        if it % log_every == 0:
            with torch.no_grad():
                print(f"it={it:05d} | match(cos)={match.item():.6f} | tv={tv.item():.6f} | loss={loss.item():.6f}")  

        if match.item() < best_match: 
            best_match = match.item() # update the best match
            
            with torch.no_grad():
                vis = (dummy_img * 0.5 + 0.5).clamp(0, 1).cpu() 
                save_image(vis, best_png, nrow=int(leak_batch_size**0.5)+1) 

        if it % save_every == 0:
            with torch.no_grad():
                vis = (dummy_img * 0.5 + 0.5).clamp(0, 1).cpu()     
                out_png = OUTDIR / f"dlg_iter_{it}.png"
                
                save_image(vis, out_png, nrow=int(leak_batch_size**0.5)+1)  
                torch.save(dummy_img.detach().cpu(), OUTDIR / f"dlg_iter_{it}.pt")  
                print(f"Image saved: {out_png}")

    with torch.no_grad():
        vis = (dummy_img * 0.5 + 0.5).clamp(0, 1).cpu()  
        save_image(vis, OUTDIR / "dlg_final_image.png", nrow=int(leak_batch_size**0.5)+1)  
        torch.save(dummy_img.detach().cpu(), OUTDIR / "dlg_final_image.pt")  
        print(f"Final PNG saved at {OUTDIR / 'dlg_final_image.png'} | best snapshot: {best_png} (match={best_match:.6f})")

def main():
    print(f"[attack/DLG] listening on {HOST}:{PORT}")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  
        s.bind((HOST, PORT)); s.listen(1)  
        conn, addr = s.accept()  # accept the connection
        print(f"[attack/DLG] got payload from {addr}")
        msg = recv_pkl(conn)  # receive the leaked data
        conn.close()

    grads_by_name = msg["grads_by_name"]  
    state_dict = msg["state_dict"]  
    label_counts = msg.get("label_counts", {})
    leak_batch_size = msg.get("leak_batch_size", 1)

    device = "cpu"
    model = SmallCNN(pretrained=False)  
    state_dict = {k: (v if isinstance(v, torch.Tensor) else torch.tensor(v)) for k, v in state_dict.items()}  
    model.load_state_dict(state_dict)  # load the weights from the client
    model.to(device)  

    model_names = [n for n, _ in model.named_parameters()]
    missing = [n for n in model_names if n not in grads_by_name]
    extra = [n for n in grads_by_name.keys() if n not in set(model_names)]
    
    if DEBUG:
        print(f"[DLG/debug] model has {len(model_names)} params; payload has {len(grads_by_name)} grads") 
        if missing:
            print("[DLG/debug] MISSING grads for:", missing[:10], ("..." if len(missing) > 10 else "")) 
        if extra:
            print("[DLG/debug] EXTRA grads in payload:", extra[:10], ("..." if len(extra) > 10 else "")) 
        print(f"[DLG/debug] label_counts={label_counts}, B={leak_batch_size}")

    if not label_counts:
        if "fc2.bias" not in grads_by_name:
             raise RuntimeError("No label_counts provided and cannot infer single label")
        gb = grads_by_name["fc2.bias"].reshape(-1).float()
        guessed_label = int(torch.argmin(gb).item())
        print(f"[DLG/debug] label_counts missing, inferred single label {guessed_label} from fc2.bias")
        label_counts = {guessed_label: leak_batch_size}

    print("[attack/DLG] starting reconstruction…")
    deep_leakage_from_gradients(model, grads_by_name, label_counts, leak_batch_size, msg=msg, iters=20000, log_every=1000, save_every=2000, device=device)
    print("[attack/DLG] done.")

if __name__ == "__main__":
    main()
