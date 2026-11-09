"""run_experiment.py -- trains AE and AE-FAR (and optionally VAE / VAE-FAR) for one dataset and
several seeds, leak-free, and saves the anomaly scores for evaluation.

Per seed:  stage 1 = train the base network (AE or VAE) alone;  stage 2 = freeze it and train the
attention + MSE-feedback RNN (AE-FAR / VAE-FAR), exactly as the official two-phase procedure.
CHANGED vs. official: validation = last VAL_FRAC of the training series (official: the test set).
"""
import argparse, json, os, sys, time

import numpy as np
import torch


def main(a):
    os.chdir(a.repo)                                  # official read_data uses ../Anomaly/<DS>/
    sys.path.insert(0, a.repo); sys.path.insert(0, a.lib_dir)
    from utils import read_data                       # official
    from my_models import CustomModel, LSTM_VAE       # official
    import aefar_lib as L

    dev = torch.device(a.device)
    train_np, test_np, _val_official, labels = read_data(a.dataset)   # scaler fit on train only
    labels = np.asarray(labels).reshape(-1).astype(int)
    if a.max_rows:                                    # DRY_RUN only
        train_np, test_np, labels = train_np[:a.max_rows], test_np[:a.max_rows], labels[:a.max_rows]
    n_val = int(len(train_np) * a.val_frac)
    X_tr = torch.tensor(train_np[:-n_val], dtype=torch.float32, device=dev)
    X_va = torch.tensor(train_np[-n_val:], dtype=torch.float32, device=dev)
    X_te = torch.tensor(test_np, dtype=torch.float32, device=dev)
    d = X_tr.shape[1]
    cfg = dict(window=a.window, batch_size=a.batch_size, lr=a.lr, max_epochs=a.max_epochs, patience=a.patience)
    print(f"[{a.dataset}] train {tuple(X_tr.shape)} val {tuple(X_va.shape)} test {tuple(X_te.shape)} "
          f"anomaly rate {labels.mean():.4f} device {dev}", flush=True)

    # all AE seeds first; the VAE (bonus) runs last, first seed only, if time remains
    jobs = [("ae", s) for s in a.seeds] + ([("vae", a.seeds[0])] if a.vae else [])
    for base, seed in jobs:
        if True:
            out = os.path.join(a.out, a.dataset, f"{base}_seed{seed}")
            os.makedirs(out, exist_ok=True)
            if os.path.exists(os.path.join(out, "scores.npz")):
                print(f"[{a.dataset}] {base} seed {seed}: already done, skipping", flush=True)
                continue
            if a.deadline and time.time() > a.deadline:
                print(f"[{a.dataset}] {base} seed {seed}: SKIPPED (session deadline)", flush=True)
                json.dump({"status": "skipped_deadline"}, open(os.path.join(out, "info.json"), "w"))
                continue
            L.set_seed(seed)
            # ---- stage 1: base network ----
            base_net = (CustomModel(a.window, d) if base == "ae"
                        else LSTM_VAE(a.window, input_size=d, hidden_size=64, latent_size=32)).to(dev)
            base_net, info1 = L.train_model(base_net, base, X_tr, X_va, cfg,
                                            os.path.join(out, "log_stage1.json"), a.deadline)
            torch.save(base_net.state_dict(), os.path.join(out, "base.pt"))
            print(f"[{a.dataset}] {base} seed {seed} stage1 {info1}", flush=True)
            # ---- stage 2: composite (frozen base + attention + MSE-feedback RNN) ----
            for p in base_net.parameters():
                p.requires_grad = False
            rnn = L.MSEFeedbackRNN(input_size=d + 1, hidden_size=d // 2, output_size=d, num_layers=8)
            comp = L.CompositeModel(base_net, rnn, attention_dim=64, input_dim=d).to(dev)
            comp, info2 = L.train_model(comp, "composite", X_tr, X_va, cfg,
                                        os.path.join(out, "log_stage2.json"), a.deadline)
            torch.save(comp.state_dict(), os.path.join(out, "composite.pt"))
            print(f"[{a.dataset}] {base} seed {seed} stage2 {info2}", flush=True)
            # ---- scores (val scores are used for the leak-free fixed threshold) ----
            np.savez_compressed(
                os.path.join(out, "scores.npz"),
                base_test=L.score(base_net, base, X_te, a.window, a.batch_size),
                far_test=L.score(comp, "composite", X_te, a.window, a.batch_size),
                base_val=L.score(base_net, base, X_va, a.window, a.batch_size),
                far_val=L.score(comp, "composite", X_va, a.window, a.batch_size),
                labels=labels[a.window:])
            json.dump({"dataset": a.dataset, "base": base, "seed": seed, "n_features": d,
                       "stage1": info1, "stage2": info2,
                       "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
                       "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3) if dev.type == "cuda" else None},
                      open(os.path.join(out, "info.json"), "w"), indent=2)
    print(f"[{a.dataset}] DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--lib_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--vae", action="store_true")
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--deadline", type=float, default=0)
    ap.add_argument("--max_rows", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    main(ap.parse_args())
