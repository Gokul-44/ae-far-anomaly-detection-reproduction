"""aefar_lib.py -- Project 4 (AE-FAR reproduction) helper library.

Our own implementation of the AE-FAR composite model (paper Eq. 1-4) and of the leak-free
training / scoring pipeline. Class and attribute names of AttentionLayer, MSEFeedbackRNN and
CompositeModel deliberately match the official notebooks (github.com/mribrahim/AE-FAR, commit
794bb0b) so that the authors' pickled pretrained models can be loaded and checked (experiment E0).

The base networks (CustomModel = the "AE", LSTM_VAE) and the evaluation helpers
(read_data, apply_adjustment, sliding_window_anomaly_detection) are imported from the official
repository at run time; this file does not copy them.
"""
import json, os, random, time

import numpy as np
import torch
import torch.nn as nn


# ----------------------------------------------------------------------------------------------
# Composite model (paper Eq. 2-4). Same structure and names as the official notebooks.
# ----------------------------------------------------------------------------------------------
class AttentionLayer(nn.Module):
    """alpha = Softmax(Linear(tanh(Linear(z)))), z' = alpha * z   (paper Eq. 2)."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1), nn.Softmax(dim=1))

    def forward(self, inputs):
        attention_weights = self.attention(inputs)
        return inputs * attention_weights, attention_weights


class MSEFeedbackRNN(nn.Module):
    """h = FC(RNN(z')) (paper Eq. 3). As in the official code, the RNN receives a sequence of
    length 1 with a zero initial state, so it carries no memory across time steps."""

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.rnn(x, h0)
        return self.fc(out)


class CompositeModel(nn.Module):
    """y_hat = X_hat + RNN(alpha * [X_hat, MSE])   (paper Eq. 4)."""

    def __init__(self, autoencoder, rnn, attention_dim=64, input_dim=None):
        super().__init__()
        self.autoencoder = autoencoder
        self.rnn = rnn
        self.attention = AttentionLayer(input_dim=input_dim + 1, hidden_dim=attention_dim)

    def forward(self, x, y):
        out = self.autoencoder(x)
        reconstructed = out[0] if isinstance(out, tuple) else out       # VAE returns a tuple
        mse_error = ((y - reconstructed) ** 2).mean(dim=1, keepdim=True)  # uses the true y
        combined_input = torch.cat((reconstructed, mse_error), dim=1)
        combined_input, _ = self.attention(combined_input)
        rnn_output = self.rnn(combined_input).squeeze(1)
        return reconstructed + rnn_output, mse_error, rnn_output


def vae_loss(recon_x, x, mu, log_var, window_size, input_dim):
    """Official VAE loss (train-model.ipynb): reconstruction of the LAST point of the window
    (sum-reduced MSE) + 0.002 * KL divergence."""
    x_data = x[:, window_size - 1, :]
    rec = nn.functional.mse_loss(recon_x, x_data.view(-1, input_dim), reduction="sum")
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return rec + 0.002 * kl


# ----------------------------------------------------------------------------------------------
# Windowing: identical batches to the official iterate_batches (window -> next point,
# consecutive, no shuffling), but built with one vectorised gather instead of a Python loop.
# ----------------------------------------------------------------------------------------------
def iterate_batches_fast(data, window_size, batch_size, start_idx=0):
    n = len(data) - window_size
    offs = torch.arange(window_size, device=data.device)
    for start in range(start_idx, n, batch_size):
        end = min(start + batch_size, n)
        idx = torch.arange(start, end, device=data.device)
        yield data[idx[:, None] + offs], data[idx + window_size]


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------------------------------
# Training (official loop: Adam lr 1e-4, batch 128, max 100 epochs, early stop when the
# validation loss has not improved for more than 10 epochs) -- with a leak-free validation set.
# ----------------------------------------------------------------------------------------------
def _step_loss(model, kind, xb, yb, window_size, input_dim, loss_fn):
    if kind == "composite":
        y_pred, _, _ = model(xb, yb)
        return loss_fn(y_pred, yb)
    if kind == "ae":
        return loss_fn(model(xb), yb)
    y_pred, mu, log_var = model(xb)                     # vae
    return vae_loss(y_pred, xb, mu, log_var, window_size, input_dim)


def train_model(model, kind, X_train, X_val, cfg, log_path, deadline=0.0):
    """Returns (model with best weights, info dict). kind in {'ae', 'vae', 'composite'}."""
    loss_fn = nn.MSELoss()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg["lr"])
    w, bs, d = cfg["window"], cfg["batch_size"], X_train.shape[1]
    best, best_state, stop_counter, rows = np.inf, None, 0, []
    status, t0 = "completed", time.time()
    for epoch in range(cfg["max_epochs"]):
        if deadline and time.time() > deadline:
            status = "stopped_at_deadline"; break
        te = time.time()
        model.train()
        tr = []
        for xb, yb in iterate_batches_fast(X_train, w, bs):
            opt.zero_grad()
            loss = _step_loss(model, kind, xb, yb, w, d, loss_fn)
            loss.backward(); opt.step(); tr.append(loss.item())
        model.eval()
        with torch.no_grad():
            va = [_step_loss(model, kind, xb, yb, w, d, loss_fn).item()
                  for xb, yb in iterate_batches_fast(X_val, w, bs)]
        tr_m, va_m = float(np.mean(tr)), float(np.mean(va))
        rows.append({"epoch": epoch, "train_loss": tr_m, "val_loss": va_m, "epoch_s": round(time.time() - te, 3)})
        if va_m < best:
            best, stop_counter = va_m, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stop_counter += 1
        if stop_counter > cfg["patience"]:
            status = "early_stopped"; break
    if best_state is not None:
        model.load_state_dict(best_state)
    with open(log_path, "w") as f:
        json.dump(rows, f)
    info = {"kind": kind, "status": status, "epochs_run": len(rows), "best_val_loss": best,
            "best_epoch": int(np.argmin([r["val_loss"] for r in rows])) if rows else None,
            "train_s": round(time.time() - t0, 1),
            "mean_epoch_s": float(np.mean([r["epoch_s"] for r in rows])) if rows else None}
    return model, info


@torch.no_grad()
def score(model, kind, X, window_size, batch_size):
    """Per-time-step anomaly score = mean squared error over features between the true value
    and the model output (official test-model.ipynb). Returns array of length len(X)-window."""
    model.eval()
    preds = []
    for xb, yb in iterate_batches_fast(X, window_size, batch_size):
        if kind == "composite":
            p = model(xb, yb)[0]
        elif kind == "ae":
            p = model(xb)
        else:
            p = model(xb)[0]
        preds.append(p.float().cpu())
    preds = torch.cat(preds).numpy()
    truth = X[window_size:].float().cpu().numpy()
    return np.mean((truth - preds) ** 2, axis=1)
