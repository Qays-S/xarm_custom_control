#!/usr/bin/env python3
"""
Blueprint Section V, step 28 (second half) and step 31: deterministic GRU forecaster.

Input per history step (L = 16 steps = 1.0 s), for elbow, wrist and hand:
  * position relative to the current position (u, v), in normalised image units
  * the current absolute position (so the model knows where on the table the arm is)
  * detector confidence c_t and missingness m_t
Missing history steps are filled with the last observed value (their m_t = 1 tells the model).
Robot state x_r is left out (the robot was parked in E1-E4), and so is the on-screen goal:
a robot would not normally know the person's goal, and in E4 it would give away the switch.

Output: the displacement from the current position for all K = 24 future steps (1.5 s),
so one model gives the 0.25, 0.5, 1.0 and 1.5 s forecasts.
Loss: mean squared error in pixels on future steps that were observed.

Tuning (step 31) uses D_val only: a small grid of hidden size and layers, early stopping on
val ADE. The test and calibration sets are not touched.

Outputs in logs/forecast_results/gru/
  learning_curve_<cfg>.csv   train loss and val ADE per epoch, for every config
  search.csv                 best val ADE per config
  gru_best.pt, gru_best.json the chosen model and its settings
  gru_val.csv                ADE/FDE per horizon (same format as baselines_val.csv)
  learning_curves.png        (if matplotlib is installed)

Usage
  python3 s5_gru.py                    # grid search, about 5-15 min on a laptop CPU
  python3 s5_gru.py --quick            # one small config, for a quick check
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from s5_data import FORECAST_KP, HORIZONS_S, ade_fde_px, eval_mask, load, windows

SIZE = np.array([1280.0, 720.0])


def ffill(h, miss):
    """Fill missing history steps with the last observed value (backward fill at the start)."""
    h = h.copy()
    N, L, J, _ = h.shape
    for j in range(J):
        x = h[:, :, j]
        bad = np.isnan(x[..., 0])
        for t in range(1, L):
            b = bad[:, t]
            x[b, t] = x[b, t - 1]
            bad[b, t] = bad[b, t - 1]
        for t in range(L - 2, -1, -1):
            b = np.isnan(x[:, t, 0])
            x[b, t] = x[b, t + 1]
        h[:, :, j] = x
    return np.nan_to_num(h)


def features(w):
    h = ffill(w["hist"], w["hmiss"])                     # (N,L,3,2)
    cur = h[:, -1:]                                      # (N,1,3,2)
    rel = (h - cur) * SIZE / 100.0                       # pixels / 100
    absn = np.repeat(cur, h.shape[1], axis=1)            # normalised 0..1
    x = np.concatenate([rel.reshape(*rel.shape[:2], -1), absn.reshape(*absn.shape[:2], -1),
                        w["hconf"], w["hmiss"].astype(float)], axis=-1)
    return x.astype(np.float32), cur[:, 0]


def targets(w, cur):
    d = (w["fut"] - cur[:, None]) * SIZE / 100.0         # (N,K,3,2) pixels / 100
    m = (w["fmiss"] == 0)[..., None].repeat(2, -1)
    return np.nan_to_num(d).astype(np.float32), m.astype(np.float32)


class GRUForecaster(nn.Module):
    def __init__(self, n_in, hidden, layers, K, J=3, dropout=0.1):
        super().__init__()
        self.K, self.J = K, J
        self.gru = nn.GRU(n_in, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, K * J * 2))

    def forward(self, x):
        _, hN = self.gru(x)
        return self.head(hN[-1]).view(-1, self.K, self.J, 2)


def predict(model, x, cur, bs=2048):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            out.append(model(torch.from_numpy(x[i:i + bs])).numpy())
    d = np.concatenate(out) * 100.0 / SIZE
    return cur[:, None] + d                               # normalised positions (N,K,3,2)


def val_ade(pred, w, ks):
    r = []
    for k in ks:
        m = eval_mask(w, k)
        a, _ = ade_fde_px(pred[m], {"fut": w["fut"][m]}, k)
        r.append(a.mean())
    return float(np.mean(r))


def train_one(cfg, data, ks, seed, max_epochs, patience, log_path):
    torch.manual_seed(seed); np.random.seed(seed)
    xtr, ytr, mtr, wv, xv, cv = data
    model = GRUForecaster(xtr.shape[-1], cfg["hidden"], cfg["layers"], ytr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    X, Y, M = map(torch.from_numpy, (xtr, ytr, mtr))
    best, best_state, bad, curve = 1e9, None, 0, []
    for ep in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(len(X))
        tot = 0.0
        for i in range(0, len(X), cfg["batch"]):
            b = perm[i:i + cfg["batch"]]
            p = model(X[b])
            loss = (((p - Y[b]) ** 2) * M[b]).sum() / M[b].sum().clamp(min=1)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b)
        v = val_ade(predict(model, xv, cv), wv, ks)
        curve.append((ep, tot / len(X), v))
        if v < best - 1e-3:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    with open(log_path, "w", newline="") as f:
        cw = csv.writer(f); cw.writerow(["epoch", "train_mse_px2_per_1e4", "val_mean_ade_px"]); cw.writerows(curve)
    model.load_state_dict(best_state)
    return model, best, curve


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepared", default="logs/prepared")
    ap.add_argument("--out", default="logs/forecast_results/gru")
    ap.add_argument("--max-epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    m, sess = load(a.prepared)
    ks = [round(h / m["dt"]) for h in HORIZONS_S]
    wt, wv = windows(m, sess, "train"), windows(m, sess, "val")
    keep = wt["hmiss"][:, -1].sum(1) == 0                 # train only on windows whose current pose is observed
    wt = {k: v[keep] for k, v in wt.items()}
    xtr, ctr = features(wt); ytr, mtr = targets(wt, ctr)
    xv, cv = features(wv)
    data = (xtr, ytr, mtr, wv, xv, cv)
    print(f"train {len(xtr)} windows, val {len(xv)} windows, input {xtr.shape[-1]} features x {xtr.shape[1]} steps")

    grid = [dict(hidden=64, layers=1, lr=2e-3, batch=256)] if a.quick else \
        [dict(hidden=h, layers=l, lr=lr, batch=256) for h in (64, 128) for l in (1, 2) for lr in (1e-3, 3e-3)]
    results = []
    for cfg in grid:
        name = f"h{cfg['hidden']}_l{cfg['layers']}_lr{cfg['lr']:g}"
        t0 = time.time()
        model, best, curve = train_one(cfg, data, ks, a.seed, a.max_epochs, a.patience,
                                       os.path.join(a.out, f"learning_curve_{name}.csv"))
        results.append((best, name, cfg, model, curve))
        print(f"  {name:<18} best val mean ADE {best:6.2f} px  ({len(curve)} epochs, {time.time() - t0:.0f} s)")
    with open(os.path.join(a.out, "search.csv"), "w", newline="") as f:
        cw = csv.writer(f); cw.writerow(["config", "val_mean_ade_px", "epochs"])
        cw.writerows([(n, round(b, 3), len(c)) for b, n, _, _, c in results])
    best, name, cfg, model, curve = min(results, key=lambda r: r[0])
    torch.save(model.state_dict(), os.path.join(a.out, "gru_best.pt"))
    json.dump({"config": cfg, "name": name, "n_in": int(xtr.shape[-1]), "K": int(ytr.shape[1]),
               "keypoints": FORECAST_KP, "dt": m["dt"], "history_steps": m["history_steps"],
               "seed": a.seed, "chosen_on": "val", "val_mean_ade_px": round(best, 3)},
              open(os.path.join(a.out, "gru_best.json"), "w"), indent=1)
    print(f"\nChosen on val: {name}")

    # ---- val report, same layout as s5_baselines.py
    p = predict(model, xv, cv)
    print(f"\n{'model':<28}{'horizon':>8}{'n':>7}" + "".join(f"{k + ' ADE':>12}" for k in FORECAST_KP) + f"{'mean ADE':>10}{'mean FDE':>10}")
    rows = []
    for h, k in zip(HORIZONS_S, ks):
        msk = eval_mask(wv, k)
        ad, fd = ade_fde_px(p[msk], {"fut": wv["fut"][msk]}, k)
        print(f"{'GRU':<28}{h:>7}s{msk.sum():>7}" + "".join(f"{v:>12.1f}" for v in ad.mean(0)) + f"{ad.mean():>10.1f}{fd.mean():>10.1f}")
        rows.append(["val", "GRU " + name, h, int(msk.sum())] + [round(float(v), 2) for v in ad.mean(0)]
                    + [round(float(v), 2) for v in fd.mean(0)] + [round(float(ad.mean()), 2), round(float(fd.mean()), 2)])
    k1 = ks[HORIZONS_S.index(1.0)]
    groups = [("cond", c) for c in sorted(set(wv["cond"]))] + [("phase", ph) for ph in ("cued", "moving", "hold")]
    print(f"\nHand ADE (px) at 1.0 s:      " + "".join(f"{g[1]:>9}" for g in groups))
    line = f"{'GRU':<28}"
    for key, val in groups:
        sel = (wv[key] == val) & eval_mask(wv, k1)
        ad, _ = ade_fde_px(p[sel], {"fut": wv["fut"][sel]}, k1)
        line += f"{ad[:, 2].mean():>9.1f}"
    print(line)
    with open(os.path.join(a.out, "gru_val.csv"), "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["set", "model", "horizon_s", "n_windows"] + [f"{k}_ade_px" for k in FORECAST_KP]
                    + [f"{k}_fde_px" for k in FORECAST_KP] + ["mean_ade_px", "mean_fde_px"])
        cw.writerows(rows)

    # ---- inference time, one window at a time (what the robot would do online)
    model.eval(); torch.set_num_threads(1)
    x1 = torch.from_numpy(xv[:1]); ts = []
    with torch.no_grad():
        for i in range(300):
            t0 = time.perf_counter(); model(torch.from_numpy(xv[i % len(xv)][None])); ts.append((time.perf_counter() - t0) * 1e3)
    ts = np.array(ts[20:])
    print(f"\nInference time, one window, 1 CPU thread: median {np.median(ts):.2f} ms, p95 {np.percentile(ts, 95):.2f} ms")
    json.dump({"median_ms": round(float(np.median(ts)), 3), "p95_ms": round(float(np.percentile(ts, 95)), 3)},
              open(os.path.join(a.out, "gru_inference_time.json"), "w"))

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        for b, n, _, _, c in results:
            c = np.array(c)
            ax[0].plot(c[:, 0], c[:, 1], label=n, lw=1); ax[1].plot(c[:, 0], c[:, 2], label=n, lw=1)
        ax[0].set(xlabel="epoch", ylabel="train loss", yscale="log", title="Training loss")
        ax[1].set(xlabel="epoch", ylabel="val mean ADE (px)", title="Validation ADE")
        ax[1].legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(a.out, "learning_curves.png"), dpi=150)
    except ImportError:
        pass
    print(f"Saved in {a.out}/")


if __name__ == "__main__":
    main()
