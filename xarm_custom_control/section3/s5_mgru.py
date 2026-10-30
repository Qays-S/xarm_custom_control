#!/usr/bin/env python3
"""
Blueprint Section V, step 29 (with 31): M-mode probabilistic GRU forecaster.

Same inputs as the deterministic GRU (s5_gru.py): for elbow, wrist and hand, the position relative to
the current position, the current absolute position, detector confidence and missingness, over the last
L = 16 steps (1.0 s). Robot state is left out (the robot was parked) and so is the on-screen goal.

Output: a mixture of M trajectory-level modes over the next K = 24 steps (1.5 s)
    p(Y+ | Z) = sum_m pi_m  prod_{k,j} N( y_kj ; mu_mkj , Sigma_mkj )
  * pi_m        mode probabilities (softmax), one per whole trajectory, so a mode is one way the
                arm can move, not a per-step choice
  * mu_mkj      mean position of keypoint j at step k in mode m (as a displacement from now)
  * Sigma_mkj   2x2 covariance, made positive definite through its Cholesky factor
                L = [[a, 0], [c, b]] with a, b = softplus(.) + 1 px  ->  Sigma = L L^T
  Independence approximation (block diagonal): within a mode, the 2x2 blocks of different keypoints
  and different future steps are treated as independent; correlations across time and joints are
  carried only by the mode means and by the choice of mode.
Loss: negative log-likelihood of the observed future points (missing future points are left out).
Tuning (step 31) on D_val only: M in {1, 3, 5} x hidden {64, 128}, 2 layers, early stopping on val NLL.
M = 1 is the single-Gaussian version, so the grid also shows whether more than one mode helps.

Reported on D_val (same windows as every other model: current pose and future observed)
  * ADE / FDE of the most likely mode (the forecast a planner would use), and of the mixture mean
  * minADE over the M modes (oracle, a secondary diagnostic only: it needs the true future to pick)
  * NLL in nats per observed 2-D point, in pixel units of the 1280x720 image, over the whole
    trajectory up to each horizon (mixture-aware, no collapse to one Gaussian)
  * calibration: how often the true point falls inside the forecast's 50 / 80 / 95% highest-density
    region of the per-point mixture (estimated with 400 samples per point)
  * inference time for one window on one CPU thread

Outputs in logs/forecast_results/mgru/
  learning_curve_<cfg>.csv, search.csv, mgru_best.pt, mgru_best.json, mgru_val.csv,
  mgru_calibration.csv, mgru_inference_time.json, learning_curves.png

Usage
  python3 s5_mgru.py            # 6 runs, about 15-30 min on a laptop CPU
  python3 s5_mgru.py --quick    # one run, for a quick check
"""
import argparse
import csv
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from s5_data import FORECAST_KP, HORIZONS_S, ade_fde_px, eval_mask, load, windows
from s5_gru import SIZE, features, targets

MIN_STD = 0.01            # 1 px, in the model's units (pixels / 100)
LN100 = math.log(100.0)   # model units are px/100, so a density per px^2 is 100^2 smaller


class MixtureGRU(nn.Module):
    def __init__(self, n_in, hidden, layers, K, M, J=3, dropout=0.1):
        super().__init__()
        self.K, self.M, self.J = K, M, J
        self.gru = nn.GRU(n_in, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.body = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
        self.logit = nn.Linear(hidden, M)
        self.mean = nn.Linear(hidden, M * K * J * 2)
        self.chol = nn.Linear(hidden, M * K * J * 3)

    def forward(self, x):
        _, hN = self.gru(x)
        h = self.body(hN[-1])
        N = len(x)
        logpi = F.log_softmax(self.logit(h), -1)                              # (N,M)
        mu = self.mean(h).view(N, self.M, self.K, self.J, 2)
        c = self.chol(h).view(N, self.M, self.K, self.J, 3)
        a = F.softplus(c[..., 0]) + MIN_STD
        b = F.softplus(c[..., 1]) + MIN_STD
        return logpi, mu, a, b, c[..., 2]                                     # L = [[a,0],[off,b]]


def point_logpdf(y, mu, a, b, off):
    """log N(y; mu, L L^T) for every mode/step/keypoint. y (N,1,K,J,2) -> (N,M,K,J)."""
    d = y - mu
    z1 = d[..., 0] / a
    z2 = (d[..., 1] - off * z1) / b
    return -0.5 * (z1 ** 2 + z2 ** 2) - torch.log(a) - torch.log(b) - math.log(2 * math.pi)


def traj_loglik(out, y, mask, k=None):
    """Mixture log-likelihood of each window's observed future up to step k; also the point count.
    y (N,K,J,2) displacement in model units, mask (N,K,J) 1 = observed."""
    logpi, mu, a, b, off = out
    lp = point_logpdf(y[:, None], mu, a, b, off) * mask[:, None]
    if k is not None:
        lp, mask = lp[:, :, :k], mask[:, :k]
    per_mode = lp.sum((2, 3))                                                 # (N,M)
    return torch.logsumexp(logpi + per_mode, -1), mask.sum((1, 2))


def run(model, x, bs=2048):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            outs.append(model(torch.from_numpy(x[i:i + bs])))
    return tuple(torch.cat([o[q] for o in outs]) for q in range(5))


def to_positions(out, cur):
    """Mode means as normalised image positions (N,M,K,J,2), and mode probabilities (N,M)."""
    logpi, mu = out[0].numpy(), out[1].numpy()
    pos = cur[:, None, None] + mu * 100.0 / SIZE
    return np.exp(logpi), pos


def nll_px(model_out, w, cur, k):
    """Mean NLL in nats per observed 2-D point (pixel units) up to step k, on eval_mask windows."""
    y, m = targets(w, cur)
    msk = eval_mask(w, k)
    out = tuple(t[torch.from_numpy(msk)] for t in model_out)
    ll, n = traj_loglik(out, torch.from_numpy(y[msk]), torch.from_numpy(m[msk][..., 0]), k)
    return float(-(ll.sum() / n.sum()).item() + 2 * LN100), int(msk.sum())


def train_one(cfg, data, seed, max_epochs, patience, log_path):
    torch.manual_seed(seed); np.random.seed(seed)
    xtr, ytr, mtr, wv, xv, cv = data
    model = MixtureGRU(xtr.shape[-1], cfg["hidden"], cfg["layers"], ytr.shape[1], cfg["M"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    X, Y, Mk = torch.from_numpy(xtr), torch.from_numpy(ytr), torch.from_numpy(mtr[..., 0])
    K = ytr.shape[1]
    best, best_state, bad, curve = 1e9, None, 0, []
    for ep in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(len(X))
        tot, cnt = 0.0, 0.0
        for i in range(0, len(X), cfg["batch"]):
            bi = perm[i:i + cfg["batch"]]
            ll, n = traj_loglik(model(X[bi]), Y[bi], Mk[bi])
            loss = -ll.sum() / n.sum().clamp(min=1)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += -ll.sum().item(); cnt += n.sum().item()
        v, _ = nll_px(run(model, xv), wv, cv, K)
        curve.append((ep, tot / cnt + 2 * LN100, v))
        if v < best - 1e-3:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    with open(log_path, "w", newline="") as f:
        cw = csv.writer(f); cw.writerow(["epoch", "train_nll_nats_per_point_px", "val_nll_nats_per_point_px"])
        cw.writerows(curve)
    model.load_state_dict(best_state)
    return model, best, curve


def hpd_coverage(out, w, cur, ks, levels=(0.5, 0.8, 0.95), S=400, seed=0):
    """Share of true future points inside the forecast's highest-density region, per horizon step.
    The per-point forecast is the mixture over modes of that point's 2-D Gaussians."""
    g = torch.Generator().manual_seed(seed)
    y, m = targets(w, cur)
    rows = []
    for k in ks:
        msk = eval_mask(w, k)
        logpi, mu, a, b, off = (t[torch.from_numpy(msk)] for t in out)
        mu, a, b, off = mu[:, :, k - 1], a[:, :, k - 1], b[:, :, k - 1], off[:, :, k - 1]   # (N,M,J,...)
        yt = torch.from_numpy(y[msk][:, k - 1])                                               # (N,J,2)
        N, M, J = a.shape

        def mix_logpdf(pt):                                    # pt (N,S,J,2) -> (N,S,J)
            d = pt[:, :, None] - mu[:, None]                   # (N,S,M,J,2)
            z1 = d[..., 0] / a[:, None]
            z2 = (d[..., 1] - off[:, None] * z1) / b[:, None]
            lp = -0.5 * (z1 ** 2 + z2 ** 2) - torch.log(a[:, None]) - torch.log(b[:, None]) - math.log(2 * math.pi)
            return torch.logsumexp(logpi[:, None, :, None] + lp, 2)
        mode = torch.multinomial(logpi.exp(), S, replacement=True, generator=g)              # (N,S)
        e = torch.randn(N, S, J, 2, generator=g)
        idx = mode[:, :, None].expand(N, S, J)
        am, bm, om = (torch.gather(t, 1, idx) for t in (a, b, off))
        mm = torch.gather(mu, 1, idx[..., None].expand(N, S, J, 2))
        samp = torch.stack([mm[..., 0] + am * e[..., 0], mm[..., 1] + om * e[..., 0] + bm * e[..., 1]], -1)
        ls = mix_logpdf(samp)                                                                  # (N,S,J)
        lt = mix_logpdf(yt[:, None])[:, 0]                                                     # (N,J)
        for lv in levels:
            thr = torch.quantile(ls, 1 - lv, dim=1)                                            # (N,J)
            inside = (lt >= thr).float().mean(0).numpy()                                       # per keypoint
            rows.append([k, lv, int(N)] + [round(float(v), 4) for v in inside] + [round(float(inside.mean()), 4)])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepared", default="logs/prepared")
    ap.add_argument("--out", default="logs/forecast_results/mgru")
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
    keep = wt["hmiss"][:, -1].sum(1) == 0
    wt = {k: v[keep] for k, v in wt.items()}
    xtr, ctr = features(wt); ytr, mtr = targets(wt, ctr)
    xv, cv = features(wv)
    data = (xtr, ytr, mtr, wv, xv, cv)
    print(f"train {len(xtr)} windows, val {len(xv)} windows")

    grid = [dict(M=3, hidden=64, layers=2, lr=1e-3, batch=256)] if a.quick else \
        [dict(M=M, hidden=h, layers=2, lr=1e-3, batch=256) for M in (1, 3, 5) for h in (64, 128)]
    results = []
    for cfg in grid:
        name = f"M{cfg['M']}_h{cfg['hidden']}_l{cfg['layers']}_lr{cfg['lr']:g}"
        t0 = time.time()
        model, best, curve = train_one(cfg, data, a.seed, a.max_epochs, a.patience,
                                       os.path.join(a.out, f"learning_curve_{name}.csv"))
        out = run(model, xv)
        pi, pos = to_positions(out, cv)
        top = pos[np.arange(len(pos)), pi.argmax(1)]
        k1 = ks[HORIZONS_S.index(1.0)]
        mk = eval_mask(wv, k1)
        ad, _ = ade_fde_px(top[mk], {"fut": wv["fut"][mk]}, k1)
        results.append((best, name, cfg, model, curve, float(ad.mean())))
        print(f"  {name:<22} best val NLL {best:6.3f} nats/point   most-likely-mode mean ADE at 1.0 s {ad.mean():5.1f} px"
              f"  ({len(curve)} epochs, {time.time() - t0:.0f} s)", flush=True)
    with open(os.path.join(a.out, "search.csv"), "w", newline="") as f:
        cw = csv.writer(f); cw.writerow(["config", "val_nll_nats_per_point_px", "val_top_mode_mean_ade_1s_px", "epochs"])
        cw.writerows([(n, round(b, 4), round(d, 2), len(c)) for b, n, _, _, c, d in results])
    best, name, cfg, model, curve, _ = min(results, key=lambda r: r[0])
    torch.save(model.state_dict(), os.path.join(a.out, "mgru_best.pt"))
    json.dump({"config": cfg, "name": name, "n_in": int(xtr.shape[-1]), "K": int(ytr.shape[1]), "M": cfg["M"],
               "keypoints": FORECAST_KP, "dt": m["dt"], "history_steps": m["history_steps"], "min_std_px": MIN_STD * 100,
               "covariance": "per keypoint and step 2x2 Cholesky [[a,0],[c,b]], a,b = softplus + 1 px; blocks independent",
               "seed": a.seed, "chosen_on": "val NLL", "val_nll_nats_per_point_px": round(best, 4)},
              open(os.path.join(a.out, "mgru_best.json"), "w"), indent=1)
    print(f"\nChosen on val NLL: {name}")

    # ---- val report
    out = run(model, xv)
    pi, pos = to_positions(out, cv)
    top = pos[np.arange(len(pos)), pi.argmax(1)]
    mean = (pi[:, :, None, None, None] * pos).sum(1)
    det = {}
    p_det = os.path.join(os.path.dirname(a.out), "gru", "gru_val.csv")
    if os.path.exists(p_det):
        det = {float(r["horizon_s"]): float(r["mean_ade_px"]) for r in csv.DictReader(open(p_det))}
    print(f"\n{'horizon':>8}{'n':>7}" + "".join(f"{k + ' ADE':>12}" for k in FORECAST_KP)
          + f"{'mean ADE':>10}{'mean FDE':>10}{'mixture mean':>14}{'minADE':>9}{'det. GRU':>10}{'NLL':>8}")
    rows = []
    for h, k in zip(HORIZONS_S, ks):
        msk = eval_mask(wv, k)
        ad, fd = ade_fde_px(top[msk], {"fut": wv["fut"][msk]}, k)
        am, _ = ade_fde_px(mean[msk], {"fut": wv["fut"][msk]}, k)
        per_mode = np.stack([ade_fde_px(pos[msk][:, q], {"fut": wv["fut"][msk]}, k)[0].mean(1) for q in range(cfg["M"])], 1)
        mn = per_mode.min(1).mean()
        nll, _ = nll_px(out, wv, cv, k)
        print(f"{h:>7}s{msk.sum():>7}" + "".join(f"{v:>12.1f}" for v in ad.mean(0)) + f"{ad.mean():>10.1f}{fd.mean():>10.1f}"
              f"{am.mean():>14.1f}{mn:>9.1f}{det.get(h, float('nan')):>10.1f}{nll:>8.2f}")
        rows.append(["val", "MGRU " + name, h, int(msk.sum())] + [round(float(v), 2) for v in ad.mean(0)]
                    + [round(float(v), 2) for v in fd.mean(0)] + [round(float(ad.mean()), 2), round(float(fd.mean()), 2),
                                                                  round(float(am.mean()), 2), round(float(mn), 2), round(nll, 3)])
    with open(os.path.join(a.out, "mgru_val.csv"), "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["set", "model", "horizon_s", "n_windows"] + [f"{k}_ade_px" for k in FORECAST_KP]
                    + [f"{k}_fde_px" for k in FORECAST_KP] + ["mean_ade_px", "mean_fde_px", "mixture_mean_ade_px",
                                                             "min_ade_oracle_px", "nll_nats_per_point_px"])
        cw.writerows(rows)
    print(f"\nMode probabilities on val: mean largest {pi.max(1).mean():.2f}; "
          f"windows where the top mode has < 0.6: {(pi.max(1) < 0.6).mean():.0%}")

    cal = hpd_coverage(out, wv, cv, ks)
    with open(os.path.join(a.out, "mgru_calibration.csv"), "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["horizon_step", "nominal", "n_windows"] + [f"{k}_coverage" for k in FORECAST_KP] + ["mean_coverage"])
        cw.writerows(cal)
    print("\nCalibration: share of true points inside the 50 / 80 / 95% region (mean over elbow, wrist, hand)")
    for h, k in zip(HORIZONS_S, ks):
        r = [c for c in cal if c[0] == k]
        print(f"  {h:>4} s   " + "   ".join(f"{c[1]:.0%}: {c[-1]:.0%}" for c in r))

    model.eval(); torch.set_num_threads(1); ts = []
    with torch.no_grad():
        for i in range(300):
            t0 = time.perf_counter(); model(torch.from_numpy(xv[i % len(xv)][None])); ts.append((time.perf_counter() - t0) * 1e3)
    ts = np.array(ts[20:])
    print(f"\nInference time, one window, 1 CPU thread: median {np.median(ts):.2f} ms, p95 {np.percentile(ts, 95):.2f} ms")
    json.dump({"median_ms": round(float(np.median(ts)), 3), "p95_ms": round(float(np.percentile(ts, 95)), 3)},
              open(os.path.join(a.out, "mgru_inference_time.json"), "w"))

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        for b_, n, _, _, c, _ in results:
            c = np.array(c)
            ax[0].plot(c[:, 0], c[:, 1], label=n, lw=1); ax[1].plot(c[:, 0], c[:, 2], label=n, lw=1)
        ax[0].set(xlabel="epoch", ylabel="train NLL (nats per point)", title="Training loss")
        ax[1].set(xlabel="epoch", ylabel="val NLL (nats per point)", title="Validation NLL")
        ax[1].legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(a.out, "learning_curves.png"), dpi=150)
    except ImportError:
        pass
    print(f"Saved in {a.out}/")


if __name__ == "__main__":
    main()
