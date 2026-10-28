#!/usr/bin/env python3
"""
Blueprint Section V, step 30 (and Part II step 5): fine-tune the pre-trained STS-GCN forecaster
released with ManiCast on our camera recordings, following the ManiCast recipe.

Needs a local copy of the ManiCast repository (code + released checkpoints):
  git clone https://github.com/portal-cornell/manicast ~/manicast
Nothing from that repository is copied into ours; this script imports its model class and loads
its checkpoint files from --manicast-dir.

The released model is fixed in shape: 10 input frames -> 25 output frames at 25 Hz (0.4 s of
history, 1.0 s forecast), 7 joints in 3-D metres, centred on the waist at the last input frame:
  BackTop, LShoulderBack, RShoulderBack, LElbowOut, RElbowOut, LWristOut, RWristOut
How our 2-D camera keypoints are fed in:
  * joints   BackTop = midpoint of the two shoulders; the other six are MediaPipe's shoulders,
             elbows and wrists (both arms; the left arm mostly rests and is often missing)
  * timing   our raw detections are interpolated at 25 Hz (gaps longer than 0.25 s stay missing)
  * 2-D->3-D image u,v become the two horizontal axes (x = person's right, z = backwards), scaled
             so that the median shoulder width matches CoMaD's (0.32 m); the vertical axis is held
             at each joint's average CoMaD height. The pose is placed so BackTop sits where it
             sits on average in CoMaD. Only the horizontal axes are used in the loss and the score.
  * missing  inside the input, a missing joint is filled with its nearest observation in the
             window, or with its average offset from BackTop; missing targets are left out of the loss
Fine-tuning recipe (ManiCast): Adam, weight decay 1e-5; every update is one ordinary batch plus one
batch of "transition" windows (a reach starts or a switch is cued within the next 1.0 s); both
wrists are weighted 5x in the loss. The learning rate is tuned on D_val; each run stops when the
validation error has not improved for 10 epochs (at most 100).

Compared on exactly the same validation windows as s5_baselines.py and s5_gru.py (the forecast is
interpolated to the same 16 Hz future steps), for the right elbow and right wrist at 0.25, 0.5 and
1.0 s. Runs:
  zero-shot     released AMASS model, no fine-tuning
  fine-tuned    starts from the AMASS weights   (lr 3e-4, 1e-3, 3e-3; chosen on val)
  from scratch  same network, random start      (lr 1e-3, 1e-2; chosen on val) - shows what pre-training adds

Usage
  python3 s5_stsgcn.py --manicast-dir ~/manicast           # 5 training runs, roughly 45-75 min on a laptop CPU
  python3 s5_stsgcn.py --manicast-dir ~/manicast --quick   # one run per start (about 25 min), for a quick check
"""
import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np
import torch

from s5_data import ade_fde_px, eval_mask, load, windows

FPS, DT, N_IN, N_OUT, MAX_GAP = 25, 0.04, 10, 25, 0.25
CJ = ["BackTop", "LShoulderBack", "RShoulderBack", "LElbowOut", "RElbowOut", "LWristOut", "RWristOut"]
OURS = [None, "L_shoulder", "R_shoulder", "L_elbow", "R_elbow", "L_wrist", "R_wrist"]   # None = shoulder midpoint
R_ELBOW, R_WRIST = 4, 6
SIZE = np.array([1280.0, 720.0])


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


# ------------------------------------------------------------------ CoMaD reference geometry
def comad_reference(manicast_dir):
    """Average joint positions relative to the waist (WaistLBack) over CoMaD handover/train."""
    mapping = json.load(open(os.path.join(manicast_dir, "data", "mapping.json")))
    J = [mapping[n] for n in CJ]
    rel, width = [], []
    for f in sorted(glob.glob(os.path.join(manicast_dir, "data", "comad_data", "handover", "train", "*.json"))):
        d = json.load(open(f))
        for who in d:
            a = np.asarray(d[who], float)
            if a.ndim != 3:
                continue
            ok = (np.abs(a[:, J + [21]]).sum(-1) > 0).all(1)
            r = a[ok][:, J] - a[ok][:, 21:22]
            rel.append(r)
            width.append(np.linalg.norm(a[ok][:, J[2], [0, 2]] - a[ok][:, J[1], [0, 2]], axis=1))
    rel = np.concatenate(rel)
    return rel.mean(0), float(np.median(np.concatenate(width)))      # (7,3), metres


# ------------------------------------------------------------------ our tracks at 25 Hz
class RawTracks:
    def __init__(self, session_dir):
        meta = json.load(open(os.path.join(session_dir, "meta.json")))
        self.W, self.H = meta.get("actual_size", [1280, 720])
        fr = [f for f in csv.DictReader(open(os.path.join(session_dir, "frames.csv"))) if f["phase"] != "setup"]
        self.t = np.array([fnum(f["t_read_end"]) for f in fr])
        self.k = {}
        for name in set(o for o in OURS if o):
            u = np.array([fnum(f[f"{name}_u"]) for f in fr])
            v = np.array([fnum(f[f"{name}_v"]) for f in fr])
            obs = np.array([f[f"{name}_missing"] == "0" for f in fr]) & ~np.isnan(u)
            self.k[name] = (self.t[obs], u[obs], v[obs])
        reaches = list(csv.DictReader(open(os.path.join(session_dir, "reaches.csv"))))
        ev = [fnum(r["t_depart"]) for r in reaches] + [fnum(r["t_switch_cue"]) for r in reaches]
        self.events = np.sort(np.array([e for e in ev if e == e]))

    def at(self, name, tq):
        to, u, v = self.k[name]
        out = np.full(tq.shape + (2,), np.nan)
        if len(to) < 2:
            return out
        i = np.searchsorted(to, tq, side="right")
        ip, inx = np.clip(i - 1, 0, len(to) - 1), np.clip(i, 0, len(to) - 1)
        exact = (i > 0) & (to[ip] == tq)
        ok = exact | ((i > 0) & (i < len(to)) & (to[inx] - to[ip] <= MAX_GAP))
        out[ok, 0] = np.interp(tq[ok], to, u)
        out[ok, 1] = np.interp(tq[ok], to, v)
        return out

    def joints(self, tq):
        """(…,7,2) pixel positions in CoMaD joint order, NaN = missing."""
        p = {n: self.at(n, tq) for n in self.k}
        bt = (p["L_shoulder"] + p["R_shoulder"]) / 2
        return np.stack([bt] + [p[n] for n in OURS[1:]], axis=-2)


def build(m, sess_dirs, prepared_t, setname):
    """Positions (N,35,7,2) px at t0-0.36 … t0+1.0 s for every anchor of one set, plus labels."""
    P, trans, names, idxs = [], [], [], []
    offs = np.arange(-(N_IN - 1), N_OUT + 1) * DT
    for name, info in m["sessions"].items():
        idx = np.array(info["anchor_lists"].get(setname, []), int)
        if not len(idx):
            continue
        rt = sess_dirs[name]
        t0 = prepared_t[name][idx]
        P.append(rt.joints(t0[:, None] + offs[None]))
        e = rt.events
        j = np.searchsorted(e, t0, side="right")
        nxt = np.where(j < len(e), e[np.clip(j, 0, len(e) - 1)], np.inf)
        trans.append(nxt - t0 <= N_OUT * DT)
        names += [name] * len(idx); idxs.append(idx)
    return np.concatenate(P), np.concatenate(trans), np.array(names), np.concatenate(idxs)


# ------------------------------------------------------------------ to and from the model's frame
class Frame:
    def __init__(self, ref_rel, comad_width, train_P):
        self.ref = ref_rel                                       # (7,3)
        w = np.linalg.norm(train_P[:, N_IN - 1, 2] - train_P[:, N_IN - 1, 1], axis=-1)
        self.s = comad_width / float(np.nanmedian(w))            # metres per pixel
        rel = train_P[:, N_IN - 1] - train_P[:, N_IN - 1, :1]    # offsets from BackTop, px
        self.default_off = np.nanmedian(rel, axis=0)             # (7,2)

    def fill_history(self, P):
        h = P[:, :N_IN].copy()
        for j in range(7):
            x = h[:, :, j]
            for t in range(1, N_IN):
                b = np.isnan(x[:, t, 0]); x[b, t] = x[b, t - 1]
            for t in range(N_IN - 2, -1, -1):
                b = np.isnan(x[:, t, 0]); x[b, t] = x[b, t + 1]
            h[:, :, j] = x
        for j in range(1, 7):                                     # never seen in the window
            b = np.isnan(h[:, :, j, 0])
            h[:, :, j][b] = (h[:, :, 0] + self.default_off[j])[b]
        return h

    def to3d(self, px, origin):
        """px (N,T,7,2), origin (N,2) = BackTop at the last input frame -> (N,T,7,3) metres."""
        d = px - origin[:, None, None]
        x = -d[..., 0] * self.s + self.ref[0, 0]
        z = -d[..., 1] * self.s + self.ref[0, 2]
        y = np.broadcast_to(self.ref[:, 1], x.shape)
        return np.stack([x, y, z], -1)

    def to_px(self, xyz, origin):
        u = origin[:, None, None, 0] - (xyz[..., 0] - self.ref[0, 0]) / self.s
        v = origin[:, None, None, 1] - (xyz[..., 2] - self.ref[0, 2]) / self.s
        return np.stack([u, v], -1)


def tensors(fr, P):
    """Model input (N,3,10,7), targets (N,25,7,3), target mask (N,25,7), origin, usable flag."""
    h = fr.fill_history(P)
    origin = h[:, -1, 0]
    usable = ~np.isnan(P[:, N_IN - 1, 0, 0]) & ~np.isnan(h).any((1, 2, 3))
    origin = np.nan_to_num(origin)
    X = fr.to3d(np.nan_to_num(h), origin).transpose(0, 3, 1, 2)
    fut = P[:, N_IN:]
    Y = fr.to3d(np.nan_to_num(fut), origin)
    M = ~np.isnan(fut[..., 0])
    return X.astype(np.float32), Y.astype(np.float32), M.astype(np.float32), origin, usable


# ------------------------------------------------------------------ model
def make_model(manicast_dir, init):
    sys.path.insert(0, os.path.abspath(manicast_dir))
    from model.manicast import ManiCast                          # STS-GCN, as released with ManiCast
    net = ManiCast(3, N_IN, N_OUT, 0.1, 7, 4, [3, 3], 0.0)
    if init != "scratch":
        net.load_state_dict(torch.load(os.path.join(manicast_dir, "model_checkpoints", init), map_location="cpu"))
    return net


def forward(net, X, bs=1024):
    net.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(net(torch.from_numpy(X[i:i + bs])).permute(0, 1, 3, 2).numpy())
    return np.concatenate(out)                                    # (N,25,7,3)


def to_eval16(fr, pred3d, origin, cur_px, K16=16):
    """Right elbow and wrist forecast at the 16 Hz future steps, as (N,16,3,2) normalised (hand = NaN)."""
    px = fr.to_px(pred3d, origin)                                 # (N,25,7,2)
    traj = np.concatenate([cur_px[:, None], px], 1)               # frame 0 = now
    tau = np.arange(1, K16 + 1) * 0.0625 / DT
    i0 = np.floor(tau).astype(int); a = (tau - i0)[None, :, None, None]
    i1 = np.minimum(i0 + 1, N_OUT)
    q = traj[:, i0] * (1 - a) + traj[:, i1] * a                   # (N,16,7,2)
    out = np.full((len(q), K16, 3, 2), np.nan)
    out[:, :, 0], out[:, :, 1] = q[:, :, R_ELBOW], q[:, :, R_WRIST]
    return out / SIZE


def score(pred16, wv, ks):
    """Mean ADE (px) of right elbow and right wrist at each horizon, on the shared eval windows."""
    r = {}
    for k in ks:
        mk = eval_mask(wv, k)
        a, f = ade_fde_px(pred16[mk], {"fut": wv["fut"][mk][:, :k]}, k)
        r[k] = dict(n=int(mk.sum()), elbow=float(a[:, 0].mean()), wrist=float(a[:, 1].mean()),
                    elbow_fde=float(f[:, 0].mean()), wrist_fde=float(f[:, 1].mean()))
    return r


def train(net, data, cfg, ks, log_path, seed, max_epochs, patience):
    torch.manual_seed(seed); np.random.seed(seed)
    Xt, Yt, Mt, tr, fr, Xv, ov, cv, wv = data
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    wj = torch.tensor([1, 1, 1, 1, 1, cfg["wrist_w"], cfg["wrist_w"]], dtype=torch.float32)[None, None, :, None]
    X, Y, M = map(torch.from_numpy, (Xt, Yt, Mt))
    T = torch.from_numpy(np.where(tr)[0])
    bs = 256
    nb = (len(X) + bs - 1) // bs
    tbs = max(1, len(T) // nb)

    def step(b):
        p = net(X[b]).permute(0, 1, 3, 2)                          # (B,25,7,3)
        d = (p - Y[b])[..., [0, 2]] * wj                            # horizontal axes only
        e = torch.linalg.norm(d, dim=-1) * M[b]
        loss = e.sum() / M[b].sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step()
        return loss.item()

    best, best_state, bad, curve = 1e9, None, 0, []
    for ep in range(1, max_epochs + 1):
        net.train()
        perm, tperm = torch.randperm(len(X)), T[torch.randperm(len(T))]
        tot = []
        for i in range(nb):
            tot.append(step(perm[i * bs:(i + 1) * bs]))
            if len(T):
                j = (i * tbs) % max(1, len(T) - tbs + 1)
                tot.append(step(tperm[j:j + tbs]))
        r = score(to_eval16(fr, forward(net, Xv), ov, cv), wv, ks)
        v = float(np.mean([(r[k]["elbow"] + r[k]["wrist"]) / 2 for k in ks]))
        curve.append((ep, float(np.mean(tot)), v))
        if v < best - 1e-3:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["epoch", "train_loss_m", "val_mean_ade_px"]); w.writerows(curve)
    net.load_state_dict(best_state)
    return net, best, curve


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manicast-dir", default=os.path.expanduser("~/manicast"))
    ap.add_argument("--prepared", default="logs/prepared")
    ap.add_argument("--out", default="logs/forecast_results/stsgcn")
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if not os.path.exists(os.path.join(a.manicast_dir, "model", "manicast.py")):
        sys.exit(f"ManiCast not found in {a.manicast_dir}. Run: git clone https://github.com/portal-cornell/manicast {a.manicast_dir}")
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    m, sess = load(a.prepared)
    ks = [4, 8, 16]                                                # 0.25, 0.5, 1.0 s at 16 Hz
    hz = [0.25, 0.5, 1.0]
    raw = {n: RawTracks(info["path"]) for n, info in m["sessions"].items()}
    pt = {n: sess[n]["t"] for n in m["sessions"]}
    ref, cw = comad_reference(a.manicast_dir)

    Pt, trt, _, _ = build(m, raw, pt, "train")
    Pv, _, nv, iv = build(m, raw, pt, "val")
    fr = Frame(ref, cw, Pt)
    Xt, Yt, Mt, _, ut = tensors(fr, Pt)
    Xt, Yt, Mt, trt = Xt[ut], Yt[ut], Mt[ut], trt[ut]
    Xv, _, _, ov, uv = tensors(fr, Pv)
    wv = windows(m, sess, "val")
    cv = np.nan_to_num(fr.fill_history(Pv)[:, -1])                # current positions, px
    print(f"train {len(Xt)} windows ({trt.mean():.0%} transitions), val {len(Xv)} windows "
          f"({(~uv).sum()} without a usable input pose -> current pose used)")
    print(f"scale {fr.s * 1000:.2f} mm per px (shoulder width matched to CoMaD's {cw:.3f} m)")
    data = (Xt, Yt, Mt, trt, fr, Xv, ov, cv, wv)

    def predict16(net):
        p = to_eval16(fr, forward(net, Xv), ov, cv)
        hold = np.repeat((cv[:, [R_ELBOW, R_WRIST]] / SIZE)[:, None], 16, 1)
        p[~uv, :, :2] = hold[~uv]
        return p

    results, rows, curves = {}, [], {}
    zs = make_model(a.manicast_dir, "base")
    results["zero-shot (AMASS weights)"] = (score(predict16(zs), wv, ks), None)

    grid = {"base": [1e-3] if a.quick else [3e-4, 1e-3, 3e-3],      # fine-tuning: ManiCast used 1e-4 on CoMaD
            "scratch": [1e-2] if a.quick else [1e-3, 1e-2]}           # from scratch: ManiCast pre-trained at 1e-2
    search = []
    for init in ("base", "scratch"):
        best = None
        for lr in grid[init]:
            cfg = dict(lr=lr, wrist_w=5)
            name = f"{'finetune' if init == 'base' else 'scratch'}_lr{lr:g}"
            t0 = time.time()
            net, b, curve = train(make_model(a.manicast_dir, init), data, cfg, ks,
                                  os.path.join(a.out, f"learning_curve_{name}.csv"), a.seed, a.max_epochs, a.patience)
            curves[name] = curve
            search.append((name, round(b, 3), len(curve)))
            print(f"  {name:<20} best val ADE {b:6.2f} px  ({len(curve)} epochs, {time.time() - t0:.0f} s)")
            if best is None or b < best[0]:
                best = (b, name, cfg, net)
        label = "fine-tuned from AMASS" if init == "base" else "same network, from scratch"
        results[label] = (score(predict16(best[3]), wv, ks), best)
        if init == "base":
            torch.save(best[3].state_dict(), os.path.join(a.out, "stsgcn_finetuned_best.pt"))
            json.dump({"name": best[1], "config": best[2], "start": "ManiCast checkpoint 'base' (AMASS)",
                       "frames_in": N_IN, "frames_out": N_OUT, "fps": FPS, "scale_mm_per_px": fr.s * 1000,
                       "comad_shoulder_width_m": cw, "chosen_on": "val", "val_mean_ade_px": round(best[0], 3),
                       "seed": a.seed}, open(os.path.join(a.out, "stsgcn_finetuned_best.json"), "w"), indent=1)
    with open(os.path.join(a.out, "search.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["run", "best_val_mean_ade_px", "epochs"]); w.writerows(search)

    # earlier models on the same windows, from their saved results
    prev = {}
    for path, key in (("logs/forecast_results/baselines_val.csv", None), ("logs/forecast_results/gru/gru_val.csv", None)):
        if os.path.exists(path):
            for r in csv.DictReader(open(path)):
                nm = r["model"].split(" h")[0] if r["model"].startswith("GRU") else r["model"]
                prev.setdefault(nm, {})[float(r["horizon_s"])] = (float(r["elbow_ade_px"]), float(r["wrist_ade_px"]))

    print(f"\nRight elbow / right wrist ADE (px) on the validation set")
    print(f"{'model':<32}" + "".join(f"{f'{h} s':>16}" for h in hz))
    for nm, d in prev.items():
        print(f"{nm:<32}" + "".join(f"{d[h][0]:>8.1f}{d[h][1]:>8.1f}" if h in d else f"{'':>16}" for h in hz))
    for nm, (r, _) in results.items():
        print(f"{'STS-GCN ' + nm:<32}" + "".join(f"{r[k]['elbow']:>8.1f}{r[k]['wrist']:>8.1f}" for k in ks))
        for h, k in zip(hz, ks):
            rows.append(["val", "STS-GCN " + nm, h, r[k]["n"], round(r[k]["elbow"], 2), round(r[k]["wrist"], 2),
                         round(r[k]["elbow_fde"], 2), round(r[k]["wrist_fde"], 2)])
    with open(os.path.join(a.out, "stsgcn_val.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["set", "model", "horizon_s", "n_windows", "elbow_ade_px", "wrist_ade_px",
                                       "elbow_fde_px", "wrist_fde_px"]); w.writerows(rows)

    # wrist at 1.0 s by condition, for the fine-tuned model
    best_ft = results["fine-tuned from AMASS"][1][3]
    p = predict16(best_ft)
    line = "Fine-tuned STS-GCN, right wrist ADE at 1.0 s:"
    for c in sorted(set(wv["cond"])):
        sel = (wv["cond"] == c) & eval_mask(wv, 16)
        ad, _ = ade_fde_px(p[sel], {"fut": wv["fut"][sel][:, :16]}, 16)
        line += f"  {c} {ad[:, 1].mean():.1f}"
    print("\n" + line)

    best_ft.eval(); torch.set_num_threads(1); ts = []
    with torch.no_grad():
        for i in range(220):
            t0 = time.perf_counter(); best_ft(torch.from_numpy(Xv[i % len(Xv)][None])); ts.append((time.perf_counter() - t0) * 1e3)
    ts = np.array(ts[20:])
    print(f"Inference time, one window, 1 CPU thread: median {np.median(ts):.2f} ms, p95 {np.percentile(ts, 95):.2f} ms")
    json.dump({"median_ms": round(float(np.median(ts)), 3), "p95_ms": round(float(np.percentile(ts, 95)), 3)},
              open(os.path.join(a.out, "stsgcn_inference_time.json"), "w"))

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        for n, c in curves.items():
            c = np.array(c); ls = "-" if n.startswith("finetune") else "--"
            ax[0].plot(c[:, 0], c[:, 1], ls, label=n, lw=1.2); ax[1].plot(c[:, 0], c[:, 2], ls, label=n, lw=1.2)
        zsv = np.mean([(results['zero-shot (AMASS weights)'][0][k]['elbow'] + results['zero-shot (AMASS weights)'][0][k]['wrist']) / 2 for k in ks])
        ax[1].axhline(zsv, color="gray", lw=1, ls=":", label="zero-shot")
        ax[0].set(xlabel="epoch", ylabel="train loss (m)", yscale="log", title="Training loss")
        ax[1].set(xlabel="epoch", ylabel="val mean ADE, elbow + wrist (px)", title="Validation ADE")
        ax[1].legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(a.out, "learning_curves.png"), dpi=150)
    except ImportError:
        pass
    print(f"Saved in {a.out}/")


if __name__ == "__main__":
    main()
