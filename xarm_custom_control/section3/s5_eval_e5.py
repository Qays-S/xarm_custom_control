#!/usr/bin/env python3
"""
Blueprint E5 (degraded viewing): how much worse do the forecasters get when the camera view is
degraded? The models are the ones already chosen on D_val; nothing is trained or tuned on E5.

Models   current pose, constant velocity (n from baselines_val.json), GRU (gru_best.pt),
         STS-GCN fine-tuned from AMASS (stsgcn_finetuned_best.pt; needs --manicast-dir)
Sets     val  = the clean validation windows (E1-E4 round 2, first third), as reported before
         e5   = every E5 window (logs/prepared_e5, made by s5_prepare.py ... --out logs/prepared_e5)
Windows  the same rule as everywhere else (current pose and future observed), all phases and the
         moving phase only. E5 has much longer holds than E1-E4 (about 49% vs 24% of windows), which
         makes "stay where you are" look good overall, so the moving-phase numbers and the within-E5
         control (none/normal) are the fair comparison.
Conditions  what was done during the reach (perturb: none / cover / head) and the lighting
         (normal = room lights at half, dim = all lights off), from reaches.csv.
Tracking quality per condition: missing rates, how often the right wrist jumps more than 60 px between
frames (in HEAD reaches mostly a left/right swap with the raised left hand), and the detector's
confidence at those jumps. In E5 the "truth" is the camera's own detection, so a swap corrupts the
target too: E5 errors are disagreement with what the camera reports, not true arm error.

Usage
  python3 s5_eval_e5.py --manicast-dir ~/manicast
Outputs in logs/forecast_results/e5/
  e5_results.csv      ADE/FDE per model, set, phase (all / moving) and horizon (elbow, wrist)
  e5_by_condition.csv wrist and hand ADE at 1.0 s per model and E5 condition (moving windows)
  e5_summary.json     the numbers used in the paper, including the tracking-quality table
"""
import argparse
import csv
import json
import os

import numpy as np
import torch

from s5_baselines import const_vel, current_pose
from s5_data import ade_fde_px, eval_mask, load, windows
from s5_gru import GRUForecaster, features, predict

HZ = [0.25, 0.5, 1.0]                 # STS-GCN forecasts 1.0 s, so all models are compared up to 1.0 s
KS = [4, 8, 16]
SIZE = np.array([1280.0, 720.0])
CONDS = [(p, l) for l in ("normal", "dim") for p in ("none", "cover", "head")]


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def e5_labels(m, w):
    """perturb and lighting of the reach each window's current step belongs to."""
    lab = {n: list(csv.DictReader(open(os.path.join(i["path"], "reaches.csv")))) for n, i in m["sessions"].items()}
    pert, light = [], []
    for name, reach in zip(w["session"], w["reach"]):
        if reach == 0:
            pert.append("setup"); light.append("normal" if name.endswith("_s01") else "dim")
        else:
            r = lab[name][reach - 1]
            pert.append(r["perturb"]); light.append(r["lighting"])
    return np.array(pert), np.array(light)


def tracking_quality(m, jump_px=60.0):
    """Per E5 condition, over raw frames between cue and arrival."""
    acc = {}
    for name, info in m["sessions"].items():
        d = info["path"]
        F = list(csv.DictReader(open(os.path.join(d, "frames.csv"))))
        R = list(csv.DictReader(open(os.path.join(d, "reaches.csv"))))
        t = np.array([fnum(r["t_read_end"]) for r in F])
        W = np.array([[fnum(r["R_wrist_u"]), fnum(r["R_wrist_v"])] for r in F])
        conf = np.array([fnum(r["R_wrist_conf"]) for r in F])
        wm = np.array([r["R_wrist_missing"] == "1" for r in F])
        hm = np.array([r["R_hand_missing"] == "1" for r in F])
        step = np.r_[np.nan, np.linalg.norm(np.diff(W, axis=0), axis=1)]
        for r in R:
            t0 = fnum(r["t_cue"]); t1 = fnum(r["t_arrive"]) if r["t_arrive"] else t0 + 8
            s = (t >= t0) & (t <= t1)
            a = acc.setdefault(f"{r['perturb']}/{r['lighting']}", {"wm": [], "hm": [], "step": [], "conf": []})
            a["wm"].append(wm[s]); a["hm"].append(hm[s]); a["step"].append(step[s]); a["conf"].append(conf[s])
    out = {}
    for c, a in acc.items():
        wm_, hm_, st, cf = (np.concatenate(a[k]) for k in ("wm", "hm", "step", "conf"))
        big, calm = st > jump_px, st < 10
        out[c] = {"frames": int(len(wm_)), "wrist_missing": round(float(wm_.mean() * 100), 1),
                  "hand_missing": round(float(hm_.mean() * 100), 1),
                  "wrist_jumps": round(float(big.mean() * 100), 2),
                  "conf_at_jumps": round(float(np.nanmedian(cf[big])), 2) if big.any() else float("nan"),
                  "conf_steady": round(float(np.nanmedian(cf[calm])), 2)}
    return out


class StsGcn:
    """Runs the fine-tuned STS-GCN on any set, with s5_stsgcn's own helpers and training-set scale."""

    def __init__(self, manicast_dir, m_train, sess_train, ckpt):
        import s5_stsgcn as S
        self.S = S
        raw = {n: S.RawTracks(i["path"]) for n, i in m_train["sessions"].items()}
        pt = {n: sess_train[n]["t"] for n in m_train["sessions"]}
        ref, cw = S.comad_reference(manicast_dir)
        Pt, _, _, _ = S.build(m_train, raw, pt, "train")
        self.fr = S.Frame(ref, cw, Pt)
        self.net = S.make_model(manicast_dir, "scratch")
        self.net.load_state_dict(torch.load(ckpt, map_location="cpu"))

    def run(self, m, sess, setname, w):
        S = self.S
        raw = {n: S.RawTracks(i["path"]) for n, i in m["sessions"].items()}
        pt = {n: sess[n]["t"] for n in m["sessions"]}
        P, _, names, idx = S.build(m, raw, pt, setname)
        assert (names == w["session"]).all() and (idx == w["anchor"]).all()
        X, _, _, origin, usable = S.tensors(self.fr, P)
        cur = np.nan_to_num(self.fr.fill_history(P)[:, -1])
        p = S.to_eval16(self.fr, S.forward(self.net, X), origin, cur)
        hold = np.repeat((cur[:, [S.R_ELBOW, S.R_WRIST]] / SIZE)[:, None], 16, 1)
        p[~usable, :, :2] = hold[~usable]
        return p                                                    # (N,16,3,2), hand = NaN


def scores(pred, w, mask):
    out = {}
    for h, k in zip(HZ, KS):
        mk = mask(w, k)
        a, f = ade_fde_px(pred[mk][:, :k], {"fut": w["fut"][mk][:, :k]}, k)
        out[h] = dict(n=int(mk.sum()), elbow=float(a[:, 0].mean()), wrist=float(a[:, 1].mean()),
                      elbow_fde=float(f[:, 0].mean()), wrist_fde=float(f[:, 1].mean()))
    return out


def moving(w, k):
    return eval_mask(w, k) & (w["phase"] == "moving")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manicast-dir", default=os.path.expanduser("~/manicast"))
    ap.add_argument("--prepared", default="logs/prepared")
    ap.add_argument("--prepared-e5", default="logs/prepared_e5")
    ap.add_argument("--results", default="logs/forecast_results")
    ap.add_argument("--out", default="logs/forecast_results/e5")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    m0, s0 = load(a.prepared)
    m5, s5 = load(a.prepared_e5)
    sets = {"val": (m0, s0, windows(m0, s0, "val")), "e5": (m5, s5, windows(m5, s5, "e5"))}
    for m, s, w in sets.values():                                   # reach index of each window
        w["reach"] = np.concatenate([s[n]["reach"][w["anchor"][w["session"] == n]] for n in m["sessions"]
                                     if (w["session"] == n).any()])

    cvn = json.load(open(os.path.join(a.results, "baselines_val.json")))["cv_window_chosen"]
    gcfg = json.load(open(os.path.join(a.results, "gru", "gru_best.json")))
    gru = GRUForecaster(gcfg["n_in"], gcfg["config"]["hidden"], gcfg["config"]["layers"], gcfg["K"])
    gru.load_state_dict(torch.load(os.path.join(a.results, "gru", "gru_best.pt"), map_location="cpu"))
    sts = None
    if os.path.exists(os.path.join(a.manicast_dir, "model", "manicast.py")):
        sts = StsGcn(a.manicast_dir, m0, s0, os.path.join(a.results, "stsgcn", "stsgcn_finetuned_best.pt"))
    else:
        print(f"(ManiCast not found in {a.manicast_dir}: STS-GCN is skipped)")

    preds = {}
    for sn, (m, s, w) in sets.items():
        x, cur = features(w)
        p = {"current pose": current_pose(w, 24), f"constant velocity (n={cvn})": const_vel(w, 24, cvn),
             "GRU": predict(gru, x, cur)}
        if sts:
            p["STS-GCN fine-tuned"] = sts.run(m, s, sn, w)
        preds[sn] = p

    summary = {"note": "ADE in px of the 1280x720 image; wrist = right wrist. In E5 the 'truth' is the camera's "
                       "own (degraded) detection, so errors mix forecast error and detection error.",
               "phase_share": {sn: {ph: round(float((w["phase"] == ph).mean()), 3) for ph in ("cued", "moving", "hold")}
                               for sn, (_, _, w) in sets.items()},
               "sets": {}}
    rows = []
    print("\nRight wrist ADE (px), windows whose current pose and future were observed")
    print(f"{'model':<28}{'set':<5}{'phase':<8}" + "".join(f"{f'{h} s':>9}" for h in HZ) + f"{'n (1.0 s)':>11}")
    for sn, (m, s, w) in sets.items():
        for ph, fn in (("all", eval_mask), ("moving", moving)):
            for model, pa in preds[sn].items():
                r = scores(pa, w, fn)
                print(f"{model:<28}{sn:<5}{ph:<8}" + "".join(f"{r[h]['wrist']:>9.1f}" for h in HZ) + f"{r[1.0]['n']:>11}")
                for h in HZ:
                    rows.append([sn, ph, model, h, r[h]["n"]] + [round(r[h][q], 2) for q in
                                                                ("elbow", "wrist", "elbow_fde", "wrist_fde")])
                summary["sets"].setdefault(sn, {}).setdefault(ph, {})[model] = \
                    {str(h): round(r[h]["wrist"], 2) for h in HZ} | {"n_1.0s": r[1.0]["n"]}
    with open(os.path.join(a.out, "e5_results.csv"), "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["set", "phase", "model", "horizon_s", "n_windows", "elbow_ade_px", "wrist_ade_px",
                     "elbow_fde_px", "wrist_fde_px"])
        cw.writerows(rows)

    m, s, w = sets["e5"]
    pert, light = e5_labels(m, w)
    cn = [f"{p_}/{l_}" for p_, l_ in CONDS]
    q = tracking_quality(m)
    summary["e5_tracking_quality"] = q
    print("\nE5 tracking quality per condition (raw frames during reaches, right wrist / hand)")
    print(f"{'':<28}" + "".join(f"{c:>14}" for c in cn))
    for key, lab, fmt in (("frames", "frames", "{:>14}"), ("wrist_missing", "wrist missing %", "{:>14.0f}"),
                          ("hand_missing", "hand missing %", "{:>14.0f}"), ("wrist_jumps", "wrist jumps > 60 px %", "{:>14.1f}"),
                          ("conf_at_jumps", "confidence at jumps", "{:>14.2f}"), ("conf_steady", "confidence otherwise", "{:>14.2f}")):
        print(f"{lab:<28}" + "".join(fmt.format(q[c][key]) if q[c][key] == q[c][key] else f"{'-':>14}" for c in cn))

    print("\nE5 by condition, MOVING windows: ADE (px) at 1.0 s, right wrist [hand]")
    print(f"{'model':<28}" + "".join(f"{c:>14}" for c in cn))
    em, crow = moving(w, 16), []
    for model, pa in preds["e5"].items():
        line = f"{model:<28}"
        for (p_, l_), c in zip(CONDS, cn):
            sel = (pert == p_) & (light == l_) & em
            ad, _ = ade_fde_px(pa[sel][:, :16], {"fut": w["fut"][sel][:, :16]}, 16)
            wr = float(ad[:, 1].mean())
            hd = float(ad[:, 2].mean()) if not np.isnan(ad[:, 2]).any() else float("nan")
            line += f"{wr:>8.1f}" + (f" [{hd:>3.0f}]" if hd == hd else " " * 6)
            crow.append([model, p_, l_, int(sel.sum()), round(wr, 2), round(hd, 2) if hd == hd else ""])
            summary.setdefault("e5_moving_1s", {}).setdefault(model, {})[c] = \
                {"wrist": round(wr, 2), "hand": round(hd, 2) if hd == hd else None}
        print(line)
    nmov = {c: int(((pert == p_) & (light == l_) & em).sum()) for (p_, l_), c in zip(CONDS, cn)}
    summary["e5_moving_windows"] = nmov
    print(f"{'windows':<28}" + "".join(f"{nmov[c]:>14}" for c in cn))
    with open(os.path.join(a.out, "e5_by_condition.csv"), "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["model", "perturb", "lighting", "n_moving_windows", "wrist_ade_1s_px", "hand_ade_1s_px"])
        cw.writerows(crow)
    json.dump(summary, open(os.path.join(a.out, "e5_summary.json"), "w"), indent=1)
    print(f"\nSaved {a.out}/")


if __name__ == "__main__":
    main()
