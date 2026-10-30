#!/usr/bin/env python3
"""
Blueprint Section VI, steps 36-40 (and Algorithm 1, steps 3-6): learned forecast-reliability estimator.

  r_t = f_phi(o_t, u_t, e_t),  r_t in [0, 1],  a calibrated score that the next 1.0 s of forecast error
  stays below l_max. It is a forecast-reliability score, not a probability that anything is safe.

Data use (nothing here touches D_test, which stays locked until the planner is frozen too)
  gate-training  D_val    fit phi; compare estimators by cross-validation over contiguous blocks of
                          reaches within each session (5 folds), so overlapping windows of one reach
                          never sit on both sides of a fold
  calibration    D_cal    isotonic calibration of r_t, and the trust threshold tau for a declared
                          false-trust budget (FTR <= 10% on D_cal)
  stress test    E5       degraded viewing, never used for fitting or calibration
Features and labels come from s6_features.py (frozen multimodal GRU). l_max = 32 px (about 4 cm at the
table, at 1.27 mm per px) was fixed on D_val before any estimator was fitted.

Estimators compared (step 38, H1): confidence-only (o_t), uncertainty-only (u_t), residual-only (e_t),
combined (all). Each is fitted as a logistic regression and as a small gradient-boosted tree model;
the one with the lower cross-validated log loss on D_val is kept. Single raw scores are also shown.
Metrics (Section IX): Brier score and log loss for z_rel, AUROC/AUPRC for detecting high-loss moments,
expected calibration error, false-trust rate at tau, Spearman correlation between r_t and the realised
loss (with a reach-clustered bootstrap interval), results per condition (step 39), and detection delay
for E4 target switches with false alarms per minute on E1 (Section IX).

Usage
  python3 s6_reliability.py
Outputs in logs/reliability/
  features_dev.npz, features_e5.npz     the feature tables
  estimators.csv                        cross-validated comparison on D_val, and E5 results
  by_condition.csv                      combined estimator per condition / E5 condition
  switch_detection.csv                  E4 switch detection delays
  reliability_model.pkl, summary.json   the frozen estimator, calibration, l_max and tau
  reliability_diagram.png, reliability_vs_loss.png
"""
import argparse
import csv
import json
import os
import pickle

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from s6_features import E_COLS, O_COLS, U_COLS, build

L_MAX = 32.0
FTR_BUDGET = 0.10
SETS = {"confidence-only (o)": O_COLS, "uncertainty-only (u)": U_COLS, "residual-only (e)": E_COLS,
        "combined (o+u+e)": O_COLS + U_COLS + E_COLS}
RAW = {"detector confidence": ("conf_recent_mean", +1), "between-mode spread": ("between_hand_px", -1),
       "top-mode probability": ("top_prob", +1), "within-mode spread": ("within_std_hand_px", -1),
       "0.25 s residual": ("resid_4_px", -1)}


def sub(T, m):
    return {k: v[m] for k, v in T.items()}


def X(T, cols):
    return np.stack([T[c] for c in cols], 1).astype(float)


def make(kind):
    if kind == "logistic":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                             LogisticRegression(C=1.0, max_iter=2000))
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0,
                                          random_state=0)


def folds(T, n=5):
    """Contiguous blocks of reaches within each session -> fold id."""
    f = np.zeros(len(T["reach"]), int)
    for s in np.unique(T["session"]):
        m = T["session"] == s
        r = T["reach"][m]
        u = np.unique(r)
        rank = {v: i for i, v in enumerate(u)}
        f[m] = [min(n - 1, int(rank[v] * n / len(u))) for v in r]
    return f


def ece(p, y, bins=10):
    e, n = 0.0, len(p)
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.any():
            e += m.sum() / n * abs(p[m].mean() - y[m].mean())
    return e


def metrics(r, z, L, tau=None):
    bad = 1 - z
    out = {"n": int(len(z)), "unreliable_share": round(float(bad.mean()), 3),
           "AUROC": round(float(roc_auc_score(bad, 1 - r)), 3) if 0 < bad.sum() < len(bad) else None,
           "AUPRC": round(float(average_precision_score(bad, 1 - r)), 3) if bad.sum() else None,
           "Brier": round(float(brier_score_loss(z, r)), 4),
           "log_loss": round(float(log_loss(z, np.clip(r, 1e-4, 1 - 1e-4), labels=[0, 1])), 4),
           "ECE": round(float(ece(r, z)), 4),
           "spearman_r_vs_loss": round(float(spearmanr(r, L).correlation), 3)}
    if tau is not None:
        trust = r >= tau
        out["trusted_share"] = round(float(trust.mean()), 3)
        out["FTR"] = round(float(bad[trust].mean()), 3) if trust.any() else None
        out["missed_good"] = round(float((~trust & (z == 1)).sum() / max(1, (z == 1).sum())), 3)
    return out


def boot_spearman(r, L, groups, B=500, seed=0):
    rng = np.random.default_rng(seed)
    g = np.unique(groups)
    idx = {k: np.where(groups == k)[0] for k in g}
    vals = []
    for _ in range(B):
        pick = np.concatenate([idx[k] for k in rng.choice(g, len(g), replace=True)])
        vals.append(spearmanr(r[pick], L[pick]).correlation)
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def choose_tau(r, z, budget):
    """Smallest tau whose trusted set has false-trust rate <= budget (maximises the trusted share)."""
    for tau in np.unique(np.round(r, 4)):
        t = r >= tau
        if t.any() and (1 - z[t]).mean() <= budget:
            return float(tau)
    return 1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="logs/reliability")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    dev = build("logs/prepared", ["val", "cal"])
    e5 = build("logs/prepared_e5", ["e5"])
    np.savez_compressed(os.path.join(a.out, "features_dev.npz"), **dev)
    np.savez_compressed(os.path.join(a.out, "features_e5.npz"), **e5)
    lab = lambda T: sub(T, ~np.isnan(T["L_future"]))
    Vall = sub(dev, dev["set"] == "val")
    fall = folds(Vall)                                   # fold of every val row, labelled or not
    V, C, E = lab(Vall), lab(sub(dev, dev["set"] == "cal")), lab(e5)
    for T in (V, C, E):
        T["z"] = (T["L_future"] <= L_MAX).astype(int)
    fv = fall[~np.isnan(Vall["L_future"])]
    print(f"l_max = {L_MAX} px.  labelled windows: val {len(V['z'])} ({1 - V['z'].mean():.0%} unreliable), "
          f"cal {len(C['z'])} ({1 - C['z'].mean():.0%}), E5 {len(E['z'])} ({1 - E['z'].mean():.0%})")

    # ---- step 38: compare estimators on D_val by cross-validation
    rows, chosen, oof_store, fold_store = [], {}, {}, {}
    for name, cols in SETS.items():
        best = None
        for kind in ("logistic", "boosted trees"):
            oof, fold_models = np.zeros(len(V["z"])), []
            for k in range(fv.max() + 1):
                tr, te = fv != k, fv == k
                mdl = make(kind).fit(X(V, cols)[tr], V["z"][tr])
                cal = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(
                    mdl.predict_proba(X(C, cols))[:, 1], C["z"])                  # calibrated on D_cal
                oof[te] = cal.predict(mdl.predict_proba(X(V, cols)[te])[:, 1])
                fold_models.append((mdl, cal))
            ll = log_loss(V["z"], np.clip(oof, 1e-4, 1 - 1e-4))
            if best is None or ll < best[0]:
                best = (ll, kind, oof, fold_models)
        chosen[name] = best[1]
        fold_store[name] = best[3]
        oof_store[name] = best[2]
        mtr = metrics(best[2], V["z"], V["L_future"])
        rows.append(["val (cross-validated)", name, best[1]] + [mtr[k] for k in ("AUROC", "AUPRC", "Brier", "log_loss", "ECE", "spearman_r_vs_loss")])
    for name, (col, sign) in RAW.items():
        s = sign * V[col]
        s = np.where(np.isnan(s), np.nanmedian(s), s)
        bad = 1 - V["z"]
        rows.append(["val", f"raw score: {name}", "-", round(float(roc_auc_score(bad, -s)), 3),
                     round(float(average_precision_score(bad, -s)), 3), "", "", "", round(float(spearmanr(s, -V["L_future"]).correlation), 3)])

    # ---- final estimators: fit on all of D_val, calibrate on D_cal, tau on D_cal
    final = {}
    for name, cols in SETS.items():
        mdl = make(chosen[name]).fit(X(V, cols), V["z"])
        cal = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(mdl.predict_proba(X(C, cols))[:, 1], C["z"])
        rc = cal.predict(mdl.predict_proba(X(C, cols))[:, 1])
        tau = choose_tau(rc, C["z"], FTR_BUDGET)
        final[name] = (mdl, cal, tau, cols)
        re = cal.predict(mdl.predict_proba(X(E, cols))[:, 1])
        mtr = metrics(re, E["z"], E["L_future"], tau)
        rows.append(["E5 (held out)", name, chosen[name]] + [mtr[k] for k in ("AUROC", "AUPRC", "Brier", "log_loss", "ECE", "spearman_r_vs_loss")]
                    + [round(tau, 3), mtr["trusted_share"], mtr["FTR"]])
        mo = metrics(oof_store[name], V["z"], V["L_future"], tau)
        rows[list(SETS).index(name)] += [round(tau, 3), mo["trusted_share"], mo["FTR"]]
    hdr = ["data", "estimator", "model", "AUROC", "AUPRC", "Brier", "log_loss", "ECE", "spearman_r_vs_loss",
           "tau (from cal)", "trusted_share", "FTR"]
    with open(os.path.join(a.out, "estimators.csv"), "w", newline="") as f:
        cw = csv.writer(f); cw.writerow(hdr); cw.writerows(rows)
    print(f"\n{'data':<23}{'estimator':<38}{'model':<15}{'AUROC':>7}{'AUPRC':>7}{'Brier':>8}{'ECE':>7}{'rho':>7}{'tau':>7}{'trust':>7}{'FTR':>6}")
    for r in rows:
        g = lambda i: (f"{r[i]:.3f}" if isinstance(r[i], float) else str(r[i])) if i < len(r) else ""
        print(f"{r[0]:<23}{r[1]:<38}{r[2]:<15}{g(3):>7}{g(4):>7}{g(5):>8}{g(7):>7}{g(8):>7}{g(9):>7}{g(10):>7}{g(11):>6}")

    # ---- step 39/40: combined estimator per condition, and association with the realised loss
    comb = "combined (o+u+e)"
    mdl, cal, tau, cols = final[comb]
    rv, re = oof_store[comb], cal.predict(mdl.predict_proba(X(E, cols))[:, 1])
    E["perturb"], E["lighting"] = e5_labels(E)
    crow = []
    groups = [("val", c, V["cond"] == c) for c in sorted(set(V["cond"]))] + \
             [("val", "moving", V["phase"] == "moving"), ("val", "hold", V["phase"] == "hold")] + \
             [("E5", f"{p}/{l}", (E["perturb"] == p) & (E["lighting"] == l)) for l in ("normal", "dim") for p in ("none", "cover", "head")]
    print(f"\nCombined estimator per condition (val: cross-validated; E5: held out). tau = {tau:.3f}")
    print(f"{'':<18}{'n':>6}{'unrel.':>8}{'AUROC':>7}{'mean r':>8}{'trusted':>9}{'FTR':>6}")
    for src, nm, m in groups:
        T, r = (V, rv) if src == "val" else (E, re)
        mt = metrics(r[m], T["z"][m], T["L_future"][m], tau)
        crow.append([src, nm, mt["n"], mt["unreliable_share"], mt["AUROC"], round(float(r[m].mean()), 3), mt["trusted_share"], mt["FTR"]])
        print(f"{src + ' ' + nm:<18}{mt['n']:>6}{mt['unreliable_share']:>8.2f}{(mt['AUROC'] or float('nan')):>7.3f}{r[m].mean():>8.2f}"
              f"{mt['trusted_share']:>9.2f}{(mt['FTR'] if mt['FTR'] is not None else float('nan')):>6.2f}")
    with open(os.path.join(a.out, "by_condition.csv"), "w", newline="") as f:
        cw = csv.writer(f); cw.writerow(["data", "condition", "n", "unreliable_share", "AUROC", "mean_r", "trusted_share", "FTR"]); cw.writerows(crow)
    gv = np.array([f"{s}_{r}" for s, r in zip(V["session"], V["reach"])])
    ge = np.array([f"{s}_{r}" for s, r in zip(E["session"], E["reach"])])
    assoc = {"val": {"spearman": round(float(spearmanr(rv, V["L_future"]).correlation), 3), "ci95_reach_bootstrap": boot_spearman(rv, V["L_future"], gv)},
             "E5": {"spearman": round(float(spearmanr(re, E["L_future"]).correlation), 3), "ci95_reach_bootstrap": boot_spearman(re, E["L_future"], ge)}}
    print(f"\nr_t vs realised 1.0 s loss (Spearman, 95% reach-bootstrap CI): val {assoc['val']}, E5 {assoc['E5']}")

    # ---- abrupt changes: E4 switches in D_val, detection delay; false alarms per minute on E1 val
    sw = switch_detection(dev, final[comb], tau)
    with open(os.path.join(a.out, "switch_detection.csv"), "w", newline="") as f:
        cw = csv.writer(f); cw.writerow(["session", "reach", "t_switch_cue", "t_physical_onset", "t_detect", "delay_s"]); cw.writerows(sw["rows"])
    print(f"\nE4 switches in D_val: {sw['n']}; r_t fell below tau within 1.5 s of the physical onset in {sw['detected']} "
          f"(delay median {sw['median']} s, p90 {sw['p90']} s); on E1 val: {sw['alarms_per_min_E1']} alarms per minute, "
          f"{sw['unneeded_alarms_per_min_E1']} of them unneeded")

    # ---- r_t for EVERY row (labelled or not), for the offline planner replay:
    #      val rows use the model of their own cross-validation fold (never fitted on them); cal and E5 use the
    #      final estimator. Saved for the combined and the uncertainty-only estimator.
    C_all = sub(dev, dev["set"] == "cal")
    scores = {}
    for name in ("combined (o+u+e)", "uncertainty-only (u)"):
        cols = SETS[name]
        rv_all = np.zeros(len(fall))
        for k, (mdl_k, cal_k) in enumerate(fold_store[name]):
            m = fall == k
            if m.any():
                rv_all[m] = cal_k.predict(mdl_k.predict_proba(X(Vall, cols)[m])[:, 1])
        mdl, cal, _, _ = final[name]
        key = "combined" if name.startswith("combined") else "uncertainty"
        scores[key] = {"val": rv_all, "cal": cal.predict(mdl.predict_proba(X(C_all, cols))[:, 1]),
                       "e5": cal.predict(mdl.predict_proba(X(e5, cols))[:, 1])}
    for setname, T in (("val", Vall), ("cal", C_all), ("e5", e5)):
        np.savez_compressed(os.path.join(a.out, f"scores_{setname}.npz"), session=T["session"], anchor=T["anchor"],
                            r_combined=scores["combined"][setname], r_uncertainty=scores["uncertainty"][setname])

    pickle.dump({"estimators": {k: (v[0], v[1], v[2], v[3]) for k, v in final.items()}, "l_max_px": L_MAX,
                 "ftr_budget": FTR_BUDGET}, open(os.path.join(a.out, "reliability_model.pkl"), "wb"))
    json.dump({"l_max_px": L_MAX, "l_max_note": "about 4 cm at the table (1.27 mm/px); fixed on D_val before fitting",
               "label": "mean ADE of the most likely mode over elbow, wrist and hand for the next 1.0 s <= l_max",
               "ftr_budget_on_cal": FTR_BUDGET, "chosen_models": chosen, "tau": {k: v[2] for k, v in final.items()},
               "association": assoc, "switch_detection": {k: v for k, v in sw.items() if k != "rows"},
               "counts": {"val": len(V["z"]), "cal": len(C["z"]), "e5": len(E["z"])},
               "unreliable_share": {"val": round(float(1 - V["z"].mean()), 3), "cal": round(float(1 - C["z"].mean()), 3),
                                    "e5": round(float(1 - E["z"].mean()), 3)}},
              open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    plots(a.out, V, rv, E, re, oof_store)
    print(f"\nSaved in {a.out}/")


def e5_labels(T):
    lab = {}
    for s in np.unique(T["session"]):
        lab[s] = list(csv.DictReader(open(os.path.join("logs", s, "reaches.csv"))))
    p, l = [], []
    for s, r in zip(T["session"], T["reach"]):
        if r == 0:
            p.append("setup"); l.append("normal" if s.endswith("_s01") else "dim")
        else:
            row = lab[s][int(r) - 1]; p.append(row["perturb"]); l.append(row["lighting"])
    return np.array(p), np.array(l)


def switch_detection(dev, est, tau, win=1.5):
    """Delay from the physical change onset of each E4 switch (in D_val) to r_t < tau."""
    from s4_dataset_summary import effector_track, load as load_session, switch_onset
    mdl, cal, _, cols = est
    V = sub(dev, dev["set"] == "val")
    r_all = cal.predict(mdl.predict_proba(X(V, cols))[:, 1])
    import numpy as _np
    rows, delays = [], []
    pm = {}
    for s in _np.unique(V["session"]):
        pm[s] = _np.load(os.path.join("logs", "prepared", s + ".npz"))["t"]
    for s in [x for x in _np.unique(V["session"]) if x.startswith("E4")]:
        meta, frames, _, reaches = load_session(os.path.join("logs", s))
        t, xy = effector_track(frames, meta["side"])
        T = {int(k): v for k, v in meta["targets"].items()}
        m = V["session"] == s
        ta = pm[s][V["anchor"][m]]
        rr = r_all[m]
        val_reaches = set(V["reach"][m].tolist())
        for r in reaches:
            if r["valid"] != "1" or not r["t_switch_cue"] or int(r["reach"]) not in val_reaches:
                continue
            tc = float(r["t_switch_cue"])
            on = switch_onset(t, xy, tc, T[int(r["goal_cued"])]["px"], 45.0)
            if on is None:
                continue
            after = (ta >= on) & (ta <= on + win)
            hit = _np.where(after & (rr < tau))[0]
            td = float(ta[hit[0]]) if len(hit) else None
            rows.append([s, r["reach"], round(tc, 3), round(on, 3), round(td, 3) if td else "", round(td - on, 3) if td else ""])
            if td is not None:
                delays.append(td - on)
    # alarms on E1 val (no switches): an alarm episode is a run of consecutive steps with r_t < tau.
    # It is "unneeded" if the forecast stayed good (L_future <= l_max) at every labelled step of the run.
    fa, fa_unneeded, minutes = 0, 0, 0.0
    for s in [x for x in _np.unique(V["session"]) if x.startswith("E1")]:
        m = V["session"] == s
        order = _np.argsort(V["anchor"][m])
        below = r_all[m][order] < tau
        Lf = V["L_future"][m][order]
        i = 0
        while i < len(below):
            if not below[i]:
                i += 1; continue
            j = i
            while j < len(below) and below[j]:
                j += 1
            fa += 1
            seg = Lf[i:j]; seg = seg[~_np.isnan(seg)]
            if len(seg) and (seg <= L_MAX).all():
                fa_unneeded += 1
            i = j
        minutes += m.sum() * 0.0625 / 60
    q = lambda p: round(float(_np.percentile(delays, p)), 3) if delays else None
    return {"rows": rows, "n": len(rows), "detected": len(delays), "median": q(50), "p90": q(90),
            "alarms_per_min_E1": round(fa / minutes, 2) if minutes else None,
            "unneeded_alarms_per_min_E1": round(fa_unneeded / minutes, 2) if minutes else None,
            "note": "alarm = run of steps with r_t < tau; unneeded = the forecast stayed within l_max throughout"}


def plots(out, V, rv, E, re, oof):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    edges = np.linspace(0, 1, 11)
    for nm, (r, z) in {"val, combined (cross-validated)": (rv, V["z"]), "E5, combined (held out)": (re, E["z"])}.items():
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (r >= lo) & (r <= hi if hi == 1 else r < hi)
            if m.sum() >= 20:
                xs.append(r[m].mean()); ys.append(z[m].mean())
        ax[0].plot(xs, ys, "o-", label=nm)
    ax[0].plot([0, 1], [0, 1], "k:", lw=1)
    ax[0].set(xlabel="predicted reliability r_t", ylabel="observed share with loss <= l_max", title="Reliability diagram")
    ax[0].legend(fontsize=8)
    for nm, (r, L) in {"val": (rv, V["L_future"]), "E5": (re, E["L_future"])}.items():
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (r >= lo) & (r <= hi if hi == 1 else r < hi)
            if m.sum() >= 20:
                xs.append(r[m].mean()); ys.append(np.median(L[m]))
        ax[1].plot(xs, ys, "o-", label=nm)
    ax[1].axhline(L_MAX, color="gray", ls="--", lw=1, label="l_max")
    ax[1].set(xlabel="predicted reliability r_t", ylabel="median realised loss over next 1.0 s (px)", yscale="log",
              title="Reliability vs realised loss")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "reliability_diagram.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
