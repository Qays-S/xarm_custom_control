#!/usr/bin/env python3
"""
H3: freeze everything before Phase C. Saves the SHA-256 of every model, setting and script that the
live planners use, so the paper can show nothing changed after the comparison started.

  python3 ../s9_freeze.py            (run in the h3 folder, after Phase B)
Writes logs/live_freeze.json; refuses to overwrite an existing one.
"""
import glob
import hashlib
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ["logs/prepared/split_manifest.json", "logs/forecast_results/baselines_val.json",
         "logs/forecast_results/gru/gru_best.pt", "logs/forecast_results/gru/gru_best.json",
         "logs/forecast_results/mgru/mgru_best.pt", "logs/forecast_results/mgru/mgru_best.json",
         "logs/forecast_results/stsgcn/stsgcn_finetuned_best.pt", "logs/reliability/reliability_model.pkl",
         "logs/reliability/summary.json", "logs/replay/summary.json", "logs/selftest/selftest.json"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    out = "logs/live_freeze.json"
    if os.path.exists(out):
        raise SystemExit(f"{out} already exists; Phase C has been frozen. Delete it only if you are starting over.")
    missing = [f for f in FILES if not os.path.exists(f)]
    if missing:
        raise SystemExit(f"Missing (finish Phase B first): {missing}")
    st = json.load(open("logs/selftest/selftest.json"))
    if not st.get("all_pass"):
        raise SystemExit("The last s9_selftest.py run did not pass; fix that before freezing.")
    scripts = sorted(glob.glob(os.path.join(HERE, "s*.py")))
    rec = {"frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "note": "Fixed before the first Phase C (planner comparison) session.",
           "sha256": {**{f: sha(f) for f in FILES}, **{os.path.basename(s): sha(s) for s in scripts}}}
    json.dump(rec, open(out, "w"), indent=1)
    print(f"Froze {len(rec['sha256'])} files -> {out}")


if __name__ == "__main__":
    main()
