"""Rebuild the shipped track1_struct2.json from the dense run, with correct sampling.

The original file was produced when the CA reservoir kept the FIRST 150 frames of each
state rather than a random sample. Adaptive sampling makes trajectory order meaningful,
so that biased every per-state ensemble and RMSF toward the earliest epochs. The dense
run uses proper reservoir sampling, so the fix is to derive the shipped file from it
rather than spend another 45 minutes re-reading 12 GB of trajectories.

Only the two reservoir-derived fields change: `rmsf` (now over 1200 sampled frames
instead of the first 150) and `ensemble` (subsampled back to the display count). Every
other quantity is accumulated over all frames and is unaffected, which this script
checks rather than assumes.

Run:  py -3.10 derive_struct2.py [--nens 6]
"""
import argparse, json, os, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
DENSE = os.path.join(HERE, "data", "track1_struct2_dense.json")
OUT = os.path.join(HERE, "data", "track1_struct2.json")
BACKUP = os.path.join(HERE, "data", "track1_struct2_prereservoirfix.json")

ap = argparse.ArgumentParser()
ap.add_argument("--nens", type=int, default=6, help="frames kept per state for display")
args = ap.parse_args()

old = json.load(open(OUT))
dense = json.load(open(DENSE))

if not os.path.exists(BACKUP):
    shutil.copy(OUT, BACKUP)
    print(f"backed up biased file -> {os.path.basename(BACKUP)}")

# Fields that never passed through the reservoir must be identical; if they are not,
# the two runs disagree about something they should agree on and the swap is unsafe.
UNAFFECTED = ("n", "beta", "helix", "drug", "drugNull",
              "rgMean", "rgStd", "betaMean", "betaStd")
mismatch, changed_rmsf = [], []
for so, sd in zip(old["systems"], dense["systems"]):
    for K in so["params"]:
        for k, vo in so["params"][K]["states"].items():
            vd = sd["params"][K]["states"][k]
            for f in UNAFFECTED:
                if vo.get(f) != vd.get(f):
                    mismatch.append(f"{so['tag']} K={K} S{k} {f}")
            if vo.get("rmsf") != vd.get("rmsf"):
                changed_rmsf.append(f"{so['tag']} K={K} S{k}")

print(f"unaffected fields differing : {len(mismatch)}  (expect 0)")
if mismatch:
    for m in mismatch[:6]:
        print("   ", m)
    raise SystemExit("aborting: runs disagree on quantities the reservoir cannot touch")
print(f"RMSF profiles changed       : {len(changed_rmsf)}  (expect many: the bug's effect)")

# take the dense file wholesale, then thin the ensembles back to display size
for sd in dense["systems"]:
    for K in sd["params"]:
        for k, vd in sd["params"][K]["states"].items():
            ens = vd.get("ensemble") or []
            if len(ens) > args.nens:
                step = (len(ens) - 1) / (args.nens - 1) if args.nens > 1 else 1
                vd["ensemble"] = [ens[int(round(i * step))] for i in range(args.nens)]
dense["meta"]["reservoirSampling"] = "algorithm-R"

json.dump(dense, open(OUT, "w"), separators=(",", ":"))
print(f"wrote {OUT} ({os.path.getsize(OUT)/1024:.0f} KB, {args.nens} frames/state)")
