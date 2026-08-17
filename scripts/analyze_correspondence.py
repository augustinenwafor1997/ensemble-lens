"""Test how well CoVAMPnet's state indices actually correspond across systems.

A reviewer's domain expert objected that our per-state structure preview implies the
three systems share identical conformations, "but that is not what happens. Some states
across different systems could be classified as the same, while others could not."

That is testable. The trap is choosing the descriptor. Beta-strand propensity is the
obvious handle, but it is exactly the quantity the drug is reported to change, so a drug
that works would register as a matching failure. Using it would manufacture the very
result we are testing for.

The descriptor here is therefore the mean Ca-Ca distance matrix, which describes chain
topology, is invariant to rotation and translation, needs no superposition, and is not
the channel the aggregation claim runs through. Rg and beta are reported alongside it as
effects rather than as evidence of correspondence.

The control has to be the right one, and the obvious choice is wrong. Comparing a
cross-system distance between two MEAN structures against the frame-to-frame spread of
the ensemble compares incommensurable quantities: the mean of 200 frames is far better
determined than any single frame, so that test buries every real difference under the
ensemble's breadth. The correct floor is a split-half bootstrap, which asks how far apart
two mean structures land when drawn from the SAME ensemble and differing only by
sampling. Ensemble spread is still reported, as the yardstick for whether a real
difference is also a big one.

Run:  py -3.10 analyze_correspondence.py --data track1_struct2_dense.json
      (the dense file carries 200 frames per state; the shipped one carries 6, which is
      too few to estimate a mean structure)
Writes: data/track1_correspondence.json
"""
import json, pathlib, itertools, argparse
import numpy as np

BASE = pathlib.Path(__file__).resolve().parent
_ap = argparse.ArgumentParser()
_ap.add_argument("--data", default="track1_struct2.json",
                 help="track1_struct2_dense.json holds 200 frames/state instead of 6")
_ap.add_argument("--out", default="track1_correspondence.json")
_args = _ap.parse_args()
D = json.load(open(BASE / "data" / _args.data))
K = "3"
SYS = D["systems"]
NS = len(SYS)
SHORT = ["free", "TMP", "SPA"]
rng = np.random.default_rng(0)


IU = np.triu_indices(42, k=1)


def dvec(frame):
    """Upper triangle of the Ca-Ca distance matrix, flattened.

    Superposition-free description of chain topology, so no alignment choice can
    influence the comparison.
    """
    P = np.asarray(frame, float)
    return np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)[IU]


def dmat_dist(a, b):
    """Mean absolute difference between two distance vectors, in Angstrom."""
    return float(np.abs(a - b).mean())


def pairwise_mean(V):
    """Mean pairwise distance within a set of frames, row-blocked to bound memory.

    This is the spread of the ENSEMBLE, not the uncertainty of its mean. It says how
    broad the state is; it is the wrong yardstick for judging a difference between two
    mean structures. Reported for context only.
    """
    tot, cnt = 0.0, 0
    for i in range(len(V) - 1):
        d = np.abs(V[i] - V[i + 1:]).mean(1)
        tot += d.sum(); cnt += len(d)
    return tot / cnt if cnt else float("nan")


def splithalf(V, reps=200):
    """Sampling noise of the MEAN structure, by repeated random half/half splits.

    This is the correct floor for comparing mean structures across systems: it is how
    far apart two mean structures land when they are drawn from the SAME ensemble and
    differ only by sampling. A cross-system distance above this is a real difference.
    """
    n = len(V)
    out = []
    for _ in range(reps):
        idx = rng.permutation(n)
        a, b = V[idx[: n // 2]].mean(0), V[idx[n // 2:]].mean(0)
        out.append(np.abs(a - b).mean())
    return float(np.mean(out))


st = {}
for si, s in enumerate(SYS):
    for k, v in s["params"][K]["states"].items():
        V = np.array([dvec(f) for f in v["ensemble"]])
        st[(si, int(k))] = {
            "V": V, "mean": V.mean(0),
            "rg": v["rgMean"], "beta": np.array(v["beta"], float),
            "betaMean": v["betaMean"], "n": v["n"],
        }
NK = len({k for _, k in st})
print(f"{len(next(iter(st.values()))['V'])} frames per state\n")

# ---- control: within-system, within-state scatter -----------------------
print("=== CONTROL: spread between frames of the SAME system and state ===")
within, halves = [], []
print(f"  {'':12s}{'spread':>9s}{'mean-noise':>12s}")
for (si, k), v in sorted(st.items()):
    m, h = pairwise_mean(v["V"]), splithalf(v["V"])
    within.append(m); halves.append(h)
    print(f"  {SHORT[si]:5s} S{k}: {m:8.2f} A {h:10.2f} A   (n={v['n']}, {len(v['V'])} frames)")
W = float(np.mean(within))
H = float(np.mean(halves))
print(f"\n  ensemble spread            = {W:.2f} A   (how broad a state is)")
print(f"  mean-structure noise floor = {H:.2f} A   <- the yardstick for the test below")

# ---- cross-system comparison, matched vs mismatched states --------------
print("\n=== cross-system distance, matched vs mismatched states ===")
report = {"K": int(K), "short": SHORT, "spread": round(W, 3),
          "meanNoiseFloor": round(H, 3), "states": {}}
for k in range(NK):
    same = [dmat_dist(st[(i, k)]["mean"], st[(j, k)]["mean"])
            for i, j in itertools.combinations(range(NS), 2)]
    diff = [dmat_dist(st[(i, k)]["mean"], st[(j, m)]["mean"])
            for i in range(NS) for j in range(NS) for m in range(NK)
            if i != j and m != k]
    dm, do = float(np.mean(same)), float(np.mean(diff))
    ratio = do / dm if dm else float("inf")
    rgs = [st[(i, k)]["rg"] for i in range(NS)]
    bts = [st[(i, k)]["betaMean"] * 100 for i in range(NS)]

    # Two separate questions, and they have different answers:
    #  (a) is the cross-system difference real, or just sampling noise?
    #  (b) is it small compared with the state's own conformational breadth?
    real = dm > 2 * H                     # detectably above the same-ensemble floor
    tight = dm < 0.5 * W                  # yet well inside the state's own spread
    verdict = ("real but small against the ensemble spread" if real and tight else
               "real and large" if real else
               "indistinguishable from sampling noise")
    print(f"\nState {k}: {verdict}   (separation from other states {ratio:.2f}x)")
    print(f"  same state across systems : {dm:5.2f} A  = {dm/H:4.1f}x the {H:.2f} A noise floor")
    print(f"  other states across systems: {do:5.2f} A")
    print(f"  ensemble spread            : {W:5.2f} A  (cross-system gap is {dm/W*100:.0f}% of it)")
    print(f"  Rg   : " + ", ".join(f"{SHORT[i]} {rgs[i]:5.1f}" for i in range(NS))
          + f"   spread {max(rgs)-min(rgs):.1f} A")
    print(f"  beta : " + ", ".join(f"{SHORT[i]} {bts[i]:4.1f}%" for i in range(NS))
          + f"   spread {max(bts)-min(bts):.1f} pp   <- drug EFFECT, not mismatch")
    report["states"][str(k)] = {
        "sameState": round(dm, 3), "diffState": round(do, 3),
        "ratio": round(ratio, 3), "real": bool(real), "tight": bool(tight), "verdict": verdict,
        "rg": [round(r, 2) for r in rgs], "rgSpread": round(max(rgs) - min(rgs), 2),
        "beta": [round(b, 2) for b in bts], "betaSpread": round(max(bts) - min(bts), 2),
    }

json.dump(report, open(BASE / "data" / _args.out, "w"), indent=1)
print("\nwrote data/track1_correspondence.json")
