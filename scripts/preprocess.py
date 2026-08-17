"""
Bio+MedVis Challenge 2026 - Track 1 (Adaptive Molecular Dynamics)
Preprocess the CoVAMPnet Markov-state assignment files into one compact JSON
for the D3/HTML comparative visualization.

Inputs (per system zsab2/zsab3/zsab4):
  zsabX_names.txt         - trajectory names (alphabetical)
  zsabX_order.txt         - "globalStartTime: name" (temporal alignment)
  zsabX_concatenated.txt  - concatenated trajectory blocks (not required here)
  zsabX_statesY.txt       - "name: s0,s1,..." per-frame states for Y=2,3,4 state MSM

Outputs:
  data/track1.json
"""
import os, re, json, collections, sys
import numpy as np

BV = r"C:/Users/AUGUST~1/bv/t1"
OUT = os.path.join(os.path.dirname(__file__), "data", "track1.json")

N_BOOT = 1000                       # bootstrap resamples for confidence intervals
ITS_LAGS = [1, 5, 10, 20, 40, 80, 160]   # lags (frames) for the implied-timescale plot


def trans_counts(seqs, K, lag):
    """K x K count matrix of state i -> state j pairs at the given lag."""
    C = np.zeros(K * K)
    for s in seqs:
        if len(s) > lag:
            a = s[:-lag].astype(int); b = s[lag:].astype(int)
            C += np.bincount(a * K + b, minlength=K * K)
    return C.reshape(K, K)


def row_norm(C):
    rs = C.sum(1, keepdims=True)
    return np.divide(C, rs, out=np.zeros_like(C, dtype=float), where=rs > 0)


def mfpt_frames(P, tau):
    """Mean first passage time (in frames) from state i to state j, from lag-tau matrix P."""
    K = P.shape[0]; M = [[None] * K for _ in range(K)]
    for j in range(K):
        idx = [i for i in range(K) if i != j]
        A = np.eye(len(idx)) - P[np.ix_(idx, idx)]
        try:
            m = np.linalg.solve(A, np.ones(len(idx)))
        except np.linalg.LinAlgError:
            continue
        for r, i in enumerate(idx):
            M[i][j] = round(float(m[r] * tau), 1) if m[r] > 0 else None
        M[j][j] = 0.0
    return M


def implied_timescales(seqs, K, lags):
    """For each lag, the K-1 slowest implied timescales t = -lag/ln|lambda|."""
    out = []
    for lag in lags:
        P = row_norm(trans_counts(seqs, K, lag))
        mag = np.sort(np.abs(np.linalg.eigvals(P)))[::-1]
        ts = []
        for lam in mag[1:]:
            ts.append(round(float(-lag / np.log(lam)), 1) if 0 < lam < 1 else None)
        out.append(ts)
    return out


def bootstrap_ci(occ_arr, tc_arr, K, B=N_BOOT, seed=0):
    """Percentile CIs for occupancy and transition probabilities, resampling trajectories."""
    rng = np.random.default_rng(seed); n = len(occ_arr)
    occs = np.empty((B, K)); trs = np.empty((B, K, K))
    for b in range(B):
        sel = rng.integers(0, n, n)
        o = occ_arr[sel].sum(0); occs[b] = o / o.sum()
        trs[b] = row_norm(tc_arr[sel].sum(0))
    olo, ohi = np.percentile(occs, [2.5, 97.5], axis=0)
    tlo, thi = np.percentile(trs, [2.5, 97.5], axis=0)
    return ([[round(float(olo[k]), 4), round(float(ohi[k]), 4)] for k in range(K)],
            [[[round(float(tlo[i, j]), 4), round(float(thi[i, j]), 4)] for j in range(K)] for i in range(K)])

SYSTEMS = [
    ("zsab2", "Abeta42 (free)",              "Free disordered Abeta42 peptide"),
    ("zsab3", "Abeta42 + Tramiprosate (TMP)", "Abeta42 with phase-3 drug tramiprosate"),
    ("zsab4", "Abeta42 + SPA",               "Abeta42 with TMP metabolite 3-sulfopropanoic acid"),
]
NAME_RE = re.compile(r"^e(\d+)s(\d+)_(?:e(\d+)s(\d+)p0f(\d+)|0)$")

N_TIME_BINS = 200   # temporal streamgraph resolution
N_STRIP_BINS = 160  # per-trajectory strip resolution (majority state per bin)
TAU = 40            # transition-matrix lag (frames); states are highly metastable at lag 1


def read_lines(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


def parse_states_file(path):
    """Return dict name -> list[int] of per-frame states."""
    out = {}
    for ln in read_lines(path):
        if ":" not in ln:
            continue
        name, seq = ln.split(":", 1)
        vals = [int(v) for v in seq.strip().replace(" ", "").split(",") if v != ""]
        out[name.strip()] = vals
    return out


def rle_downsample(seq, nbins):
    """Majority state per bin -> list of [state, runlength] (RLE of the binned track)."""
    n = len(seq)
    if n == 0:
        return []
    binned = []
    for b in range(nbins):
        lo = b * n // nbins
        hi = max(lo + 1, (b + 1) * n // nbins)
        seg = seq[lo:hi]
        binned.append(collections.Counter(seg).most_common(1)[0][0])
    # RLE compress the binned track
    runs = []
    cur, cnt = binned[0], 1
    for s in binned[1:]:
        if s == cur:
            cnt += 1
        else:
            runs.append([cur, cnt]); cur, cnt = s, 1
    runs.append([cur, cnt])
    return runs


def build_system(sysid):
    base = os.path.join(BV, sysid)
    order = {}
    for ln in read_lines(os.path.join(base, f"{sysid}_order.txt")):
        if ":" in ln:
            t, nm = ln.split(":", 1)
            try:
                order[nm.strip()] = int(t.strip())
            except ValueError:
                pass

    params = {}
    for Y in (2, 3, 4):
        states = parse_states_file(os.path.join(base, f"{sysid}_states{Y}.txt"))
        names = list(states.keys())

        # --- occupancy, transitions, uncertainty, kinetics ---
        # States are highly metastable (diagonal ~100% at lag 1); we count transitions
        # at lag TAU, WITHIN each simulation, to expose the rare inter-state moves (T2).
        seqs = [np.asarray(s, dtype=np.int8) for s in states.values()]
        occ_arr = np.stack([np.bincount(s, minlength=Y) for s in seqs]).astype(float)  # per-traj
        tc_arr = np.stack([trans_counts([s], Y, TAU) for s in seqs])                    # per-traj
        total_frames = int(occ_arr.sum())
        occ_counts = occ_arr.sum(0)
        occ_frac = list(occ_counts / occ_counts.sum())
        trans = tc_arr.sum(0)
        prob = row_norm(trans)
        occCI, transCI = bootstrap_ci(occ_arr, tc_arr, Y)
        mfpt = mfpt_frames(prob, TAU)
        its = implied_timescales(seqs, Y, ITS_LAGS)

        # --- temporal state proportions (aligned by global start time) ---
        max_t = max((order.get(n, 0) + len(states[n]) for n in names), default=0)
        bins_state = [[0] * Y for _ in range(N_TIME_BINS)]
        bins_active = [0] * N_TIME_BINS
        if max_t > 0:
            for n in names:
                start = order.get(n, 0)
                seq = states[n]
                for f, s in enumerate(seq):
                    g = start + f
                    bi = min(N_TIME_BINS - 1, g * N_TIME_BINS // max_t)
                    bins_state[bi][s] += 1
                    bins_active[bi] += 1
        temporal = []
        for bi in range(N_TIME_BINS):
            tot = sum(bins_state[bi])
            frac = [(bins_state[bi][s] / tot if tot else 0.0) for s in range(Y)]
            temporal.append({"active": bins_active[bi], "frac": [round(x, 4) for x in frac]})

        # --- per-trajectory strips (downsampled), ordered by global start time ---
        ordered_names = sorted(names, key=lambda n: order.get(n, 0))
        strips = []
        for n in ordered_names:
            strips.append({
                "n": n,
                "t0": order.get(n, 0),
                "len": len(states[n]),
                "dom": collections.Counter(states[n]).most_common(1)[0][0],
                "rle": rle_downsample(states[n], N_STRIP_BINS),
            })

        params[str(Y)] = {
            "occupancy": [round(float(x), 4) for x in occ_frac],
            "occCI": occCI,
            "transProb": [[round(float(x), 4) for x in row] for row in prob],
            "transCI": transCI,
            "transCount": trans.astype(int).tolist(),
            "mfpt": mfpt,
            "its": {"lags": ITS_LAGS, "timescales": its},
            "totalFrames": total_frames,
            "maxTime": max_t,
            "temporal": temporal,
            "strips": strips,
        }

    return params


def build_genealogy(sysid):
    """Adaptive-sampling forest from trajectory names. Node id = e{epoch}s{sim}."""
    base = os.path.join(BV, sysid)
    names = read_lines(os.path.join(base, f"{sysid}_names.txt"))
    # dominant state from the 3-state parametrisation for colour
    states3 = parse_states_file(os.path.join(base, f"{sysid}_states3.txt"))
    order = {}
    for ln in read_lines(os.path.join(base, f"{sysid}_order.txt")):
        if ":" in ln:
            t, nm = ln.split(":", 1)
            try:
                order[nm.strip()] = int(t.strip())
            except ValueError:
                pass
    nodes, edges = {}, []
    discovery = {}   # state -> {node, gtime}: earliest global appearance of each 3-state
    for nm in names:
        m = NAME_RE.match(nm)
        if not m:
            continue
        ep, sm, pe, ps, pf = m.groups()
        nid = f"e{ep}s{sm}"
        seq = states3.get(nm, [])
        dom = collections.Counter(seq).most_common(1)[0][0] if seq else -1
        nodes[nid] = {"id": nid, "epoch": int(ep), "sim": int(sm),
                      "len": len(seq), "dom": dom,
                      "parent": (f"e{pe}s{ps}" if pe else None),
                      "branchFrame": (int(pf) if pf else 0)}
        t0 = order.get(nm, 0)
        for s in set(seq):
            gt = t0 + int(np.argmax(np.asarray(seq) == s))   # first frame in this traj with state s
            if s not in discovery or gt < discovery[s]["gtime"]:
                discovery[s] = {"node": nid, "gtime": gt}
    for nid, nd in nodes.items():
        if nd["parent"] and nd["parent"] in nodes:
            edges.append({"source": nd["parent"], "target": nid, "frame": nd["branchFrame"]})
    return {"nodes": list(nodes.values()), "edges": edges,
            "discovery": {str(k): v for k, v in discovery.items()}}


def main():
    data = {"systems": [], "meta": {
        "source": "CoVAMPnet (Marques et al., JACS Au 2024); Bio+MedVis Challenge 2026 Track 1",
        "nTimeBins": N_TIME_BINS, "nStripBins": N_STRIP_BINS, "tau": TAU}}
    for sysid, label, desc in SYSTEMS:
        sys.stderr.write(f"processing {sysid} ...\n")
        params = build_system(sysid)
        gen = build_genealogy(sysid)
        n_traj = len(params["3"]["strips"])
        data["systems"].append({
            "id": sysid, "label": label, "desc": desc,
            "nTraj": n_traj, "params": params, "genealogy": gen})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    sz = os.path.getsize(OUT) / 1024
    sys.stderr.write(f"wrote {OUT}  ({sz:.0f} KB)\n")
    # quick summary
    for s in data["systems"]:
        p3 = s["params"]["3"]
        sys.stderr.write(f"  {s['id']}: {s['nTraj']} trajs, occ3={p3['occupancy']}, "
                         f"{len(s['genealogy']['nodes'])} genealogy nodes, "
                         f"{len(s['genealogy']['edges'])} edges\n")


if __name__ == "__main__":
    main()
