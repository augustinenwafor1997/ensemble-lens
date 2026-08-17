"""
Bio+MedVis Challenge 2026 - Track 1 - Task T3, second-pass structural quantities
(added after the expert review). Complements analyze_structure.py / track1_struct.json.

Per system and per Markov state (2/3/4-state models), from the atomistic trajectories:
  - SECONDARY STRUCTURE propensity per residue: fraction of frames each residue is
    beta-strand (DSSP 'E') and helix (DSSP 'H')  -> the aggregation-relevant observable.
  - CONFORMATIONAL LANDSCAPE features per frame: radius of gyration Rg and beta-content
    (fraction of residues in 'E'); accumulated as a per-SYSTEM 2D histogram (free energy
    = -ln count) and per-STATE centroid + spread.
  - DRUG NULL baseline: peptide-vs-drug contact using drug coordinates from a
    time-decorrelated frame -> "contact expected at this ligand concentration by chance",
    so panel 7 can show specific binding above background.
  - ENSEMBLE representation: ~6 Kabsch-superposed CA frames per state + per-residue RMSF,
    so an intrinsically disordered state is shown as a spread, not one fold.

Run:  py -3.10 analyze_structure2.py --root <dir with ZS-ab*/> [--stride N]
Output: data/track1_struct2.json
"""
import os, glob, json, argparse, warnings
import numpy as np
warnings.filterwarnings("ignore")
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array
from MDAnalysis.analysis.dssp import DSSP

HERE = os.path.dirname(__file__)
STATES_ROOT = r"C:/Users/AUGUST~1/bv/t1"
RESN = 42
DRUG_CUTOFF = 4.5
RG_EDGES = np.linspace(7.0, 25.0, 37)     # 36 bins of 0.5 A
BETA_EDGES = np.linspace(0.0, 0.6, 25)    # 24 bins of beta-content fraction
CA_RESERVOIR = 150                        # frames kept per state for RMSF + ensemble
N_ENSEMBLE = 6                            # superposed frames stored per state
RES_RNG = np.random.default_rng(20260804)  # fixed seed: reservoir sampling must be reproducible

SYS = [("ZS-ab2", "zsab2", "Abeta42 (free)", None),
       ("ZS-ab3", "zsab3", "Abeta42 + Tramiprosate (TMP)", "TMP"),
       ("ZS-ab4", "zsab4", "Abeta42 + SPA", "SPA")]
AA3to1 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G',
          'HIS':'H','HSD':'H','HSE':'H','HSP':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M',
          'PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
# Abeta42 functional regions (1-indexed inclusive) for the frontend to band the sequence axis
REGIONS = [
    {"name": "N-term (charged)", "a": 1,  "b": 16},
    {"name": "KLVFF",           "a": 16, "b": 20},
    {"name": "CHC",             "a": 17, "b": 21},
    {"name": "turn (D23-K28)",  "a": 22, "b": 28},
    {"name": "C-term (hydrophobic)", "a": 30, "b": 42},
]


def load_states(zid):
    out = {}
    for Y in (2, 3, 4):
        p = os.path.join(STATES_ROOT, zid, f"{zid}_states{Y}.txt")
        d = {}
        for ln in open(p, encoding="utf-8"):
            if ":" in ln:
                n, s = ln.split(":", 1)
                d[n.strip()] = np.array([int(x) for x in s.strip().replace(" ", "").split(",") if x != ""], dtype=np.int8)
        out[Y] = d
    return out


def residue_blocks(pep):
    resids = pep.resindices
    starts = [0]
    for i in range(1, len(resids)):
        if resids[i] != resids[i - 1]:
            starts.append(i)
    return np.array(starts)


def kabsch(P, Q):
    """rotation aligning P onto Q (both centered, N x 3)."""
    H = P.T @ Q
    V, S, Wt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Wt.T @ V.T))
    D = np.diag([1.0, 1.0, d])
    return V @ D @ Wt  # apply as P @ R


def process(tag, zid, label, drug, root, stride):
    top = os.path.join(root, tag, "filtered.pdb")
    u = mda.Universe(top)
    pep = u.select_atoms("not resname MOL")
    drug_sel = u.select_atoms("resname MOL")
    starts = residue_blocks(pep)
    pep_idx = pep.indices
    drug_idx = drug_sel.indices if len(drug_sel) else None
    ca_local = np.array([i for i, a in enumerate(pep.atoms) if a.name == "CA"])
    seq = "".join(AA3to1.get(r.resname, "X") for r in pep.residues)

    states = load_states(zid)
    sim_dirs = sorted(glob.glob(os.path.join(root, tag, "e*s*_*")))

    acc = {Y: {} for Y in (2, 3, 4)}
    def slot(Y, s):
        if s not in acc[Y]:
            acc[Y][s] = dict(n=0, beta=np.zeros(RESN), helix=np.zeros(RESN),
                             ndrug=np.zeros(RESN), nnull=0, null=np.zeros(RESN),
                             rg=[], bfrac=[], ca=[], nca=0)
        return acc[Y][s]
    land = {Y: np.zeros((len(RG_EDGES) - 1, len(BETA_EDGES) - 1)) for Y in (2, 3, 4)}

    nused = 0
    for sd in sim_dirs:
        sim = os.path.basename(sd)
        xtc = os.path.join(sd, "output.filtered.xtc")
        if not os.path.exists(xtc) or sim not in states[3]:
            continue
        uu = mda.Universe(top, xtc)
        pep_ag = uu.atoms[pep_idx]
        drug_ag = uu.atoms[drug_idx] if drug_idx is not None else None
        slen = len(states[3][sim])
        # DSSP for the whole sim at matching stride (one extra pass)
        try:
            dssp = DSSP(uu.atoms[pep_idx]).run(step=stride).results.dssp  # (nf, RESN) chars
        except Exception:
            continue
        prev_drug = None
        for k, ts in enumerate(uu.trajectory[::stride]):
            fi = ts.frame
            if fi >= slen or k >= len(dssp):
                break
            ppos = pep_ag.positions
            ss = dssp[k]
            isE = (ss == "E").astype(np.float64)
            isH = (ss == "H").astype(np.float64)
            bfrac = float(isE.mean())
            c = ppos - ppos.mean(0)
            rg = float(np.sqrt((c ** 2).sum(1).mean()))
            # drug + decorrelated null
            dvec = nullvec = None
            if drug_ag is not None:
                dpos = drug_ag.positions
                D = distance_array(ppos, dpos, box=ts.dimensions)
                rowmin = np.minimum.reduceat(D, starts, axis=0)
                dvec = (rowmin.min(1) < DRUG_CUTOFF).astype(np.float64)
                if prev_drug is not None:
                    Dn = distance_array(ppos, prev_drug, box=ts.dimensions)
                    rn = np.minimum.reduceat(Dn, starts, axis=0)
                    nullvec = (rn.min(1) < DRUG_CUTOFF).astype(np.float64)
                prev_drug = dpos.copy()
            capos = ppos[ca_local].copy()
            rb = np.clip(np.digitize(rg, RG_EDGES) - 1, 0, len(RG_EDGES) - 2)
            bb = np.clip(np.digitize(bfrac, BETA_EDGES) - 1, 0, len(BETA_EDGES) - 2)
            for Y in (2, 3, 4):
                s = int(states[Y][sim][fi]); a = slot(Y, s)
                a["n"] += 1; a["beta"] += isE; a["helix"] += isH
                a["rg"].append(rg); a["bfrac"].append(bfrac)
                if dvec is not None: a["ndrug"] += dvec
                if nullvec is not None: a["null"] += nullvec; a["nnull"] += 1
                # Reservoir sampling (Algorithm R). Keeping the FIRST N frames instead
                # would bias the ensemble toward whichever trajectories were read first,
                # and adaptive sampling makes trajectory order meaningful, not arbitrary.
                a["nca"] += 1
                if len(a["ca"]) < CA_RESERVOIR:
                    a["ca"].append(capos)
                else:
                    j = int(RES_RNG.integers(a["nca"]))
                    if j < CA_RESERVOIR:
                        a["ca"][j] = capos
                land[Y][rb, bb] += 1
            nused += 1

    out = {"label": label, "drug": drug, "seq": seq, "params": {}}
    for Y in (2, 3, 4):
        states_out = {}
        for s, a in acc[Y].items():
            if a["n"] == 0:
                continue
            # ensemble: superpose reservoir CA to their mean, RMSF + representative frames
            ca = np.array(a["ca"])                       # (m, RESN, 3)
            ca = ca - ca.mean(1, keepdims=True)          # center each
            ref = ca[0]
            agg = np.zeros_like(ca)
            for i in range(len(ca)):
                agg[i] = ca[i] @ kabsch(ca[i], ref)
            # iterate reference once toward the mean for stability
            ref = agg.mean(0)
            for i in range(len(ca)):
                agg[i] = ca[i] @ kabsch(ca[i], ref)
            mean_struct = agg.mean(0)
            rmsf = np.sqrt(((agg - mean_struct) ** 2).sum(2).mean(0))   # (RESN,)
            pick = np.linspace(0, len(agg) - 1, min(N_ENSEMBLE, len(agg))).astype(int)
            ens = [[[round(float(v), 1) for v in row] for row in agg[i]] for i in pick]
            states_out[str(s)] = dict(
                n=a["n"],
                beta=[round(float(x), 3) for x in a["beta"] / a["n"]],
                helix=[round(float(x), 3) for x in a["helix"] / a["n"]],
                drug=[round(float(x), 3) for x in a["ndrug"] / a["n"]] if drug else None,
                drugNull=[round(float(x), 3) for x in (a["null"] / a["nnull"])] if (drug and a["nnull"]) else None,
                rgMean=round(float(np.mean(a["rg"])), 2), rgStd=round(float(np.std(a["rg"])), 2),
                betaMean=round(float(np.mean(a["bfrac"])), 3), betaStd=round(float(np.std(a["bfrac"])), 3),
                rmsf=[round(float(x), 2) for x in rmsf],
                ensemble=ens,
            )
        out["params"][str(Y)] = dict(
            states=states_out,
            landscape=[[int(x) for x in row] for row in land[Y]],
        )
    print(f"  {tag}: {nused} frames, states(3)={sorted(int(k) for k in out['params']['3']['states'])}")
    return out


def main():
    global CA_RESERVOIR, N_ENSEMBLE
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(HERE, "data", "track1_struct2.json"))
    # Raised for the correspondence analysis, which needs a real sample per state rather
    # than the handful of frames the prototype inlines for display.
    ap.add_argument("--reservoir", type=int, default=CA_RESERVOIR)
    ap.add_argument("--nens", type=int, default=N_ENSEMBLE)
    args = ap.parse_args()
    CA_RESERVOIR, N_ENSEMBLE = args.reservoir, args.nens
    print(f"reservoir={CA_RESERVOIR} frames/state, storing {N_ENSEMBLE}/state")
    data = {"meta": {"nres": RESN, "stride": args.stride,
                     "rgEdges": [round(float(x), 2) for x in RG_EDGES],
                     "betaEdges": [round(float(x), 3) for x in BETA_EDGES],
                     "regions": REGIONS, "drugCutoff": DRUG_CUTOFF},
            "systems": []}
    for tag, zid, label, drug in SYS:
        if not os.path.isdir(os.path.join(args.root, tag)):
            print(f"  (skip {tag})"); continue
        print(f"processing {tag} ...")
        data["systems"].append(dict(id=zid, tag=tag, **process(tag, zid, label, drug, args.root, args.stride)))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(data, open(args.out, "w"), separators=(",", ":"))
    print(f"wrote {args.out} ({os.path.getsize(args.out)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
