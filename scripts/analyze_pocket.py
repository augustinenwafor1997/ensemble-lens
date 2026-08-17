"""
Bio+MedVis Challenge 2026 - Track 1 - Task T3, binding-pocket extraction.
Written after the expert meeting (Dr. First): "show me the structures, and show me
where the drug actually sits."

The .xtc files carry full all-atom coordinates for BOTH the peptide and the 100 drug
copies, so a real binding-pocket view is recoverable. For each system (TMP, SPA), each
Markov-state parametrisation (2/3/4) and each state, this script produces:

  - REFERENCE frame: the state's mean CA structure (Kabsch-superposed), so every pose
    and every density point below lives in ONE consistent coordinate frame per state.
  - BOUND POSES: the frames with the most peptide-drug residue contacts, stored as
    peptide CA trace + the bound drug copy's heavy atoms + the pocket lining (peptide
    heavy atoms near the drug) + the contacting residue list.
  - BINDING-SITE DENSITY: the drug centroid of every bound frame, rotated into the
    state reference frame -> a point cloud showing WHERE on the peptide the ligand sits.
  - Per-state contact frequency for the pocket residues.

Only the closest drug copy of the 100 is considered per frame: that is the one that is
actually engaged, the other 99 are bulk ligand.

PBC: the box is periodic and the 100 copies are scattered across images, so every
peptide-drug distance passes box=ts.dimensions (min image). The peptide itself is whole
and must NOT be min-imaged. Drug copies are made whole about their own first atom before
being placed next to the peptide.

Run:  py -3.10 analyze_pocket.py --root <dir with ZS-ab*/> [--stride N]
Output: data/track1_pocket.json
"""
import os, glob, json, argparse, warnings
import numpy as np
warnings.filterwarnings("ignore")
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array, minimize_vectors

HERE = os.path.dirname(os.path.abspath(__file__))
STATES_ROOT = r"C:/Users/AUGUST~1/bv/t1"
RESN = 42
CONTACT_CUTOFF = 4.5      # A, peptide residue counts as touching the ligand
POCKET_CUTOFF = 8.0       # A, peptide heavy atoms drawn as the pocket lining
PRESCREEN = 14.0          # A, CA-based screen before the expensive all-atom pass
RESERVOIR = 320           # bound frames kept per (parametrisation, state)
N_POSES = 3               # fully detailed poses emitted per state
N_CLOUD = 260             # drug-centroid density points emitted per state

SYS = [("ZS-ab3", "zsab3", "Abeta42 + Tramiprosate (TMP)", "TMP"),
       ("ZS-ab4", "zsab4", "Abeta42 + SPA", "SPA")]
AA3to1 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
          'GLY': 'G', 'HIS': 'H', 'HSD': 'H', 'HSE': 'H', 'HSP': 'H', 'ILE': 'I', 'LEU': 'L',
          'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
          'TYR': 'Y', 'VAL': 'V'}


def elem_of(name):
    """Element from a PDB atom name (the files carry no element column we can trust)."""
    n = name.strip().lstrip("0123456789")
    if not n:
        return "C"
    if n[:2].upper() in ("CL", "BR", "NA", "MG", "ZN", "CA"):
        # inside a peptide 'CA' is the alpha carbon, not calcium
        return "C" if n[:2].upper() == "CA" else n[:2].capitalize()
    return n[0].upper()


def load_states(zid):
    out = {}
    for Y in (2, 3, 4):
        p = os.path.join(STATES_ROOT, zid, f"{zid}_states{Y}.txt")
        d = {}
        for ln in open(p, encoding="utf-8"):
            if ":" in ln:
                n, s = ln.split(":", 1)
                d[n.strip()] = np.array([int(x) for x in s.strip().replace(" ", "").split(",") if x != ""],
                                        dtype=np.int8)
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
    """Rotation aligning centered P onto centered Q; apply as P @ R."""
    H = P.T @ Q
    V, S, Wt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Wt.T @ V.T))
    return V @ np.diag([1.0, 1.0, d]) @ Wt


def make_whole(pos, box):
    """Reassemble a molecule split across a periodic boundary, anchored on its first atom."""
    return pos[0] + minimize_vectors(pos - pos[0], box)


def process(tag, zid, label, drug, root, stride, limit=0):
    top = os.path.join(root, tag, "filtered.pdb")
    u = mda.Universe(top)
    pep = u.select_atoms("not resname MOL")
    drg = u.select_atoms("resname MOL")
    if not len(drg):
        return None

    starts = residue_blocks(pep)
    pep_idx, drug_idx = pep.indices, drg.indices
    ca_local = np.array([i for i, a in enumerate(pep.atoms) if a.name == "CA"])
    seq = "".join(AA3to1.get(r.resname, "X") for r in pep.residues)

    pep_elems = np.array([elem_of(a.name) for a in pep.atoms])
    pep_heavy = np.where(pep_elems != "H")[0]                 # local indices into pep
    pep_heavy_res = np.searchsorted(starts, pep_heavy, side="right") - 1
    pep_heavy_elem = pep_elems[pep_heavy]
    pep_heavy_name = np.array([a.name for a in pep.atoms])[pep_heavy]

    ncopy = len(drg.residues)
    apc = len(drg) // ncopy                                    # atoms per drug copy
    d_elems = np.array([elem_of(a.name) for a in drg.residues[0].atoms])
    d_heavy = np.where(d_elems != "H")[0]
    d_heavy_elem = [str(x) for x in d_elems[d_heavy]]

    states = load_states(zid)
    sim_dirs = sorted(glob.glob(os.path.join(root, tag, "e*s*_*")))
    if limit:
        sim_dirs = sim_dirs[:limit]

    frames = []                     # every kept bound frame, shared across parametrisations
    slots = {Y: {} for Y in (2, 3, 4)}   # Y -> state -> [frame index, ...]
    contact_n = {Y: {} for Y in (2, 3, 4)}   # Y -> state -> [nframes, contact counts per residue]
    nseen = nbound = 0

    for sd in sim_dirs:
        sim = os.path.basename(sd)
        xtc = os.path.join(sd, "output.filtered.xtc")
        if not os.path.exists(xtc) or sim not in states[3]:
            continue
        uu = mda.Universe(top, xtc)
        pep_ag, drug_ag = uu.atoms[pep_idx], uu.atoms[drug_idx]
        slen = len(states[3][sim])
        for ts in uu.trajectory[::stride]:
            fi = ts.frame
            if fi >= slen:
                break
            nseen += 1
            box = ts.dimensions
            ppos = pep_ag.positions
            dpos = drug_ag.positions
            capos = ppos[ca_local]

            # cheap screen: CA atoms vs all ligand atoms, collapsed per copy
            Dca = distance_array(capos, dpos, box=box)
            percopy = Dca.reshape(len(capos), ncopy, apc).min(axis=(0, 2))
            best = int(np.argmin(percopy))
            if percopy[best] > PRESCREEN:
                continue

            cpos = make_whole(dpos[best * apc:(best + 1) * apc], box)
            # place the copy in the peptide's periodic image
            pcen = ppos.mean(0)
            ccen = cpos.mean(0)
            cpos = pcen + minimize_vectors((ccen - pcen)[None, :], box)[0] + (cpos - ccen)

            D = distance_array(ppos, cpos)          # already in the same image, no box
            resmin = np.minimum.reduceat(D, starts, axis=0).min(1)
            touch = resmin < CONTACT_CUTOFF
            ncon = int(touch.sum())
            if ncon == 0:
                continue
            nbound += 1

            sset = {Y: int(states[Y][sim][fi]) for Y in (2, 3, 4)}
            for Y in (2, 3, 4):
                s = sset[Y]
                c = contact_n[Y].setdefault(s, [0, np.zeros(RESN)])
                c[0] += 1
                c[1] += touch

            want = any(len(slots[Y].setdefault(sset[Y], [])) < RESERVOIR for Y in (2, 3, 4))
            if not want:
                continue
            fidx = len(frames)
            frames.append(dict(
                ca=capos.astype(np.float32).copy(),
                heavy=ppos[pep_heavy].astype(np.float32).copy(),
                drug=cpos[d_heavy].astype(np.float32).copy(),
                ncon=ncon,
                contacts=np.where(touch)[0].astype(np.int16),
                resmin=resmin.astype(np.float32),
            ))
            for Y in (2, 3, 4):
                lst = slots[Y].setdefault(sset[Y], [])
                if len(lst) < RESERVOIR:
                    lst.append(fidx)

    out = {"label": label, "tag": tag, "id": zid, "drug": drug, "seq": seq,
           "drugElems": d_heavy_elem, "nDrugCopies": ncopy, "params": {}}

    for Y in (2, 3, 4):
        states_out = {}
        for s, idxs in sorted(slots[Y].items()):
            if len(idxs) < 3:
                continue
            ca = np.array([frames[i]["ca"] for i in idxs], dtype=np.float64)
            cen = ca.mean(1, keepdims=True)
            ca_c = ca - cen
            ref = ca_c[0]
            for _ in range(2):                       # one refinement pass toward the mean
                rots = [kabsch(c, ref) for c in ca_c]
                agg = np.array([c @ R for c, R in zip(ca_c, rots)])
                ref = agg.mean(0)
            mean_ca = agg.mean(0)

            # binding-site density: the bound ligand centroid in the state reference frame
            cloud = []
            for j, i in enumerate(idxs):
                dc = frames[i]["drug"].astype(np.float64).mean(0)
                cloud.append((dc - cen[j, 0]) @ rots[j])
            cloud = np.array(cloud)
            pick = np.linspace(0, len(cloud) - 1, min(N_CLOUD, len(cloud))).astype(int)

            # detailed poses: most residue contacts wins
            order = sorted(range(len(idxs)), key=lambda j: -frames[idxs[j]]["ncon"])[:N_POSES]
            poses = []
            for j in order:
                f = frames[idxs[j]]
                R, o = rots[j], cen[j, 0]
                dg = (f["drug"].astype(np.float64) - o) @ R
                near = np.where(np.linalg.norm(
                    f["heavy"].astype(np.float64)[:, None, :] - f["drug"].astype(np.float64)[None, :, :],
                    axis=2).min(1) < POCKET_CUTOFF)[0]
                lining = (f["heavy"].astype(np.float64)[near] - o) @ R
                poses.append(dict(
                    ca=[[round(float(v), 2) for v in row] for row in agg[j]],
                    drug=[[round(float(v), 2) for v in row] for row in dg],
                    pocket=[[round(float(v), 2) for v in row] for row in lining],
                    pocketRes=[int(x) for x in pep_heavy_res[near]],
                    pocketElem=[str(x) for x in pep_heavy_elem[near]],
                    contacts=[int(x) for x in f["contacts"]],
                    nContacts=int(f["ncon"]),
                    minDist=round(float(f["resmin"].min()), 2),
                ))

            cn, cf = contact_n[Y][s]
            states_out[str(s)] = dict(
                nBound=int(cn),
                ref=[[round(float(v), 2) for v in row] for row in mean_ca],
                poses=poses,
                cloud=[[round(float(v), 1) for v in cloud[p]] for p in pick],
                contactFreq=[round(float(x), 3) for x in (cf / max(cn, 1))],
                topRes=[int(i) for i in np.argsort(-cf)[:8]],
            )
        out["params"][str(Y)] = dict(states=states_out)

    tot = sum(v[0] for v in contact_n[3].values())
    print(f"  {tag}: {nseen} frames scanned, {nbound} bound ({100*nbound/max(nseen,1):.1f}%), "
          f"{len(frames)} kept, states(3)={sorted(int(k) for k in out['params']['3']['states'])}, "
          f"bound-frame total(3)={tot}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="only the first N simulations per system (smoke test)")
    ap.add_argument("--out", default=os.path.join(HERE, "data", "track1_pocket.json"))
    args = ap.parse_args()
    data = {"meta": {"nres": RESN, "stride": args.stride,
                     "contactCutoff": CONTACT_CUTOFF, "pocketCutoff": POCKET_CUTOFF},
            "systems": []}
    for tag, zid, label, drug in SYS:
        if not os.path.isdir(os.path.join(args.root, tag)):
            print(f"  (skip {tag})")
            continue
        print(f"processing {tag} ...", flush=True)
        r = process(tag, zid, label, drug, args.root, args.stride, args.limit)
        if r:
            data["systems"].append(r)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(data, open(args.out, "w"), separators=(",", ":"))
    print(f"wrote {args.out} ({os.path.getsize(args.out)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
