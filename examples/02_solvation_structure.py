#!/usr/bin/env python3
"""Solvation structure: Table 2 (first peak, first minimum and CN of the Li+-O RDFs for the four models) and the
speciation and coordination columns of Tables 3-5 (OPLS-ECC sweeps).

  OPLS-ECC, ByteFF-Pol  per-replica RDFs and Li+-O contact lists (SolvationAnalysis output) of the deposit
  MACE-FT               RDFs and contacts computed here from the 35 ns trajectory (every 0.5 ps frame); spread over
                        four contiguous blocks
  AIMD                  RDFs computed here from each curated r2SCAN trajectory (every 10 fs frame)

Values in parentheses are standard deviations over replicas (or blocks), in units of the last digit.

    DMK_DATA=/path/to/helix_data python3 examples/02_solvation_structure.py [--skip-trajectories]
"""
import glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dmk_mwe import data
from dmk_mwe import solvation as sv

PUBLISHED = {   # SSIP/CIP/AGG, CN carbonyl, CN sulfonyl, CN total, O/FSI
    "charge_sweep/q_0.650": "67/30/3 4.20 0.54 4.74 1.30", "charge_sweep/q_0.675": "61/34/5 4.24 0.62 4.86 1.30",
    "charge_sweep/q_0.700": "56/38/6 4.24 0.72 4.96 1.32", "charge_sweep/q_0.725": "49/43/8 4.21 0.86 5.07 1.35",
    "charge_sweep/q_0.750": "41/47/11 4.15 1.02 5.17 1.38", "charge_sweep/q_0.800": "25/55/21 3.94 1.46 5.40 1.47",
    "lj_cross_sweep/dsigma_+0.05": "58/37/5 4.35 0.67 5.02 1.30", "lj_cross_sweep/dsigma_+0.10": "65/31/3 4.47 0.55 5.02 1.28",
    "lj_cross_sweep/dsigma_+0.15": "72/26/2 4.57 0.44 5.00 1.25", "lj_cross_sweep/dsigma_+0.20": "77/21/1 4.61 0.37 4.98 1.22",
    "lj_cross_sweep/dsigma_+0.25": "82/17/1 4.66 0.31 4.98 1.23", "solvent_charge_sweep/f_0.85": "8/48/44 3.35 2.13 - -",
    "solvent_charge_sweep/f_1.10": "76/23/2 4.35 0.36 - -", "solvent_charge_sweep/f_1.15": "85/15/1 4.29 0.22 - -",
    "solvent_charge_sweep/f_1.20": "91/9/<1 4.19 0.12 - -", "solvent_charge_sweep/f_1.25": "95/5/<1 4.10 0.06 - -"}
TABLE2 = {"OPLS-ECC": "2.10(0) 3.70(6) 4.21(1) 2.18(1) 3.37(5) 0.86(1)",
          "ByteFF-Pol": "2.06(0) 3.16(7) 3.88(0) 2.10(1) 2.99(6) 0.24(0)",
          "MACE-FT": "1.94(0) 3.02(7) 3.82(3) 1.98(2) 2.82(9) 0.21(3)",
          "AIMD run 1": "1.94 3.06 3.57 1.98 2.66 0.45", "AIMD run 2": "1.94 3.02 3.25 2.02 2.90 0.86",
          "AIMD run 3": "1.94 3.14 3.62 2.02 3.10 0.50"}


# Box edges (A) behind the published coordination numbers: recovered for each system from the RDF normalisation
# (contacts within 3.0 A against the integral of g), as in the figure scripts; ByteFF-Pol: cube root of the mean NVT
# box volume. Re-deriving them from the deposited contact lists (box_edge_from_contacts) gives edges 0.03 % larger,
# which moves a few CNs by 0.01; both are printed.
BOX = {"charge_sweep/q_0.650": 43.286, "charge_sweep/q_0.675": 43.193, "charge_sweep/q_0.700": 43.149,
       "charge_sweep/q_0.725": 43.120, "charge_sweep/q_0.750": 43.085, "charge_sweep/q_0.800": 43.045,
       "lj_cross_sweep/dsigma_+0.05": 43.139, "lj_cross_sweep/dsigma_+0.10": 43.105, "lj_cross_sweep/dsigma_+0.15": 43.045,
       "lj_cross_sweep/dsigma_+0.20": 43.147, "lj_cross_sweep/dsigma_+0.25": 43.130,
       "solvent_charge_sweep/f_0.85": 43.431, "solvent_charge_sweep/f_1.10": 43.009, "solvent_charge_sweep/f_1.15": 42.977,
       "solvent_charge_sweep/f_1.20": 42.861, "solvent_charge_sweep/f_1.25": 42.848, "ByteFF-Pol": 35.224}


def f2(m, s=None):
    return f"{m:.2f}" if s is None else f"{m:.2f}({int(round(s * 100))})"


def from_contact_lists(folder, n_salt, n_dmk, L_published=None):
    """Replica-level analysis of SolvationAnalysis output: rdf_{DMK,FSI}_repN.dat, solvation_data_repN.parquet."""
    import pandas as pd
    reps = sorted(glob.glob(f"{folder}/rep*/"), key=lambda p: (len(p), p))
    g = {"DMK": [], "FSI": []}
    peaks, minima, Ls, spec, mol_fsi = {"DMK": [], "FSI": []}, {"DMK": [], "FSI": []}, [], [], []
    for d in reps:
        n = os.path.basename(d.rstrip("/"))[3:]
        df = pd.read_parquet(f"{d}solvation_data_rep{n}.parquet", columns=["frame", "solute_ix", "solvent_ix",
                                                                         "distance", "solvent"])
        nf, nli = df["frame"].nunique(), n_salt
        for s in ("DMK", "FSI"):
            r, gs = np.loadtxt(f"{d}rdf_{s}_rep{n}.dat").T[:2]
            g[s].append(gs)
            p, m = sv.first_peak_and_minimum(r, gs)
            peaks[s].append(p)
            minima[s].append(m)
        dm = df[df["solvent"] == "DMK"]
        Ls.append(sv.box_edge_from_contacts((dm["distance"] < 3.0).sum(), nf, nli, r, g["DMK"][-1], n_dmk))
        fs = df[(df["solvent"] == "FSI")]
        c = fs[fs["distance"] < 2.9]
        spec.append(sv.speciation(c["frame"].values, c["solvent_ix"].values, c["solute_ix"].values, nf, n_salt))
        mol_fsi.append(len(fs) / (nf * nli))                       # anions bound per Li+ (one row per molecule)
    L = L_published if L_published else np.mean(Ls)
    rho = {"DMK": n_dmk / L ** 3, "FSI": 4 * n_salt / L ** 3}
    out = {"spec": np.mean(spec, 0), "n": len(reps), "L_derived": float(np.mean(Ls)), "L": L}
    for s in ("DMK", "FSI"):
        gm, rm = np.mean(g[s], 0), np.mean(minima[s])
        out[s] = {"cn": sv.cn_at(r, gm, rho[s], rm), "r_min": (rm, np.std(minima[s], ddof=1)),
                  "r_peak": (np.mean(peaks[s]), np.std(peaks[s], ddof=1)),
                  "cn_sd": np.std([sv.cn_at(r, gi, rho[s], rm) for gi in g[s]], ddof=1)}
    out["O_per_FSI"] = out["FSI"]["cn"] / np.mean(mol_fsi)
    return out


def table2_row(o):
    return " ".join([f2(*o["DMK"]["r_peak"]), f2(*o["DMK"]["r_min"]), f2(o["DMK"]["cn"], o["DMK"]["cn_sd"]),
                     f2(*o["FSI"]["r_peak"]), f2(*o["FSI"]["r_min"]), f2(o["FSI"]["cn"], o["FSI"]["cn_sd"])])


def from_trajectory(X, L, li, carb_o, sulf_o, n_anion_o=4, blocks=None):
    """Whole-trajectory r_peak, r_min, CN (at the whole-trajectory r_min) and, optionally, their spread over blocks."""
    def one(Y, rmins=None):
        res, rm = {}, {}
        for s, idx in (("DMK", carb_o), ("FSI", sulf_o)):
            r, g = sv.rdf(Y, li, idx, L)
            p, m = sv.first_peak_and_minimum(r, g)
            rm[s] = m
            res[s] = (p, m, sv.cn_at(r, g, len(idx) / L ** 3, m if rmins is None else rmins[s]))
        return res, rm
    whole, rm = one(X)
    if not blocks:
        return whole, None
    bl = [one(Y, rm)[0] for Y in np.array_split(X, blocks)]
    sd = {s: [np.std([b[s][k] for b in bl], ddof=1) for k in range(3)] for s in ("DMK", "FSI")}
    return whole, sd


def traj_speciation(X, L, li, sulf_o_by_anion, cutoff=2.9):
    o = np.concatenate(sulf_o_by_anion)
    d = np.linalg.norm(sv.mic(X[:, o][:, :, None] - X[:, li][:, None], L), axis=3) < cutoff     # (T, n_O, n_Li)
    per_anion = d.reshape(len(X), len(sulf_o_by_anion), -1, len(li)).any(2).sum(2)             # Li per anion
    h = np.bincount(np.minimum(per_anion.ravel(), 2), minlength=3)
    return 100.0 * h / h.sum()


if __name__ == "__main__":
    print("OPLS-ECC sweeps: SSIP/CIP/AGG (%)  CN carbonyl  CN sulfonyl  CN total  O/FSI")
    t2 = {}
    for cond, pub in PUBLISHED.items():
        o = from_contact_lists(data("4_nonpolarizable_ff", cond, "solvation"), 65, 600, BOX[cond])
        sp = "/".join(f"{x:.0f}" if x >= 0.5 else "<1" for x in o["spec"])
        print(f"  {cond:30s} {sp:9s} {o['DMK']['cn']:.2f} {o['FSI']['cn']:.2f} {o['DMK']['cn'] + o['FSI']['cn']:.2f} "
              f"{o['O_per_FSI']:.2f}   published {pub}   (L {o['L']:.3f}, from contacts {o['L_derived']:.3f})")
        if cond == "charge_sweep/q_0.725":
            t2["OPLS-ECC"] = o
    print("\nTable 2: r_peak r_min CN (carbonyl) | r_peak r_min CN (sulfonyl)")
    o = from_contact_lists(data("3_polarizable_ff", "analysis", "solvation"), 35, 329, BOX["ByteFF-Pol"])
    t2["ByteFF-Pol"] = o
    for name in ("OPLS-ECC", "ByteFF-Pol"):
        print(f"  {name:11s} {table2_row(t2[name]):44s} published {TABLE2[name]}   (n = {t2[name]['n']})")
    print(f"  ByteFF-Pol speciation {'/'.join(f'{x:.1f}' for x in o['spec'])} %")
    if "--skip-trajectories" in sys.argv:
        sys.exit()
    # MACE-FT: from the deposited trajectory (atoms: 54 DMK x 10, 6 Li+, 6 FSI- x 9)
    import MDAnalysis as mda
    d = data("2_machine_learning_ff", "trajectory")
    seg = json.load(open(f"{d}/segments.json"))
    X = np.concatenate([mda.Universe(f"{d}/{seg['topology']}", f"{d}/{f}").trajectory.timeseries(order="fac")[a:b]
                        for f, a, b in seg["segments"]])[int(seg["skip_ps"] / seg["dt_ps"]):].astype(float)
    li, carb = np.arange(540, 546), np.arange(54) * 10 + 2
    sulf = [546 + 9 * k + np.array([2, 3, 5, 6]) for k in range(6)]
    w, sd = from_trajectory(X, seg["box_A"], li, carb, np.concatenate(sulf), blocks=4)
    row = " ".join(f"{w[s][k]:.2f}({int(round(sd[s][k] * 100))})" for s in ("DMK", "FSI") for k in range(3))
    print(f"  {'MACE-FT':11s} {row:44s} published {TABLE2['MACE-FT']}   (SD over 4 blocks)")
    print(f"  MACE-FT speciation {'/'.join(f'{x:.1f}' for x in traj_speciation(X, seg['box_A'], li, sulf))} %")
    # AIMD: each curated r2SCAN run, every 10 fs frame
    for i in (1, 2, 3):
        Y, el, L = sv.read_xyz(data("1_ab_initio_md", f"run_{i}", "curated_pos.xyz.xz"))
        w, _ = from_trajectory(Y, L, li, carb, np.concatenate(sulf))
        row = " ".join(f"{w[s][k]:.2f}" for s in ("DMK", "FSI") for k in range(3))
        print(f"  {'AIMD run ' + str(i):11s} {row:44s} published {TABLE2['AIMD run ' + str(i)]}   ({len(Y)} frames, L = {L} A)")
