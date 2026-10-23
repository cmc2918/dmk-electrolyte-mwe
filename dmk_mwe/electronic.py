"""Electronic structure of the Li+ shell from the r2SCAN single points on AIMD snapshots (Methods, 'Electronic structure of
the solvation shell'; Figs. 9 and 10): REPEAT charges of the donor atoms, Coulomb preference of a shell site for DMK
over FSI-, and molecular dipoles.

Conventions (as published):
  atoms       CP2K order of every snapshot: 54 DMK x (C C O C H6), 6 Li+, 6 FSI- x (N S O O S O O F F)
  d           distance of a donor O to its nearest Li+ (minimum image in the snapshot's cubic cell)
  bound       DMK with its carbonyl O within 3.1 A of a Li+; coordinating sulfonyl O within 2.9 A
  dipoles     Wannier: mu = sum_I Z_I R_I - 2 sum_n r_n, each centre on its nearest nucleus (sp_acetone.csv, 'wan';
              snapshots whose localisation was capped are excluded); REPEAT and OPLS-AA: sum_j q_j (r_j - r_O)
  E_elec      q_Li sum_j q_j / r_Li,j over the atoms of a ligand bound to that Li+, for REPEAT, OPLS-AA and OPLS-ECC
              (FSI- charges x 0.725 and q_Li 0.725; DMK unscaled) on the same geometries      -> `coulomb_pairs`
  errors      bootstrap over snapshots (1000 resamples, seed 0)                               -> `boot_mean`
"""
import csv
import numpy as np

DEBYE = 4.80320                                         # e A in D
KE = 14.399645                                          # e^2 / (4 pi eps0) in eV A
DMK = np.arange(540).reshape(54, 10)
LI = np.arange(540, 546)
FSI = 546 + np.arange(54).reshape(6, 9)
SO = [2, 3, 5, 6]                                       # sulfonyl O within an FSI-
Q_DMK = np.array([-0.18, 0.47, -0.47, -0.18] + [0.06] * 6)                       # OPLS-AA (OPLS-ECC leaves DMK unscaled)
Q_FSI = np.array([-0.66, 1.02, -0.53, -0.53, 1.02, -0.53, -0.53, -0.13, -0.13])   # OPLS-AA; x 0.725 in OPLS-ECC
R_BOUND_DMK, R_BOUND_FSI = 3.1, 2.9


def manifest(path):
    return {r["name"]: r for r in csv.DictReader(open(path))}


def read_snapshot(path):
    """Coordinates (A) of a single-point snapshot (.xyz)."""
    return np.array([l.split()[1:4] for l in open(path).read().splitlines()[2:]], float)


def read_repeat(path):
    """REPEAT charges (e) of the 600 atoms, from a CP2K '-resp.dat' file."""
    return np.array([float(p[3]) for p in (l.split() for l in open(path)) if len(p) == 4 and p[0] == "RESP"])


def li_distances(X, L):
    D = X[None, :, :] - X[LI][:, None, :]
    D -= L * np.round(D / L)
    return np.linalg.norm(D, axis=2)                    # (6 Li+, 600 atoms)


def point_charge_dipole(X, q, idx, L, origin=2):
    """|sum q_j (r_j - r_origin)| (D) of the atoms idx, unwrapped about atom idx[origin]."""
    v = X[idx] - X[idx[origin]]
    v -= L * np.round(v / L)
    return float(np.linalg.norm((q[:, None] * v).sum(0)) * DEBYE)


def donor_charges(X, q, L):
    """(d, q) of every carbonyl O and of every sulfonyl O."""
    R = li_distances(X, L)
    carb = [(R[:, DMK[m][2]].min(), q[DMK[m][2]]) for m in range(54)]
    sulf = [(R[:, FSI[k][s]].min(), q[FSI[k][s]]) for k in range(6) for s in SO]
    return carb, sulf


def coulomb_pairs(X, q, L, r_dmk=R_BOUND_DMK, r_fsi=R_BOUND_FSI):
    """E_elec (eV) of each bound DMK and each coordinating FSI- with its Li+: columns d, REPEAT, OPLS-ECC, OPLS-AA."""
    R = li_distances(X, L)
    out = {"dmk": [], "fsi": []}
    for a, li in enumerate(LI):
        r = R[a]
        for m in range(54):
            idx = DMK[m]
            if r[idx[2]] < r_dmk:
                v = 1 / r[idx]
                out["dmk"].append((r[idx[2]], KE * q[li] * (q[idx] * v).sum(), KE * 0.725 * (Q_DMK * v).sum(),
                                   KE * (Q_DMK * v).sum()))
        for k in range(6):
            idx = FSI[k]
            ds = r[idx[SO]].min()
            if ds < r_fsi:
                v = 1 / r[idx]
                out["fsi"].append((ds, KE * q[li] * (q[idx] * v).sum(), KE * 0.725 * 0.725 * (Q_FSI * v).sum(),
                                   KE * (Q_FSI * v).sum()))
    return {k: np.array(v) for k, v in out.items()}


def boot_mean(values, groups, n=1000, seed=0):
    """Mean and bootstrap SE of values, resampling whole groups (snapshots)."""
    values, groups = np.asarray(values, float), np.asarray(groups)
    u, inv = np.unique(groups, return_inverse=True)
    s, c = np.bincount(inv, weights=values, minlength=len(u)), np.bincount(inv, minlength=len(u)).astype(float)
    B = np.random.default_rng(seed).integers(0, len(u), size=(n, len(u)))
    return float(s.sum() / c.sum()), float((s[B].sum(1) / c[B].sum(1)).std())
