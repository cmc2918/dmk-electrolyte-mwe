#!/usr/bin/env python3
"""Electronic structure of the shell (Figs. 9 and 10): REPEAT charges of the donor O against their distance to Li+,
the Coulomb preference of a shell site for DMK over FSI- with three charge sets, and the DMK dipole from the Wannier
centres, the REPEAT charges, OPLS-AA and ByteFF-Pol.

Inputs: 1_ab_initio_md/single_points (manifest.csv, snapshots/*.xyz, repeat_charges/*-resp.dat, sp_acetone.csv) and
3_polarizable_ff/analysis/induced_dipoles/rep*_dmk.npz.

    DMK_DATA=/path/to/helix_data python3 examples/05_electronic_structure.py
"""
import glob, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dmk_mwe import data
from dmk_mwe import electronic as el

QBINS = np.r_[np.arange(1.70, 3.21, 0.10), np.arange(3.5, 8.01, 0.5)]

if __name__ == "__main__":
    sp = data("1_ab_initio_md", "single_points")
    man = el.manifest(f"{sp}/manifest.csv")
    carb, sulf, pairs, opls_dip = [], [], {"dmk": [], "fsi": []}, []
    for name in sorted(man):
        X, L = el.read_snapshot(f"{sp}/snapshots/{name}.xyz"), float(man[name]["cell_A"])
        q = el.read_repeat(f"{sp}/repeat_charges/{name}-resp.dat")
        c, s = el.donor_charges(X, q, L)
        carb += [(name, *x) for x in c]
        sulf += [(name, *x) for x in s]
        for k, v in el.coulomb_pairs(X, q, L).items():
            pairs[k] += [(name, *row) for row in v]
        opls_dip += [(name, el.point_charge_dipole(X, el.Q_DMK, el.DMK[m], L)) for m in range(54)]
    print(f"{len(man)} r2SCAN single points")

    # ---------------------------------------------------------------- donor O charges (Fig. 9b, c)
    C = pd.DataFrame(carb, columns=["snap", "d", "q"])
    S = pd.DataFrame(sulf, columns=["snap", "d", "q"])
    print("\nREPEAT charge of the donor O (e), binned by its distance to the nearest Li+")
    for lab, T in (("carbonyl O", C), ("sulfonyl O", S)):
        b = T.groupby(pd.cut(T.d, QBINS), observed=True).q.agg(["mean", "count"])
        near = b.loc[[i for i in b.index if i.left >= 1.79 and i.right <= 1.91][0]]
        bulk = el.boot_mean(T.q[T.d > 5.0], T.snap[T.d > 5.0])
        print(f"  {lab}: beyond 5 A {bulk[0]:+.3f} +- {bulk[1]:.3f};  1.8-1.9 A bin {near['mean']:+.3f} (n {int(near['count'])})")
    print("  published: carbonyl -0.566 free -> -0.620 at 1.86 A; sulfonyl -0.512 -> -0.582")

    # ---------------------------------------------------------------- Coulomb preference (Fig. 9d)
    print("\nE_elec(DMK) / E_elec(FSI-) over all bound ligands (DMK O < 3.1 A, sulfonyl O < 2.9 A)")
    A, B = (np.array([r[1:] for r in pairs[k]], float) for k in ("dmk", "fsi"))
    for j, lab in ((1, "REPEAT"), (2, "OPLS-ECC"), (3, "OPLS-AA")):
        print(f"  {lab:9s} {A[:, j].mean() / B[:, j].mean():.2f}", end="")
    print("   (published 0.38, 0.41, 0.30)")

    # ---------------------------------------------------------------- dipoles (Fig. 10a)
    T = pd.read_csv(f"{sp}/sp_acetone.csv")
    W = T[~T.capped & T.wan.notna()]
    print(f"\nDMK dipole (D), free / bound (carbonyl O within 3.1 A of a Li+); Wannier on {W.name.nunique()} snapshots")
    for lab, df, col in (("Wannier", W, "wan"), ("REPEAT ", T, "resp_dip")):
        f, b = (el.boot_mean(df[col][m], df.snap[m]) for m in (~df.bound, df.bound))
        print(f"  {lab}  {f[0]:.2f} +- {f[1]:.3f} / {b[0]:.2f} +- {b[1]:.3f}   change {b[0] - f[0]:+.2f} D ({100 * (b[0] / f[0] - 1):+.1f} %)")
    o = np.array([v for _, v in opls_dip])
    print(f"  OPLS-AA  {o.mean():.2f} on the AIMD geometries (SD {o.std():.3f}, range {o.min():.2f}-{o.max():.2f})")
    Z = np.concatenate([np.load(f, allow_pickle=True)["data"]
                        for f in sorted(glob.glob(data("3_polarizable_ff", "analysis", "induced_dipoles", "rep*_dmk.npz")))])
    Z = pd.DataFrame(Z, columns=["frame", "d", "bound", "CN", "p_perm", "p_ind_par", "p_tot", "muO", "muO_par"])
    zf, zb = Z.p_tot[Z.bound == 0].mean(), Z.p_tot[Z.bound == 1].mean()
    print(f"  ByteFF-Pol {zf:.2f} / {zb:.2f}   change {zb - zf:+.2f} D ({100 * (zb / zf - 1):+.1f} %), {len(Z):,} molecules")
    print("  published: Wannier 3.97 / 4.90 (+23.5 %); REPEAT 3.70, +0.34 D; OPLS-AA 3.11; ByteFF-Pol 3.85 / 4.94 (+28.5 %)")
    bins = [1.7, 2.0, 2.2, 2.5, 3.1, 4.0, 5.0, 6.0, 8.0, 30.0]
    sw = W.groupby(pd.cut(W.d, bins), observed=True).wan.std()
    sb = Z.groupby(pd.cut(Z.d, bins), observed=True).p_tot.std()
    print(f"  SD at fixed distance: Wannier {sw.min():.2f}-{sw.max():.2f}, ByteFF-Pol {sb.min():.2f}-{sb.max():.2f} D"
          "   (published 0.32-0.37, 0.27-0.33)")
