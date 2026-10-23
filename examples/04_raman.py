#!/usr/bin/env python3
"""Raman (Fig. 3, Methods 'Raman cross-section calculations', SI 'Re-interpretation of the experimental spectra'):
the bound-to-free ratio r of the C-C-C stretch from the ORCA B3LYP/6-31+G(d,p)/CPCM(acetone) clusters, its validation
(band shift, Deng & Irish double ratio, gas-phase check), the solvation number the band reports for each model's
carbonyl CN, and the three decompositions of the measured 1.5 M band.

Inputs: 5_raman/clusters_cpcm/<cluster>/<cluster>.hess and .out (17 clusters + free DMK), 5_raman/clusters_gas_phase,
5_raman/spectrum_refits/chen_fig1a_1.5M.csv (the digitised spectrum of Chen et al. 2024, Fig. 1a), and the carbonyl CNs
of 2_machine_learning_ff/analysis and 3_polarizable_ff/analysis/solvation.

    DMK_DATA=/path/to/helix_data python3 examples/04_raman.py
"""
import glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dmk_mwe import data
from dmk_mwe import raman as ra

WEIGHTS = {"ssip4": 0.86, "cip1": 0.14}             # MACE-FT shares of the bound DMK in the two classes (82 : 13 %)
st = lambda v: f"{np.mean(v):.3f} +- {np.std(v, ddof=1):.3f} (SD; {min(v):.3f}-{max(v):.3f}, n {len(v)})"

if __name__ == "__main__":
    # ---------------------------------------------------------------- r from the clusters
    cp = data("5_raman", "clusters_cpcm")
    free = ra.modes(f"{cp}/dmk/dmk.hess")
    res = {}
    for h in sorted(glob.glob(f"{cp}/*/*.hess")):
        name = os.path.basename(h)[:-5]
        if name != "dmk":
            c = ra.modes(h)
            res[name] = dict(ra.compare(c, free), pol=c["pol"], n_imag=c["n_imag"])
    print(f"{'cluster':10s} {'r':>6s} {'rock':>6s} {'double':>7s} {'shift':>6s}")
    for k, v in res.items():
        print(f"{k:10s} {v['r']:6.3f} {v['rock']:6.3f} {v['double']:7.3f} {v['shift']:+6.1f}")
    cls = {p: [v for k, v in res.items() if k.startswith(p)] for p in WEIGHTS}
    print("\nr, bound / free DMK (published 0.86 +- 0.01 for Li(DMK)4, 0.85 +- 0.01 for Li(DMK)3FSI, 0.86 overall)")
    for p, lab in (("ssip4", "Li(DMK)4    "), ("cip1", "Li(DMK)3FSI ")):
        print(f"  {lab} {st([v['r'] for v in cls[p]])}")
    r_w = sum(w * np.mean([v["r"] for v in cls[p]]) for p, w in WEIGHTS.items())
    print(f"  weighted     {r_w:.3f}")
    sh = [v["shift"] for v in res.values()]
    print(f"\nband shift {min(sh):+.1f} to {max(sh):+.1f} cm-1 (published 12.4-15.6; measured +14)")
    print(f"double ratio r / rock ratio {st([v['double'] for v in res.values()])}  (Deng & Irish 1.03)")
    pol = [(v["pol"] - 0.19) / 4 for k, v in res.items() if k.startswith("ssip4")]        # minus free Li+ (0.19 au)
    print(f"static polarizability per DMK in Li(DMK)4 {min(pol):.1f}-{max(pol):.1f} au, free {free['pol']:.1f} au "
          f"(published 50.9-51.1, 50.5)")
    gp = data("5_raman", "clusters_gas_phase")
    qg = ra.compare(ra.modes(f"{gp}/ssip4_1/ssip4_1.hess"), ra.modes(f"{gp}/dmk/dmk.hess"))
    qc = res["ssip4_1"]
    print(f"gas phase, ssip4_1: shift {qg['shift']:+.1f} (CPCM {qc['shift']:+.1f}) cm-1, double ratio {qg['double']:.2f} "
          f"(CPCM {qc['double']:.2f})")

    # ---------------------------------------------------------------- the solvation number each model's CN implies
    ml = json.load(open(data("2_machine_learning_ff", "analysis", "solvation_summary.json")))["whole"]["cn_DMK"]
    bf = json.load(open(data("3_polarizable_ff", "analysis", "solvation", "replica_summary.json")))["mean_std_sem"]
    cn = {"MACE-FT": ml, "ByteFF-Pol": bf["cn_carbonyl"][0], "OPLS-ECC": 4.21}             # OPLS-ECC: Table 2
    print("\nN = 9 r CN / (r CN + 9 - CN), r = 0.86 (published 3.50, 3.55, 3.87)")
    print("  " + "   ".join(f"{k} CN {v:.2f} -> N {ra.raman_n(v):.2f}" for k, v in cn.items()))

    # ---------------------------------------------------------------- decomposition of the measured band
    d = np.loadtxt(data("5_raman", "spectrum_refits", "chen_fig1a_1.5M.csv"), delimiter=",")
    y = np.interp(ra.GRID, d[:, 0], d[:, 1])
    y /= y.max()
    print("\nChen et al. 1.5 M band, 750-840 cm-1, N = (13.55 / 1.5) A_b / (A_b + A_f)")
    fun = {"common": ra.lg, "lg": ra.lg, "voigt": ra.pv}
    for m, lab in (("common", "common width and shape"), ("lg", "independent Lorentzian-Gaussian"),
                   ("voigt", "independent pseudo-Voigt")):
        f = ra.fit_band(y, m)
        print(f"  {lab:32s} N {f['bound_fraction'] * 13.55 / 1.5:.2f}  rms {f['rms']:.3f}  free {f['free_par'][1]:.1f} / "
              f"bound {f['bound_par'][1]:.1f} cm-1, HWHM {ra.hwhm(fun[m], f['free_par']):.1f} / "
              f"{ra.hwhm(fun[m], f['bound_par']):.1f} cm-1")
