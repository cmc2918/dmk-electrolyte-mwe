#!/usr/bin/env python3
"""Ionic conductivity, its Onsager decomposition, and self-diffusion coefficients (Table 1; sigma of Tables 3-5).

  OPLS-ECC    every published condition, from the per-replica MSDs (4_nonpolarizable_ff/*/*/transport): window on the
              replica average, every replica fitted in it, mean and standard error over replicas
  ByteFF-Pol  11 replicas (3_polarizable_ff/replicas), averaged over the lags they all share, each replica with its own
              box; mean and standard error over replicas
  MACE-FT     self-diffusion from the stitched 35 ns trajectory (2_machine_learning_ff/trajectory), automatic Fickian
              window, standard error over four contiguous blocks (requires MDAnalysis; add --skip-mlff to omit)

All D include the Yeh-Hummer correction with the simulated electrolyte viscosity, 1.06 mPa s.

    DMK_DATA=/path/to/helix_data python3 examples/01_transport.py
"""
import glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dmk_mwe import data
from dmk_mwe import transport as tr

ETA = 1.06e-3                                                   # Pa s
TERMS = ("cation_self", "anion_self", "cat_cat_distinct", "an_an_distinct", "cat_an_cross", "total")
PUBLISHED = {   # sigma (SEM) in mS/cm; Table 1 also gives the five terms and D (1e-9 m^2/s)
    "charge_sweep/q_0.650": "63.0(29)", "charge_sweep/q_0.675": "55.9(28)", "charge_sweep/q_0.700": "51.4(8)",
    "charge_sweep/q_0.725": "44.6(7)", "charge_sweep/q_0.750": "37.8(5)", "charge_sweep/q_0.800": "23.9(4)",
    "lj_cross_sweep/dsigma_+0.05": "46.8(12)", "lj_cross_sweep/dsigma_+0.10": "48.8(14)",
    "lj_cross_sweep/dsigma_+0.15": "46.2(20)", "lj_cross_sweep/dsigma_+0.20": "47.6(7)",
    "lj_cross_sweep/dsigma_+0.25": "49.1(19)", "solvent_charge_sweep/f_0.85": "20.3(8)",
    "solvent_charge_sweep/f_1.10": "44.3(11)", "solvent_charge_sweep/f_1.15": "45.9(24)",
    "solvent_charge_sweep/f_1.20": "44.5(25)", "solvent_charge_sweep/f_1.25": "38.8(9)"}
TABLE1 = {"OPLS-ECC": ("30.4(4) 34.8(4) -9.4(4) -13.6(3) +2.5(3) 44.6(7)", "0.76(1) 0.81(2) 1.40(2)"),
          "ByteFF-Pol": ("31.1(1) 38.6(1) -10.5(5) -13.5(5) +4.9(6) 50.7(8)", "0.80(2) 0.90(2) 1.59(2)"),
          "MACE-FT": (None, "0.71(3) 0.89(3) 1.44(5)")}


def fmt(m, s, d=1):
    """value(uncertainty in units of the last digit), the convention of the paper's tables."""
    if not np.isfinite(s):
        return f"{m:.{d}f}"
    u = int(round(s * 10 ** d))
    return f"{m:+.{d}f}({u})" if m < 0 else f"{m:.{d}f}({u})"


def opls(cond):
    d = data("4_nonpolarizable_ff", cond, "transport")
    info = json.load(open(glob.glob(f"{d}/*_info.json")[0]))
    z = np.load(glob.glob(f"{d}/*_replica_msds.npz")[0])
    t = z["time"]
    keys = {"self_cation": "msd_cat_self_all", "self_anion": "msd_an_self_all", "cat_cat": "msd_cat_cat_all",
            "an_an": "msd_an_an_all", "cat_an": "msd_cat_an_all"}
    reps = [{k: z[v][i] for k, v in keys.items()} for i in range(int(z["n_replicas"]))]
    N, L = info["n_cations"], info["box_length_m"]
    w, mean, sem, per = tr.conductivity(t, reps, N, info["n_anions"], info["charge_cation"], info["charge_anion"],
                                        L ** 3, info["temperature"])
    # self-diffusion: window on the replica-averaged cation self MSD; DMK from the replica-averaged solvent MSD
    avg_c = np.mean([r["self_cation"] for r in reps], 0)
    wd = tr.diffusion_window(t, avg_c)
    yh = tr.yeh_hummer(info["temperature"], ETA, L)
    D = {sp: [tr.slope(t, r[k], wd) / 6 + yh for r in reps] for sp, k in (("Li", "self_cation"), ("FSI", "self_anion"))}
    sol = np.loadtxt(glob.glob(f"{d}/*_solvent_msds.dat")[0])
    D_dmk = tr.slope(sol[:, 0], sol[:, 1], wd) / 6 + yh
    return mean, sem, len(reps), {k: (np.mean(v), np.std(v, ddof=1) / np.sqrt(len(v))) for k, v in D.items()}, D_dmk


def byteff():
    reps = []
    for d in sorted(glob.glob(data("3_polarizable_ff", "replicas", "rep*"))):
        info = json.load(open(glob.glob(f"{d}/*_info.json")[0]))
        m = np.loadtxt([f for f in glob.glob(f"{d}/*_msds.dat") if "solvent" not in os.path.basename(f)][0])
        s = np.loadtxt(glob.glob(f"{d}/*_solvent_msds.dat")[0])
        reps.append((info, m, s))
    n = min(len(m) for _, m, _ in reps)                          # the lags all replicas share
    t = reps[0][1][:n, 0]
    cols = ("self_cation", "self_anion", "cat_cat", "an_an", "cat_an")
    msds = [{k: m[:n, i + 1] for i, k in enumerate(cols)} for _, m, _ in reps]
    info0 = reps[0][0]
    N, qc, qa, T = info0["n_cations"], info0["charge_cation"], info0["charge_anion"], info0["temperature"]
    avg = {k: np.mean([r[k] for r in msds], 0) for k in cols}
    w = tr.best_window(t, tr.total_msd(avg, N, N, qc, qa))
    wd = tr.diffusion_window(t, avg["self_cation"])
    per, Ds = [], []
    for (info, _, s), r in zip(reps, msds):
        L = info["box_length_m"]
        per.append(tr.conductivity_terms({k: tr.slope(t, r[k], w) for k in cols}, N, N, qc, qa, L ** 3, T))
        yh = tr.yeh_hummer(T, ETA, L)
        Ds.append({"Li": tr.slope(t, r["self_cation"], wd) / 6 + yh, "FSI": tr.slope(t, r["self_anion"], wd) / 6 + yh,
                   "DMK": tr.slope(t, s[:n, 1], wd) / 6 + yh})
    k = len(per)
    ms = lambda v: (np.mean(v), np.std(v, ddof=1) / np.sqrt(k))
    return ({x: ms([p[x] for p in per]) for x in TERMS}, {x: ms([q[x] for q in Ds]) for x in ("Li", "FSI", "DMK")}, k)


def mlff(n_blocks=4):
    import MDAnalysis as mda
    d = data("2_machine_learning_ff", "trajectory")
    seg = json.load(open(f"{d}/segments.json"))
    xs = []
    for f, a, b in seg["segments"]:
        u = mda.Universe(f"{d}/{seg['topology']}", f"{d}/{f}")
        xs.append(u.trajectory.timeseries(order="fac")[a:b])
    x = np.concatenate(xs).astype(float)[int(seg["skip_ps"] / seg["dt_ps"]):]
    names = mda.Universe(f"{d}/{seg['topology']}").atoms.elements
    M = {"H": 1.008, "Li": 6.94, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998403163, "S": 32.06}   # ASE masses
    m = np.array([M[e.capitalize() if len(e) > 1 else e] for e in names])
    x = tr.unwrap(x, seg["box_A"])
    x -= (np.einsum("a,tad->td", m, x) / m.sum())[:, None, :]
    fsi_m = np.array([14.007, 32.065, 15.999, 15.999, 32.065, 15.999, 15.999, 18.998, 18.998])
    dmk_m = np.array([12.011, 12.011, 15.999, 12.011, 1.008, 1.008, 1.008, 1.008, 1.008, 1.008])
    com = lambda blk, mm, k: np.stack([np.einsum("taj,a->tj", blk[:, i * k:(i + 1) * k], mm) / mm.sum()
                                       for i in range(blk.shape[1] // k)], 1)
    species = {"Li": x[:, 540:546], "FSI": com(x[:, 546:600], fsi_m, 9), "DMK": com(x[:, :540], dmk_m, 10)}
    yh = tr.yeh_hummer(298.0, ETA, seg["box_A"] * 1e-10)

    def D_of(sp_x):
        t = np.arange(len(sp_x)) * seg["dt_ps"] * 1e-12
        return tr.diffusion_single(t, tr.msd_fft(sp_x * 1e-10))[0] + yh
    whole = {k: D_of(v) for k, v in species.items()}
    blocks = {k: [D_of(b) for b in np.array_split(v, n_blocks)] for k, v in species.items()}
    return {k: (whole[k], np.std(blocks[k], ddof=1) / np.sqrt(n_blocks)) for k in species}


if __name__ == "__main__":
    print("OPLS-ECC conductivity (mS/cm), every published condition")
    print(f"  {'condition':30s} {'this code':>12s} {'published':>12s}  n")
    t1 = None
    for cond, pub in PUBLISHED.items():
        mean, sem, n, D, D_dmk = opls(cond)
        print(f"  {cond:30s} {fmt(mean['total'], sem['total']):>12s} {pub:>12s}  {n}")
        if cond == "charge_sweep/q_0.725":
            t1 = (mean, sem, D, D_dmk)
    mean, sem, D, D_dmk = t1
    print("\nTable 1  (terms: sigma+self sigma-self sigma++dist sigma--dist sigma+-cross sigma; D: Li+ FSI- DMK)")
    print("  OPLS-ECC   ", " ".join(fmt(mean[k], sem[k]) for k in TERMS), "|",
          fmt(D["Li"][0] * 1e9, D["Li"][1] * 1e9, d=2), fmt(D["FSI"][0] * 1e9, D["FSI"][1] * 1e9, d=2), f"{D_dmk * 1e9:.2f}",
          "  (DMK: central value only, the per-replica solvent MSDs were not stored)")
    print("  published  ", TABLE1["OPLS-ECC"][0], "|", TABLE1["OPLS-ECC"][1])
    s, Dn, k = byteff()
    print("  ByteFF-Pol ", " ".join(fmt(*s[x]) for x in TERMS), "|",
          " ".join(fmt(Dn[x][0] * 1e9, Dn[x][1] * 1e9, 2) for x in ("Li", "FSI", "DMK")), f"  ({k} replicas)")
    print("  published  ", TABLE1["ByteFF-Pol"][0], "|", TABLE1["ByteFF-Pol"][1])
    if "--skip-mlff" not in sys.argv:
        Dm = mlff()
        print("  MACE-FT    ", "D only |", " ".join(fmt(Dm[x][0] * 1e9, Dm[x][1] * 1e9, 2) for x in ("Li", "FSI", "DMK")),
              " (SE over 4 blocks)")
        print("  published  ", "D only |", TABLE1["MACE-FT"][1])
