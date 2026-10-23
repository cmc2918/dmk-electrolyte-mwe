#!/usr/bin/env python3
"""Dynamics of the Li+ shell (Fig. 5, SI Sections S6-S7): Transport Index, residence times, ligand-exchange events,
the free energy of the five-ligand shell, and the MACE-FT exchange counts of Fig. 5c.

Inputs: the contact tables of the deposit (4_nonpolarizable_ff/contact_lists, 3_polarizable_ff/analysis/contact_lists,
2_machine_learning_ff/analysis), the ByteFF-Pol contact lists of 3_polarizable_ff/analysis/solvation, and the first
300 ps of the MACE-FT trajectory. --quick skips the OPLS-ECC residence times (~3 min per replica) and the MACE-FT counts.

    DMK_DATA=/path/to/helix_data python3 examples/03_shell_dynamics.py [--quick]
"""
import glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dmk_mwe import data
from dmk_mwe import shells as sh
from dmk_mwe import solvation as sv

LI, CARB = np.arange(540, 546), np.arange(54) * 10 + 2
SULF = [546 + 9 * k + np.array([2, 3, 5, 6]) for k in range(6)]
ms = lambda v: (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v))))


def opls_tables():
    return sorted(glob.glob(data("4_nonpolarizable_ff", "contact_lists", "opls_rep*.npz")))


def byteff_tables():
    return sorted(glob.glob(data("3_polarizable_ff", "analysis", "contact_lists", "byteff_rep*.npz")))


def mlff_table():
    return glob.glob(data("2_machine_learning_ff", "analysis", "mlff_stitched.npz"))[0]


def stride_to(z, dt_ps):
    return int(round(dt_ps / float(z["dt"])))


def residence_from_parquet(folder, r_min):
    """tau (ps) per replica from SolvationAnalysis contact lists (one row per frame, Li+, ligand), H(t) = d < r_min."""
    import pandas as pd
    out = {"DMK": [], "FSI": []}
    for d in sorted(glob.glob(f"{folder}/rep*/")):
        n = os.path.basename(d.rstrip("/"))[3:]
        js = json.load(open(f"{d}solvation_results_rep{n}.json"))
        df = pd.read_parquet(f"{d}solvation_data_rep{n}.parquet", columns=["frame", "solute_ix", "solvent_ix", "distance", "solvent"])
        frames = np.sort(df["frame"].unique())
        step = int(np.median(np.diff(frames)))
        dt = js.get("dt_ps", None)
        for s in ("DMK", "FSI"):
            x = df[(df["solvent"] == s) & (df["distance"] < r_min[s])].drop_duplicates(["frame", "solute_ix", "solvent_ix"])
            f = (x["frame"].values - frames[0]) // step
            key = x["solute_ix"].values.astype(np.int64) * 1_000_000 + x["solvent_ix"].values
            C = sh.acf(key, f, (frames[-1] - frames[0]) // step + 1)
            out[s].append((C, step))
    return out


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    # ---------------------------------------------------------------- Transport Index, 5 ps steps
    zs_o = [np.load(f) for f in opls_tables()]
    zs_b = [np.load(f) for f in byteff_tables()]
    zm = np.load(mlff_table())
    # OPLS-ECC shells: each replica's own r_in / r_out (first minimum of its RDF, from the SolvationAnalysis run),
    # stored with the contact tables; ByteFF-Pol and MACE-FT: r_out = 3.4 / 3.2 A, r_in from their RDFs
    own = json.load(open(data("4_nonpolarizable_ff", "contact_lists", "results.json")))["runs"]["aligned"]["cutoffs"]
    own = own["OPLS-ECC_per_replica"]
    cut_o = [{sp: {k: own[os.path.basename(f)[:-4]][sp][k] for k in ("r_in", "r_out")} for sp in ("DMK", "FSI")}
             for f in opls_tables()]
    ti_o = [sh.transport_index(z, stride_to(z, 5.0), c) for z, c in zip(zs_o, cut_o)]
    ti_b = [sh.transport_index(z, stride_to(z, 5.0)) for z in zs_b]
    ti_m = sh.transport_index(zm, stride_to(zm, 5.0))
    print("Transport Index (5 ps steps; published OPLS-ECC 0.41, ByteFF-Pol 0.19, MACE-FT 0.07)")
    print(f"  OPLS-ECC {np.mean(ti_o):.3f} ({len(ti_o)} replicas)  ByteFF-Pol {np.mean(ti_b):.3f} ({len(ti_b)})  MACE-FT {ti_m:.3f}")

    # ---------------------------------------------------------------- residence times
    print("\nResidence times tau (ps), H(t) = donor O within the first RDF minimum")
    # MACE-FT: contact table every 5 ps, r_min of its own RDF, standard error over four contiguous blocks
    r_min = {s: sv.first_peak_and_minimum(zm["rdf_r"], zm[f"rdf_{s.lower()}"])[1] for s in sh.SP}
    st = stride_to(zm, 5.0)
    tau_m, blocks = {}, {}
    for s in sh.SP:
        key, fr, li, _ = sh.single_cutoff_bound(sh.rows(zm, s, st), r_min[s], int(zm[f"n_{s.lower()}"]))
        T = (int(zm["n_frames"]) - 1) // st + 1
        tau_m[s] = sh.residence_time(sh.acf(key, fr, T), 5.0)
        edges = np.linspace(0, T, 5).astype(int)
        blocks[s] = [sh.residence_time(sh.acf(key[(fr >= a) & (fr < b)], fr[(fr >= a) & (fr < b)] - a, b - a), 5.0)
                     for a, b in zip(edges[:-1], edges[1:])]
    print(f"  MACE-FT    DMK {tau_m['DMK']:.0f} +- {np.std(blocks['DMK'], ddof=1) / 2:.0f}, FSI {tau_m['FSI']:.0f} +- "
          f"{np.std(blocks['FSI'], ddof=1) / 2:.0f}   published 485 +- 21, 143 +- 24   (r_min {r_min['DMK']:.2f} / {r_min['FSI']:.2f} A)")
    # ByteFF-Pol: contact lists of every replica at the first minimum of the replica-mean RDF
    folder = data("3_polarizable_ff", "analysis", "solvation")
    rdf = {s: np.loadtxt(f"{folder}/rdf_cn_{s}_mean.dat").T for s in sh.SP}
    r_min_b = {s: sv.first_peak_and_minimum(rdf[s][0], rdf[s][1])[1] for s in sh.SP}
    acfs = residence_from_parquet(folder, r_min_b)
    dt_b = 1.0                                                           # ps between ByteFF-Pol frames
    tb = {s: [sh.residence_time(C, step * dt_b) for C, step in acfs[s]] for s in sh.SP}
    print(f"  ByteFF-Pol DMK {ms(tb['DMK'])[0]:.1f} +- {ms(tb['DMK'])[1]:.1f}, FSI {ms(tb['FSI'])[0]:.1f} +- {ms(tb['FSI'])[1]:.1f}"
          f"   published 197.4 +- 0.5, 57.0 +- 0.3   ({len(tb['DMK'])} replicas, r_min {r_min_b['DMK']:.2f} / {r_min_b['FSI']:.2f} A)")
    # OPLS-ECC: each replica's 5 ps contact table at its own first minimum (the r_out of its TI shells); ~10 min per
    # replica, skipped with --quick. The published value averages 15 replicas; the 14 deposited ones give the same
    # to the quoted digits.
    if not quick:
        to = {"DMK": [], "FSI": []}
        for z, c in zip(zs_o, cut_o):
            for s in sh.SP:
                key, fr, li, _ = sh.single_cutoff_bound(sh.rows(z, s, 1), c[s]["r_out"], int(z[f"n_{s.lower()}"]))
                to[s].append(sh.residence_time(sh.acf(key, fr, int(z["n_frames"])), float(z["dt"])))
        print(f"  OPLS-ECC   DMK {ms(to['DMK'])[0]:.1f} +- {ms(to['DMK'])[1]:.1f}, FSI {ms(to['FSI'])[0]:.1f} +- "
              f"{ms(to['FSI'])[1]:.1f}   published 158 +- 1, 150 +- 1   ({len(to['DMK'])} replicas)")

    # ---------------------------------------------------------------- exchange events (SI Table S7) and Delta G
    print("\nLigand exchange under the hysteresis shells (SI Table S7)")
    fmt = lambda e: (f"five-ligand {100 * e['five_time']:.1f} %, formed {e['formed_per_ns']:.1f} /ns, success {e['success']:.2f}, "
                     f"exchanges {e['exchanges_per_ns']:.1f} /ns, routes {'/'.join(f'{100 * x:.0f}' for x in e['routes'])}, "
                     f"tau_ex {e['tau_ex_ps']['DMK']:.0f} / {e['tau_ex_ps']['FSI']:.0f} ps")
    eb = [sh.exchanges(z, stride_to(z, 1.0)) for z in zs_b]
    W = sum(e["li_ns"] for e in eb)
    pool = {"five_time": sum(e["five_time"] * e["li_ns"] for e in eb) / W,
            "formed_per_ns": sum(e["over"].get("formed", 0) for e in eb) / W,
            "success": sum(e["over"].get("success", 0) for e in eb) / sum(e["over"].get("success", 0) + e["over"].get("failed", 0) for e in eb),
            "exchanges_per_ns": sum(e["exchanges_per_ns"] * e["li_ns"] for e in eb) / W,
            "routes": tuple(sum(e["routes"][k] * e["exchanges_per_ns"] * e["li_ns"] for e in eb) / sum(e["exchanges_per_ns"] * e["li_ns"] for e in eb) for k in range(3)),
            "tau_ex_ps": {s: 1000 * sum(e["cn"][s] * e["li_ns"] for e in eb) / W /
                          (sum(e["cn"][s] * e["li_ns"] / e["tau_ex_ps"][s] * 1000 for e in eb) / W) for s in sh.SP}}
    print("  ByteFF-Pol, 1 ps  ", fmt(pool))
    print("     published       five-ligand 7.1 %, formed 26.3 /ns, success 0.67, exchanges 32.6 /ns, routes 54/35/11, tau_ex 139 / 40 ps")
    for dt_ps, pub in ((1.0, "1.3 %, 6.5, 0.67, 10.2, 42/44/14, 449 / 109"), (0.5, "1.6 %, 9.9, 0.64, 10.7, 59/30/11, 431 / 104")):
        e = sh.exchanges(zm, stride_to(zm, dt_ps))
        print(f"  MACE-FT, {dt_ps:g} ps   ", fmt(e))
        print(f"     published       {pub}")
        if dt_ps == 1.0:
            dG_m = sh.dG_five(e["P_n"], e["n0"])
    Pb = sum(e["P_n"] * e["li_ns"] for e in eb) / W
    eo = [sh.exchanges(z, 1) for z in zs_o]                      # OPLS-ECC: its 5 ps frames
    Po = sum(e["P_n"] * e["li_ns"] for e in eo) / sum(e["li_ns"] for e in eo)
    print(f"\nFree energy of a fifth ligand, -kB T ln(P5/P4) at 298 K (Fig. 5c; published -1.2, +6.4, +10.7 kJ/mol)")
    print(f"  OPLS-ECC {sh.dG_five(Po):+.1f} (P5 {100 * Po[5]:.0f} %)   ByteFF-Pol {sh.dG_five(Pb):+.1f} (P5 {100 * Pb[5]:.1f} %)"
          f"   MACE-FT {dG_m:+.1f}")
    if quick:
        sys.exit()

    # ---------------------------------------------------------------- MACE-FT exchange counts (Fig. 5c strips)
    print("\nMACE-FT exchange counts, 1 ps frames (Fig. 5c strips)")
    print("  per Li+: DMK entries + exits / 100 ps; FSI- entries + exits per bound FSI- / 100 ps; time in five-ligand shells")

    def strips(z, r_dmk, r_fsi):
        st = stride_to(z, 1.0)
        dmk = sh.entries_exits(z, "DMK", st, r_dmk)[0]
        f_rate, f_cn, _ = sh.entries_exits(z, "FSI", st, r_fsi)
        return dmk, f_rate / f_cn, 100 * sh.exchanges(z, st)["five_time"]
    import MDAnalysis as mda
    d = data("2_machine_learning_ff", "trajectory")
    seg = json.load(open(f"{d}/segments.json"))
    f0, a0, _ = seg["segments"][0]
    X = mda.Universe(f"{d}/{seg['topology']}", f"{d}/{f0}").trajectory.timeseries(order="fac")[:601].astype(float)
    z = sh.contact_table(X, seg["box_A"], LI, CARB, SULF, seg["dt_ps"])
    print("  MACE-FT 0-300 ps %.2f   %.2f   %.2f %%" % strips(z, 3.02, 2.82), "   published 3.83, 4.98, 1.50 %")
    print("  MACE-FT 1-35 ns  %.2f   %.2f   %.2f %%" % strips(zm, 3.02, 2.82), "   published 3.01, 9.81, 1.31 %")
