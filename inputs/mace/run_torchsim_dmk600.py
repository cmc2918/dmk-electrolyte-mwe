#!/usr/bin/env python3
"""
Control experiment: 600-atom 1.5 M LiFSI/DMK in TorchSim, started from the SAME r2SCAN AIMD snapshot
as the 4 ns ASE/NVE run (mace_training/test.xyz frame 0, DMK_run_1), with the production settings
(NVT v-rescale 298 K, tau 0.1 ps, 1 fs, float32 + cuEq) and NO minimisation.

Question: do the FSI decomposition events seen in the 1000-atom Packmol run still occur when the start
is an equilibrated, in-distribution configuration?  Success metric: number of Li-F(FSI) encounters
below 2.0 A that the anion survives (the ASE run with the combined model survived 13 in 4 ns).

A geometric watchdog runs every WATCH_EVERY steps: it records the closest Li-F(FSI) distance and the
longest S-F, N-S and X-H bonds in the HDF5, and aborts the run as soon as a covalent bond is broken.

    python run_torchsim_dmk600.py                       # DMK-only model, 2 ns
    python run_torchsim_dmk600.py --model <combined>    # swap the model, everything else identical
    python run_torchsim_dmk600.py --smoke --device cpu  # code-path test (also tests the abort)
"""
import argparse, os, sys, time, warnings
from pathlib import Path
warnings.simplefilter("ignore"); os.environ["PYTHONWARNINGS"] = "ignore"
_BLOCK = (".grad attribute of a Tensor that is not a leaf", "if t.grad is not None:", "warp/_src/torch.py", "TensorBody.h")
class _Filtered:
    def __init__(self, real): self.real, self._p = real, ""
    def write(self, msg):
        self._p += msg
        if "\n" not in self._p: return
        done, self._p = self._p.rsplit("\n", 1)
        keep = [l for l in done.split("\n") if not any(b in l for b in _BLOCK)]
        if keep: self.real.write("\n".join(keep) + "\n")
    def flush(self):
        if self._p and not any(b in self._p for b in _BLOCK): self.real.write(self._p); self._p = ""
        self.real.flush()
    def __getattr__(self, n): return getattr(self.real, n)
sys.stderr = _Filtered(sys.stderr)
print = lambda *a, **k: __builtins__.print(*a, **{**k, "flush": True})  # noqa: E731

import numpy as np, torch
from ase.io import read, write as ase_write
import MDAnalysis as mda
import torch_sim as ts
from torch_sim.models.mace import MaceModel

HERE = Path(__file__).resolve().parent
ROOT = os.environ.get("DMK_ROOT", str(HERE))                         # folder holding mace_training/ and mace_training_DMK/
START_XYZ, START_INDEX = f"{ROOT}/mace_training/test.xyz", 0          # the frame the ASE run started from
MODEL_DEFAULT = f"{ROOT}/mace_training_DMK/LiFSI_DMK_mh0.model"
TIMESTEP_PS, TEMPERATURE_K, TAU_PS, RNG_SEED = 0.001, 298.0, 0.1, 42
STATE_EVERY, WATCH_EVERY = 500, 500                                   # 0.5 ps

class BondBroken(RuntimeError): pass

def build_bonds(atoms):
    """Covalent bonds from the start frame: 54 DMK (10 atoms), 6 Li, 6 FSI (9 atoms), minimum image."""
    sym = np.array(atoms.get_chemical_symbols()); x = atoms.get_positions(); L = atoms.cell.lengths()[0]
    blocks = [np.arange(10*m, 10*m + 10) for m in range(54)] + [np.arange(546 + 9*k, 555 + 9*k) for k in range(6)]
    out = {"S-F": [], "N-S": [], "X-H": [], "other": []}
    for a in blocks:
        d = x[a][:, None] - x[a][None]; d -= L * np.round(d / L); r = np.linalg.norm(d, axis=-1)
        for p in range(len(a)):
            for q in range(p + 1, len(a)):
                s = {sym[a[p]], sym[a[q]]}
                if s == {"H"}: continue
                if r[p, q] < (1.35 if "H" in s else 1.95):
                    key = "S-F" if s == {"S", "F"} else "N-S" if s == {"N", "S"} else "X-H" if "H" in s else "other"
                    out[key].append((a[p], a[q]))
    return {k: np.array(v) for k, v in out.items()}, L

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=MODEL_DEFAULT); p.add_argument("--ns", type=float, default=2.0)
    p.add_argument("--tag", default="torchsim_dmk600"); p.add_argument("--device", default="cuda")
    p.add_argument("--equil-ps", type=float, default=100.0, help="initial part to discard in the analysis (recorded in the metadata)")
    p.add_argument("--sf-limit", type=float, default=2.3, help="abort if any S-F or N-S bond exceeds this (A)")
    p.add_argument("--xh-limit", type=float, default=1.6, help="abort if any X-H bond exceeds this (A)")
    p.add_argument("--smoke", action="store_true", help="20 steps, watchdog every 5, then a forced abort test")
    p.add_argument("--start-h5", default=None, help="restart: take the start positions from this TorchSim trajectory")
    p.add_argument("--start-ps", type=float, default=None, help="restart: simulated time (ps) of the frame to start from")
    p.add_argument("--seed", type=int, default=RNG_SEED, help="seed of the Maxwell-Boltzmann velocities and the thermostat")
    a = p.parse_args()
    out = HERE / ("smoke_test" if a.smoke else "."); out.mkdir(exist_ok=True)
    n_steps = 20 if a.smoke else int(round(a.ns * 1000.0 / TIMESTEP_PS))
    state_every, watch_every = (10, 5) if a.smoke else (STATE_EVERY, WATCH_EVERY)
    if not Path(a.model).is_file(): sys.exit(f"Model not found: {a.model}")

    atoms = read(START_XYZ, index=START_INDEX); atoms.set_pbc(True)
    assert len(atoms) == 600 and atoms.info.get("config_type", "").startswith("DMK"), "unexpected start frame"
    start_label = f"{START_XYZ}[{START_INDEX}]"
    if a.start_h5:
        # restart: positions and cell of a stored frame, new Maxwell-Boltzmann velocities (no momenta carried over)
        ref_symbols = atoms.get_chemical_symbols()
        tr0 = ts.TorchSimTrajectory(a.start_h5, mode="r")
        steps = np.ravel(tr0.get_steps("positions")); target = int(round(a.start_ps / TIMESTEP_PS))
        fr = int(np.argmin(np.abs(steps - target))); atoms = tr0.get_atoms(frame=fr); tr0.close()
        assert atoms.get_chemical_symbols() == ref_symbols, "atom order of the restart frame differs from the AIMD start frame"
        for k in ("momenta", "velocities"):
            if k in atoms.arrays: del atoms.arrays[k]
        atoms.set_pbc(True); atoms.wrap()
        start_label = f"{a.start_h5} frame {fr} (step {int(steps[fr])}, {steps[fr]*TIMESTEP_PS:.1f} ps), new velocities, seed {a.seed}"
    bonds, L = build_bonds(atoms)
    print(f"=== 600-atom run from {start_label}, L = {L:.4f} A ===")
    print(f"Model: {a.model}\nSteps: {n_steps:,} ({n_steps*TIMESTEP_PS/1000:.3f} ns), NVT v-rescale {TEMPERATURE_K} K, tau {TAU_PS} ps, device {a.device}")
    print("Bonds monitored: " + ", ".join(f"{k} {len(v)}" for k, v in bonds.items()) + "  (expected S-F 12, N-S 12, X-H 324, other 186)")

    device = torch.device(a.device)
    raw = torch.load(a.model, map_location="cpu", weights_only=False)
    model = MaceModel(model=raw, device=device, dtype=torch.float32, compute_forces=True, compute_stress=False,
                      enable_cueq=(a.device == "cuda"))
    li = torch.arange(540, 546, device=device)
    fF = torch.tensor([546 + 9*k + j for k in range(6) for j in (7, 8)], device=device)
    tb = {k: torch.tensor(v, device=device) for k, v in bonds.items()}
    limits = {"S-F": a.sf_limit, "N-S": a.sf_limit, "X-H": a.xh_limit, "other": 2.3}
    def mic(d): return d - L * torch.round(d / L)
    def blen(x, b): return torch.linalg.norm(mic(x[b[:, 0]] - x[b[:, 1]]), dim=-1)
    nwatch = {"n": 0}
    def make_watch(key):
        def fn(state, model=None):
            with torch.no_grad():
                x = state.positions.detach()
                if key == "min_LiF":
                    return torch.linalg.norm(mic(x[fF][:, None] - x[li][None]), dim=-1).min().reshape(1)
                r = blen(x, tb[key]); m = r.max()
                lim = limits[key] if not (a.smoke and nwatch["n"] >= 3 and key == "S-F") else 1.0   # forced abort in smoke mode
                if m > lim:
                    i, j = tb[key][int(r.argmax())].tolist()
                    raise BondBroken(f"{key} bond between atoms {i} and {j} reached {m.item():.2f} A (limit {lim} A)")
                if key == "S-F": nwatch["n"] += 1
                return m.reshape(1)
        return fn
    def T(state, model=None):
        with torch.no_grad(): return ts.calc_temperature(masses=state.masses, momenta=state.momenta.detach(), system_idx=state.system_idx)
    def KE(state, model=None):
        with torch.no_grad(): return ts.calc_kinetic_energy(masses=state.masses, momenta=state.momenta.detach(), system_idx=state.system_idx)
    def PE(state, model=None):
        with torch.no_grad(): return state.energy.detach()

    h5 = out / f"{a.tag}.h5"
    if h5.exists(): h5.unlink()
    reporter = ts.TrajectoryReporter(
        filenames=str(h5), state_frequency=state_every,
        prop_calculators={watch_every: {"temperature_K": T, "kinetic_energy_eV": KE, "potential_energy_eV": PE,
                                        "min_LiF_A": make_watch("min_LiF"), "max_SF_A": make_watch("S-F"),
                                        "max_NS_A": make_watch("N-S"), "max_XH_A": make_watch("X-H"), "max_other_A": make_watch("other")}},
        metadata={"system": "1.5 M LiFSI/DMK, 6 LiFSI + 54 DMK (600 atoms), AIMD snapshot start, no minimisation",
                  "start": start_label, "model": a.model, "ensemble": "NVT v-rescale",
                  "temperature_K": TEMPERATURE_K, "thermostat_tau_ps": TAU_PS, "timestep_ps": TIMESTEP_PS,
                  "n_steps": n_steps, "equil_ps": a.equil_ps, "n_equil_steps": int(round(a.equil_ps / TIMESTEP_PS)),
                  "note": "discard the first equil_ps in every analysis", "rng_seed": a.seed, "sf_limit_A": a.sf_limit, "xh_limit_A": a.xh_limit})
    state = ts.initialize_state(atoms, device=device, dtype=torch.float32); state.rng = a.seed
    tau = TAU_PS * ts.units.MetalUnits.time
    print("\nWarm-up…"); t0 = time.time()
    w = ts.initialize_state(atoms, device=device, dtype=torch.float32); w.rng = a.seed
    ts.integrate(system=w, model=model, integrator=ts.Integrator.nvt_vrescale, n_steps=5 if a.smoke else 100,
                 temperature=TEMPERATURE_K, timestep=TIMESTEP_PS, tau=tau)
    print(f"  {time.time()-t0:.1f}s\nRunning…"); t0 = time.time(); status = "completed"
    try:
        ts.integrate(system=state, model=model, integrator=ts.Integrator.nvt_vrescale, n_steps=n_steps,
                     temperature=TEMPERATURE_K, timestep=TIMESTEP_PS, tau=tau, trajectory_reporter=reporter,
                     pbar=False if a.smoke else {"desc": a.tag, "smoothing": 0.0,
                           "bar_format": "{desc}: {n_fmt}/{total_fmt} steps | elapsed {elapsed} | ETA {remaining} | {rate_fmt}"})
    except BondBroken as e:
        status = f"ABORTED by watchdog: {e}"
    finally:
        try: reporter.close()
        except Exception: pass
    el = time.time() - t0
    print(f"\n=== {status} ===\n  wall time {el/3600:.2f} h")
    # summary of contacts survived + conversion
    tr = ts.TorchSimTrajectory(str(h5), mode="r")
    try:
        st = np.ravel(tr.get_steps("min_LiF_A")); d = np.ravel(tr.get_array("min_LiF_A")); sf = np.ravel(tr.get_array("max_SF_A"))
        n = min(len(st), len(d), len(sf)); st, d, sf = st[:n], d[:n], sf[:n]; close = np.where(d < 2.0)[0]
        ep = 0 if len(close) == 0 else 1 + int((np.diff(st[close]) > 5000).sum())
        print(f"  simulated {st[-1]*TIMESTEP_PS:.1f} ps | closest Li-F(FSI) {d.min():.2f} A | encounters below 2.0 A: {ep} episodes ({len(close)} samples) | longest S-F seen {sf.max():.2f} A")
        nfr = len(tr.get_steps("positions")); first = tr.get_atoms(frame=0); pdb, dcd = out / f"{a.tag}.pdb", out / f"{a.tag}.dcd"
        ase_write(str(pdb), first); u = mda.Universe(str(pdb))
        with mda.Writer(str(dcd), n_atoms=u.atoms.n_atoms) as W:
            for i in range(nfr):
                at = tr.get_atoms(frame=i); u.atoms.positions = at.get_positions(); c = at.cell.lengths()
                u.dimensions = np.array([c[0], c[1], c[2], 90.0, 90.0, 90.0]); W.write(u.atoms)
        print(f"  {nfr} frames -> {dcd.name}")
    finally:
        tr.close()
    sys.exit(0 if status == "completed" else 2)

if __name__ == "__main__":
    main()
