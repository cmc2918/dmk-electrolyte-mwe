# Simulation inputs

The simulations themselves were run with open codes. Their inputs are listed below, stored either in the data
deposit (one per run) or here, where a driver script was used in place of a plain input file. The driver scripts are
the versions that produced the published trajectories. Only their local paths have been replaced:
- `finetune_mace.sh` runs in its own folder, next to `train.xyz`, `val.xyz` and `test.xyz`, and takes the conda base
  and environment from `conda`;
- `run_chain.py` expects the model `LiFSI_FAN_DMK_mh0.model` in its own folder;
- `run_torchsim_dmk600.py` reads its start frame from `$DMK_ROOT/mace_training/test.xyz` (default: its own folder).
  Production runs pass the model with `--model`;
- the ByteFF-Pol scripts run in a folder holding `transport_config.json` and `transport_params/`.

| Model | Code | Input | Where |
|---|---|---|---|
| r²SCAN AIMD, 3 runs | CP2K 2025.2 | `cp2k_production.inp`, `initial_structure_<i>.xyz` | deposit `1_ab_initio_md/run_<i>/` |
| r²SCAN single points (charges, Wannier centres) | CP2K 2025.2 | `sp_template.inp`, `snapshots/<name>.xyz`, cell in `manifest.csv` | deposit `1_ab_initio_md/single_points/` |
| MACE-FT fine-tuning | MACE (`mace_run_train`) | `mace/finetune_mace.sh` (MACE-MH-0 foundation model, `matpes_r2scan` head) | here |
| MACE-FT MD | TorchSim | `mace/run_torchsim_dmk600.py` (one segment, bond check), `mace/run_chain.py` (segments and 100 ps rewinds after an S–F break) | here; model in deposit `2_machine_learning_ff/model/` |
| ByteFF-Pol MD | OpenMM | `byteff_pol/run_production.py` (replica 1), `byteff_pol/run_replica.py` (replicas 2–10), `byteff_pol/restart_nvt.py`, `byteff_pol/transport_config.json` | here; parameters and topology in deposit `3_polarizable_ff/parameters/` |
| OPLS-ECC MD, every sweep point | LAMMPS | `equil_script.in`, `prod_first.in`, `prod_continue.in`, `1.5M_LiFSI_conf1.data` | deposit `4_nonpolarizable_ff/<sweep>/<point>/inputs/` |
| OPLS-ECC shear viscosity | LAMMPS | `visc_fionn.in` | deposit `4_nonpolarizable_ff/viscosity/` |
| Raman clusters | ORCA 6.1.1 | `<cluster>.inp`, `<cluster>.xyz` | deposit `5_raman/clusters_cpcm/`, `5_raman/clusters_gas_phase/` |

## ByteFF-Pol system

The simulated box is the one in the deposit, `parameters/solvent_salt.gro` and `parameters/system.top`: 35 LiFSI and
329 DMK (3640 atoms). The 34/320 composition in `transport_config.json` and in the docstring of `run_production.py` is
only a target, used to build a new box when `solvent_salt.gro` is absent.

## MACE-FT production commands

The 35 ns trajectory is one continuous path made of the segments listed in
`2_machine_learning_ff/trajectory/segments.json`:

```bash
# first segment, from the AIMD snapshot (1 ns discarded as equilibration)
python3 run_torchsim_dmk600.py --model LiFSI_FAN_DMK_mh0.model --ns 10 --tag torchsim_dmk600_combined_10ns --equil-ps 1000
# restart 100 ps before the first S-F break, with new Maxwell-Boltzmann velocities
python3 run_torchsim_dmk600.py --model LiFSI_FAN_DMK_mh0.model --start-h5 torchsim_dmk600_combined_10ns.h5 \
        --start-ps 5529.5 --seed 2026 --ns 9.5 --equil-ps 10 --tag torchsim_dmk600_combined_restart
# the remaining segments chain_s01 ... chain_s06, rewinding automatically after each break
python3 run_chain.py --hours 336
```

## Variants of the OPLS-ECC model

All sweep points start from the same OPLS-AA data file. The variant is set in the LAMMPS scripts:

- `charge_sweep/q_<q>`: ionic charges ±q, through `set type 5-9 charge` in `equil_script.in`. The production scripts
  read them back from `equilibrated_box.data`.
- `lj_cross_sweep/dsigma_+<Δσ>`: q = 0.725, with the Li⁺–O(sulfonyl) cross term overridden by
  `pair_coeff 5 8 0.01 <2.915 + Δσ>` in all three scripts.
- `solvent_charge_sweep/f_<f>`: q = 0.725, with the DMK charges × f through `set type 1-4 charge` in `equil_script.in`.
