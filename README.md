# dmk-electrolyte-mwe

[![DOI](https://helix.imperial.ac.uk/badge/DOI/10.82186/23kt9-tkq05.svg)](https://doi.org/10.82186/23kt9-tkq05)

Minimal working examples for

> C. Corral-Casas, J. Alary, J. P. Ewen, D. Dini, *From Electron Density to Ionic Conductivity for the Mechanistic
> Understanding of a Fast Charging Acetone Electrolyte*, submitted to npj Computational Materials (2026).

The analyses of the paper were run with research scripts written for the authors' file system. This repository
reimplements each analysis as a short, documented function, using the same definitions, cutoffs, windows and
fitting choices as those scripts. Each example applies the functions to the published dataset and prints its results
next to the published values.

## Data

The examples read the dataset deposited at Imperial College London's Helix repository
(https://doi.org/10.82186/23kt9-tkq05, CC BY 4.0). Download the archives, unzip them all in one folder, and point the
examples to it:

```bash
export DMK_DATA=/path/to/helix_data
```

## Installation

```bash
conda env create -f environment.yml
conda activate dmk-mwe
```

The modules need NumPy, SciPy and pandas. MDAnalysis reads the MACE-FT DCD trajectory, and pyarrow reads the
SolvationAnalysis contact lists.

## Examples

| Script | Reproduces | Inputs in the dataset | Runtime |
|---|---|---|---|
| `examples/01_transport.py` | Table 1 (σ, Onsager terms, D for OPLS-ECC, ByteFF-Pol, MACE-FT); σ(SEM) of every point of Tables 3–5 | per-replica MSDs; MACE-FT trajectory | 2 min |
| `examples/02_solvation_structure.py` | Table 2 (r_peak, r_min, CN for all models and AIMD runs); SSIP/CIP/AGG, CN and O/FSI of Tables 3–5; Fig. 3b | RDFs and contact lists; MACE-FT and AIMD trajectories | 2 min |
| `examples/03_shell_dynamics.py` | Transport Index; residence times τ; SI Table S7 (exchange events, τ_ex); ΔG of the five-ligand shell; MACE-FT exchange counts of Fig. 5c | contact tables; MACE-FT trajectory | 40 min (`--quick`: 7 min) |
| `examples/04_raman.py` | r per cluster class, band shift, double ratio, gas-phase check, static polarizability; N for each model's CN (Fig. 3c); the three decompositions of the measured band (SI) | ORCA `.hess`/`.out` files; digitised spectrum | < 1 min |
| `examples/05_electronic_structure.py` | REPEAT donor-O charges vs distance (Fig. 9b, c); Coulomb ratio DMK/FSI⁻ for three charge sets (Fig. 9d); DMK dipoles from Wannier centres, REPEAT, OPLS-AA and ByteFF-Pol (Fig. 10a) | r²SCAN single points; ByteFF-Pol induced dipoles | < 1 min |

```bash
python3 examples/01_transport.py              # --skip-mlff omits the MACE-FT trajectory
python3 examples/02_solvation_structure.py    # --skip-trajectories omits MACE-FT and AIMD
python3 examples/03_shell_dynamics.py         # --quick omits the OPLS-ECC residence times and the MACE-FT counts
python3 examples/04_raman.py
python3 examples/05_electronic_structure.py
```

## Package

| Module | Analysis | Paper |
|---|---|---|
| `dmk_mwe/transport.py` | FFT MSDs, barycentric frame, Onsager decomposition, fitting windows, Yeh–Hummer correction, Fickian window | Methods, ionic conductivity; SI S2 |
| `dmk_mwe/solvation.py` | RDFs, first peak and minimum, running CN, anion-centric speciation | Methods, solvation structure |
| `dmk_mwe/shells.py` | bound indicator and its autocorrelation, hysteresis shells, Transport Index, exchange events, contact tables from coordinates | Methods, residence times and transport mechanisms; SI S6, S7 |
| `dmk_mwe/raman.py` | ORCA Raman activities to intensities, bound/free ratio, band decomposition | Methods, Raman cross-section calculations; SI S4 |
| `dmk_mwe/electronic.py` | REPEAT charges, ligand Coulomb energies, point-charge dipoles | Methods, electronic structure of the solvation shell |

Simulation inputs and drivers are described in [`inputs/README.md`](inputs/README.md).

## Agreement with the published values

All examples reproduce the published values to the quoted digits, except for these rounding differences in the last
digit:

- Table 2: OPLS-ECC CN(sulfonyl) 0.855 (published 0.86) and the SD of its carbonyl r_min, 0.07 (0.06); ByteFF-Pol
  carbonyl r_min 3.15 (3.16).
- Tables 3 and 4: CN(sulfonyl) 0.855 at q = 0.725 (0.86); CN(carbonyl) 4.56 at Δσ = +0.15 Å (4.57); O/FSI 0.01 lower
  at Δσ = +0.10, +0.15 and +0.25 Å.
- Figs. 9 and 10: REPEAT charges of a free carbonyl and sulfonyl O, −0.565 and −0.514 e averaged beyond 5 Å (the
  figure's bulk bins give −0.566 and −0.512 e); REPEAT dipole of a free DMK 3.695 D (3.70); ByteFF-Pol dipole gain
  +28.45 % (28.5 %).

## Tests

`pytest tests` checks the core functions on synthetic data and needs no download.

## Licence and citation

MIT (see `LICENSE`). If you use this code, please cite the paper and the archived release (https://doi.org/10.5281/zenodo.XXXXXXX).
