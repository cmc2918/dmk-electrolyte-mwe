#!/usr/bin/env python3
"""
ByteFF-Pol production run: 1.5M LiFSI in DMK (acetone)

System composition:
    34 LiFSI (Li⁺ + FSI⁻) + 320 DMK (acetone, (CH₃)₂CO)
    = 3,540 atoms (34×1 Li + 34×9 FSI + 320×10 DMK)
    Concentration: 1.5 mol/L LiFSI in DMK

Force field:
    ByteFF-Pol — polarisable force field parameterised by a GNN trained on
    ωB97M-V/def2-TZVPD quantum chemistry data. Uses AMOEBA-style induced
    dipoles for polarisation with Thole damping (a=0.39). Nonbonded
    interactions use Buckingham exp-6 potential (not Lennard-Jones).

Simulation protocol:
    1. NPT equilibration (3 ns, 1,500,000 steps × 2 fs)
       - Langevin thermostat at 298 K, friction 0.1/ps
       - Monte Carlo barostat at 1 atm, volume moves every 12 steps
       - Energy minimisation before dynamics
       - Purpose: equilibrate density

    2. Box rescaling
       - Rescales box to the average density from the last 1 ns of NPT
       - Ensures NVT production starts at the correct equilibrium density

    3. NVT production (50 ns, 25,000,000 steps × 2 fs)
       - Langevin thermostat at 298 K, friction 0.1/ps
       - Fixed volume (from NPT average)
       - Purpose: production trajectory for transport property analysis
         (Li-FSI residence times, diffusion, conductivity)

Integrator:
    MTSLangevinIntegrator (BAOAB-RESPA multiple timestep):
      - Outer timestep: 2 fs (long-range electrostatics + polarisation)
      - Inner timestep: 1 fs (bonded forces, evaluated every 2 sub-steps)

Compute:
    OpenMM on GPU (CUDA platform, mixed precision).

Output:
    production_results/
      npt_state.csv  — NPT thermo every 1 ps
      npt.dcd        — NPT trajectory every 1 ps
      nvt_state.csv  — NVT thermo every 1 ps
      nvt.dcd        — NVT production trajectory every 1 ps

Usage:
    source /path/to/miniconda3/etc/profile.d/conda.sh && conda activate byteff2
    export OPENMM_DIR=/path/to/byteff2/openmm_install/openmm
    export LD_LIBRARY_PATH=$OPENMM_DIR/lib:$LD_LIBRARY_PATH
    cd /path/to/byteff2 && export PYTHONPATH=$(pwd):$PYTHONPATH
    cd /path/to/run_dir     # holds transport_config.json and transport_params/
    python3 run_production.py 2>&1 | tee production.log
"""

import json
import logging
import os

import openmm.app as app

from byteff2.toolkit.openmmtool import generate_openmm_system
from byteff2.toolkit.protocol import TransportProtocol
from byteff2.md_utils.md_run import npt_run, nvt_run, rescale_box

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────
CONFIG_FILE = "transport_config.json"
OUTPUT_DIR = "production_results"
TEMPERATURE = 298  # K

# Simulation lengths (steps at 2 fs outer timestep)
NPT_STEPS = 1_500_000    # 3 ns  NPT equilibration
NVT_STEPS = 25_000_000   # 50 ns NVT production


# ── Main ──────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(CONFIG_FILE) as f:
        config = json.load(f)

    params_dir = config['params_dir']

    # Step 1: Generate force field parameters
    logger.info("Generating force field parameters")
    tp = TransportProtocol(config)
    nonbonded_params = tp.generate_ff_params(config['smiles'])

    # Step 2: Build simulation box with GROMACS
    gro_file = os.path.join(params_dir, 'solvent_salt.gro')
    top_file = os.path.join(params_dir, 'system.top')

    if not os.path.exists(gro_file):
        logger.info("Building system with GROMACS")
        tp.build_system(
            config['natoms'],
            config['components'],
            config.get('working_dir', 'working_dir'),
        )
    else:
        logger.info(f"Using existing system: {gro_file}")

    # Step 3: Create OpenMM system with AMOEBA polarisation
    logger.info("Creating polarisable OpenMM system")
    gro = app.GromacsGroFile(gro_file)
    positions = gro.positions
    unit_cell = gro.getUnitCellDimensions()

    input_top, input_system = generate_openmm_system(
        top_file, nonbonded_params, unit_cell
    )

    natoms = input_system.getNumParticles()
    logger.info(f"System: {natoms} atoms")

    # Step 4: NPT equilibration
    logger.info(f"NPT equilibration: {NPT_STEPS} steps = {NPT_STEPS * 2 / 1e6:.1f} ns")
    npt_positions, npt_box_vec = npt_run(
        input_top, input_system, positions,
        npt_steps=NPT_STEPS,
        temperature=TEMPERATURE,
        work_dir=OUTPUT_DIR,
    )

    # Step 5: Rescale box to average density
    logger.info("Rescaling box to average NPT density")
    rescale_positions, rescale_box_vec = rescale_box(
        npt_positions, npt_box_vec, work_dir=OUTPUT_DIR
    )

    # Step 6: NVT production
    logger.info(f"NVT production: {NVT_STEPS} steps = {NVT_STEPS * 2 / 1e6:.1f} ns")
    nvt_positions, nvt_box_vec = nvt_run(
        input_top, input_system,
        rescale_positions, rescale_box_vec,
        temperature=TEMPERATURE,
        work_dir=OUTPUT_DIR,
        nvt_steps=NVT_STEPS,
    )

    logger.info("Done!")
    logger.info(f"Outputs in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
