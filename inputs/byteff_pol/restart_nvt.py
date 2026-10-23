#!/usr/bin/env python3
"""
Restart ByteFF-Pol NVT production from the last frame of the previous run.

Reads final positions/box from the existing NVT DCD, creates a new simulation,
and runs for additional nanoseconds. Trajectory appended to nvt_restart.dcd.

Usage:
    conda activate byteff2
    export OPENMM_DIR=...
    export LD_LIBRARY_PATH=$OPENMM_DIR/lib:$LD_LIBRARY_PATH
    cd /path/to/byteff2 && export PYTHONPATH=$(pwd):$PYTHONPATH
    python restart_nvt.py --ns 50
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import openmm as omm
import openmm.app as app
import openmm.unit as ou
import MDAnalysis as mda

# ByteFF-Pol imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from byteff2.toolkit.protocol import TransportProtocol
from byteff2.md_utils.md_run import nvt_run
from byteff2.toolkit.openmmtool import generate_openmm_system

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────
CONFIG_FILE = "transport_config.json"
OUTPUT_DIR = "production_results"
TEMPERATURE = 298  # K


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ns", type=float, default=50.0, help="Additional ns to run")
    args = parser.parse_args()

    additional_ns = args.ns
    additional_steps = int(additional_ns * 1e6 / 2)  # 2 fs timestep
    logger.info(f"Restart NVT: {additional_ns} ns = {additional_steps:,} steps")

    with open(CONFIG_FILE) as f:
        config = json.load(f)

    params_dir = config['params_dir']
    gro_file = os.path.join(params_dir, 'solvent_salt.gro')
    top_file = os.path.join(params_dir, 'system.top')

    # Step 1: Regenerate force field parameters (needed for OpenMM system)
    logger.info("Regenerating force field parameters")
    tp = TransportProtocol(config)
    nonbonded_params = tp.generate_ff_params(config['smiles'])

    # Step 2: Create OpenMM system
    gro = app.GromacsGroFile(gro_file)
    unit_cell = gro.getUnitCellDimensions()
    input_top, input_system = generate_openmm_system(top_file, nonbonded_params, unit_cell)
    logger.info(f"System: {input_system.getNumParticles()} atoms")

    # Step 3: Extract final positions and box from existing NVT trajectory
    existing_dcd = os.path.join(OUTPUT_DIR, 'nvt.dcd')
    logger.info(f"Reading final frame from {existing_dcd}")

    u = mda.Universe(gro_file, existing_dcd)
    u.trajectory[-1]  # go to last frame
    final_positions = u.atoms.positions / 10.0  # Angstrom → nm
    box = u.dimensions[:3] / 10.0  # Angstrom → nm

    logger.info(f"  Last frame: step {u.trajectory.n_frames}, "
                f"box = [{box[0]:.4f}, {box[1]:.4f}, {box[2]:.4f}] nm")

    # Convert to OpenMM quantities
    positions_omm = final_positions * ou.nanometer
    box_vec = np.diag(box) * ou.nanometer

    # Step 4: Run continuation NVT
    logger.info(f"Starting NVT continuation: {additional_steps:,} steps ({additional_ns} ns)")
    restart_dir = os.path.join(OUTPUT_DIR, 'restart')
    os.makedirs(restart_dir, exist_ok=True)

    nvt_positions, nvt_box_vec = nvt_run(
        input_top, input_system,
        positions_omm, box_vec,
        temperature=TEMPERATURE,
        work_dir=restart_dir,
        nvt_steps=additional_steps,
    )

    logger.info("Done!")
    logger.info(f"Restart trajectory: {restart_dir}/nvt.dcd")
    logger.info(f"Restart state: {restart_dir}/nvt_state.csv")


if __name__ == "__main__":
    main()
