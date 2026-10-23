#!/usr/bin/env python3
"""
ByteFF-Pol independent replica launcher (for transport error bars).

Identical protocol to run_production.py (fresh NPT -> rescale -> NVT), but:
  - writes to an isolated output dir   production_results_rep{N}/
  - defaults to a 40 ns NVT production  (--ns to override)

Each invocation draws fresh random velocities + Langevin/barostat noise
(OpenMM uses a non-fixed seed), so successive runs are statistically
independent replicas. Combine with the original 50 ns run (replica 1) to
report mean +/- std on D and sigma.

Usage (run several, varying --rep):
    conda activate byteff2
    export OPENMM_DIR=/path/to/byteff2/openmm_install/openmm
    export LD_LIBRARY_PATH=$OPENMM_DIR/lib:$LD_LIBRARY_PATH
    cd /path/to/byteff2 && export PYTHONPATH=$(pwd):$PYTHONPATH
    cd /path/to/run_dir     # holds transport_config.json and transport_params/
    python3 run_replica.py --rep 2 --ns 40 2>&1 | tee rep2.log
    python3 run_replica.py --rep 3 --ns 40 2>&1 | tee rep3.log
    # ... rep 4, rep 5
"""

import argparse
import json
import logging
import os

import openmm.app as app

from byteff2.toolkit.openmmtool import generate_openmm_system
from byteff2.toolkit.protocol import TransportProtocol
from byteff2.md_utils.md_run import npt_run, nvt_run, rescale_box

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

CONFIG_FILE = "transport_config.json"
TEMPERATURE = 298  # K
NPT_STEPS = 1_500_000    # 3 ns NPT equilibration (same as production)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep", type=int, required=True, help="Replica index (>=2; rep 1 is the original 50 ns run)")
    parser.add_argument("--ns", type=float, default=40.0, help="NVT production length in ns")
    args = parser.parse_args()

    nvt_steps = int(args.ns * 1e6 / 2)  # 2 fs outer timestep
    output_dir = f"production_results_rep{args.rep}"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Replica {args.rep}: NVT {args.ns} ns ({nvt_steps:,} steps) -> {output_dir}/")

    with open(CONFIG_FILE) as f:
        config = json.load(f)

    params_dir = config['params_dir']

    logger.info("Generating force field parameters")
    tp = TransportProtocol(config)
    nonbonded_params = tp.generate_ff_params(config['smiles'])

    # Reuse the already-built box (same composition); fresh dynamics make it independent.
    gro_file = os.path.join(params_dir, 'solvent_salt.gro')
    top_file = os.path.join(params_dir, 'system.top')

    if not os.path.exists(gro_file):
        logger.info("Building system with GROMACS")
        tp.build_system(config['natoms'], config['components'], config.get('working_dir', 'working_dir'))
    else:
        logger.info(f"Using existing system: {gro_file}")

    logger.info("Creating polarisable OpenMM system")
    gro = app.GromacsGroFile(gro_file)
    positions = gro.positions
    unit_cell = gro.getUnitCellDimensions()
    input_top, input_system = generate_openmm_system(top_file, nonbonded_params, unit_cell)
    logger.info(f"System: {input_system.getNumParticles()} atoms")

    logger.info(f"NPT equilibration: {NPT_STEPS} steps = {NPT_STEPS * 2 / 1e6:.1f} ns")
    npt_positions, npt_box_vec = npt_run(
        input_top, input_system, positions,
        npt_steps=NPT_STEPS, temperature=TEMPERATURE, work_dir=output_dir,
    )

    logger.info("Rescaling box to average NPT density")
    rescale_positions, rescale_box_vec = rescale_box(npt_positions, npt_box_vec, work_dir=output_dir)

    logger.info(f"NVT production: {nvt_steps} steps = {nvt_steps * 2 / 1e6:.1f} ns")
    nvt_run(
        input_top, input_system,
        rescale_positions, rescale_box_vec,
        temperature=TEMPERATURE, work_dir=output_dir, nvt_steps=nvt_steps,
    )

    logger.info("Done!")
    logger.info(f"Replica {args.rep} trajectory: {output_dir}/nvt.dcd")


if __name__ == "__main__":
    main()
