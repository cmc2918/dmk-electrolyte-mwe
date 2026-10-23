#!/bin/bash
# ── MACE multi-head finetuning for LiFSI in FAN+DMK solvents ──
# Foundation model: MACE-MH-0 (multi-head, matpes_r2scan head)
# Approach: multi-head finetuning
# Training data: 439 train / 82 val / 28 test frames (stride=100, 1 ps decorrelation)
# Hardware: NVIDIA RTX A4000 (16 GB VRAM) + cuEquivariance acceleration
#
# Hyperparameters per MACE multi-head finetuning docs:
#   --lr=0.0001
#   --forces_weight=10
#   --energy_weight=1
#   --ema_decay=0.99999

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mace_env

# Make cuequivariance link against the conda env's CUDA 12.6 libs
NVIDIA_LIBS=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH=$(echo $NVIDIA_LIBS/*/lib | tr ' ' ':'):$LD_LIBRARY_PATH

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$(dirname "$0")"          # the folder holding train.xyz, val.xyz and test.xyz

FOUNDATION_MODEL="$HOME/.cache/mace/macemh0model"   # where MACE stores the downloaded MACE-MH-0 model

mace_run_train \
    --name="LiFSI_FAN_DMK_mh0" \
    --foundation_model="$FOUNDATION_MODEL" \
    --foundation_head="matpes_r2scan" \
    --train_file="train.xyz" \
    --valid_file="val.xyz" \
    --test_file="test.xyz" \
    --energy_key="REF_energy" \
    --forces_key="REF_forces" \
    --model="MACE" \
    --batch_size=1 \
    --valid_batch_size=1 \
    --max_num_epochs=80 \
    --ema \
    --ema_decay=0.99999 \
    --amsgrad \
    --restart_latest \
    --device=cuda \
    --seed=42 \
    --energy_weight=1.0 \
    --forces_weight=10.0 \
    --loss="huber" \
    --huber_delta=0.1 \
    --lr=0.0001 \
    --scheduler_patience=5 \
    --patience=20 \
    --E0s="average" \
    --enable_cueq=True \
    --keep_checkpoints \
    --save_cpu
