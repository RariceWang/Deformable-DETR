#!/usr/bin/env bash
set -x
export TORCH_HOME=/workspace/.cache/torch
export DETR_HOME=/workspace/.cache/detr
export PYTHONPATH=$PYTHONPATH:/workspace:/workspace/models/ops

GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./run_job.sh