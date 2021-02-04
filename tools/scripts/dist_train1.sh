#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}

CUDA_VISIBLE_DEVICES=9 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${NGPUS} train1.py --launcher pytorch ${PY_ARGS}

