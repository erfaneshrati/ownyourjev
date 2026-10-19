#!/bin/bash
# usage: run.sh name args...
cd /media/erfan/Data/ownyourjev/experiments/helpsteer2
n=$1; shift
NEED_GB=${NEED_GB:-3} nohup uv run python train_mh.py "$@" --out /media/erfan/Data/ownyourjev/data/v2/helpsteer2/models/$n > logs/$n.log 2>&1 &
