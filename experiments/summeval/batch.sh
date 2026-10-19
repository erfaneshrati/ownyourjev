#!/bin/bash
# usage: batch.sh "<name> <train> k=v ..." ...   (runs sequentially, prints per-epoch dev means)
export OYJ_GPU_SLOTS=0:3,1:2 NEED_GB=${NEED_GB:-5}
D=/media/erfan/Data/ownyourjev/data/v2/summeval/logs; mkdir -p $D
cd /media/erfan/Data/ownyourjev/experiments/summeval
for x in "$@"; do
  set -- $x; n=$1; tr=$2; shift 2
  uv run python run_exp.py $n --train $tr "$@" > $D/$n.log 2>&1
  echo "$n: $(grep "'epoch'" $D/$n.log | grep -v FINAL | sed -E "s/.*'epoch': ([0-9]+).*'mean': ([-0-9.]+).*/\1:\2/" | tr '\n' ' ')"
  grep FINAL $D/$n.log | sed 's/FINAL//'
done
