#!/bin/sh
# usage: run_seq.sh name "args" [name "args" ...] ; runs trainer jobs sequentially
cd "$(dirname "$0")"
export OYJ_GPU_SLOTS="0:3,1:2"
while [ $# -gt 1 ]; do
  n=$1; a=$2; shift 2
  uv run python trainer.py $a --out ../../data/v2/boolq/models/$n > ../../data/v2/boolq/logs/$n.log 2>&1
done
