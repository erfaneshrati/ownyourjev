#!/bin/bash
cd /media/erfan/Data/ownyourjev
EX=pool_b_flash_b20,pool_b_flash_b20_s,pool_b_haiku_b20
run() { name=$1; shift; for try in 1 2 3 4 5 6 7 8; do
  uv run python experiments/toxigen/train_mt.py --name $name --extra_labels $EX --backbone modernbert-large --gpu_gb 12 "$@" 2>&1 | grep -E "^\{|saved|OutOfMemory|Error|rows" > data/v2/toxigen/$name.log
  grep -q saved data/v2/toxigen/$name.log && break; sleep 90; done; }
for s in 0 1 2; do run r3_large_s$s --epochs 3 --lr 2e-5 --seed $s --save; done
echo ALLDONE
