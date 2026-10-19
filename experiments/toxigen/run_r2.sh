#!/bin/bash
cd /media/erfan/Data/ownyourjev
EX=pool_b_flash_b20,pool_b_flash_b20_s,pool_b_haiku_b20
for s in 0 1 2; do for try in 1 2 3 4 5 6; do
  uv run python experiments/toxigen/train_mt.py --name r2_ens4_s$s --extra_labels $EX --epochs 3 --seed $s --save 2>&1 | grep -E "^\{|saved|OutOfMemory|Error" > data/v2/toxigen/r2_ens4_s$s.log
  grep -q saved data/v2/toxigen/r2_ens4_s$s.log && break; sleep 60; done; done
echo ALLDONE
