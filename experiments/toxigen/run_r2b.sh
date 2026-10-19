#!/bin/bash
cd /media/erfan/Data/ownyourjev
EX=pool_b_flash_b20,pool_b_flash_b20_s,pool_b_haiku_b20
TR=dev_flash_g1,dev_b_flash_b20,dev_b_flash_b20_s,dev_b_haiku_b20
run() { name=$1; shift; for try in 1 2 3 4 5 6; do
  uv run python experiments/toxigen/train_mt.py --name $name --extra_labels $EX "$@" 2>&1 | grep -E "^\{|saved|OutOfMemory|Error|rows" > data/v2/toxigen/$name.log
  grep -q '"epoch": ' data/v2/toxigen/$name.log && ! grep -q OutOfMemory data/v2/toxigen/$name.log && break; sleep 60; done; }
run r2b_trans_s0 --epochs 3 --seed 0 --transductive $TR --save
run r2b_mse_s0 --epochs 3 --seed 0 --mse 4
run r2b_e5_s0 --epochs 5 --seed 0
run r2b_trans3_s0 --epochs 3 --seed 0 --transductive $TR --trans_weight 3
echo ALLDONE
