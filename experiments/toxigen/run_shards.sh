#!/bin/bash
cd /media/erfan/Data/ownyourjev
for s in 0 1 2 3 4 5; do
  OYJ_BUDGET_SCOPE=toxigen OYJ_SCOPE_CAP_USD=1.00 uv run python experiments/toxigen/label_pool.py $s 6 > data/v2/toxigen/label_pool.$s.log 2>&1 &
done
wait
uv run python experiments/toxigen/label_pool.py merge 6
