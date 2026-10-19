#!/bin/bash
cd /media/erfan/Data/ownyourjev/experiments/helpsteer2
export OYJ_BUDGET_SCOPE=helpsteer2 OYJ_SCOPE_CAP_USD=1.50
nohup uv run python label_pool.py qwen_silver silver > logs/label_qwen_silver.log 2>&1 &
sleep 2
nohup uv run python label_pool.py qwen_rest10k rest 10000 > logs/label_qwen_rest10k.log 2>&1 &
