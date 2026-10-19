#!/bin/bash
cd /media/erfan/Data/ownyourjev/experiments/helpsteer2
export OYJ_BUDGET_SCOPE=helpsteer2 OYJ_SCOPE_CAP_USD=1.50
nohup uv run python label_pool.py qwen_restall rest > logs/label_qwen_restall.log 2>&1 &
