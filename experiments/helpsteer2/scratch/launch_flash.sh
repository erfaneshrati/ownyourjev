#!/bin/bash
cd /media/erfan/Data/ownyourjev/experiments/helpsteer2
export OYJ_BUDGET_SCOPE=helpsteer2 OYJ_SCOPE_CAP_USD=1.50
nohup uv run python label_flash_more.py flash_more 2400 > logs/label_flash_more.log 2>&1 &
