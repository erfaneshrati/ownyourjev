#!/bin/bash
cd /media/erfan/Data/ownyourjev/experiments/helpsteer2
export OYJ_BUDGET_SCOPE=helpsteer2 OYJ_SCOPE_CAP_USD=3.44
nohup uv run python label_flash_more.py flash_more2 20000 2400 > logs/label_flash_more2.log 2>&1 &
