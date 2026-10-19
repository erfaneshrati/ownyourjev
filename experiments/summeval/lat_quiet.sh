#!/bin/bash
# wait (up to 40 min) for 1-min load < 6, then measure latency 3x; keep every run
cd /media/erfan/Data/ownyourjev/experiments/summeval
for i in $(seq 1 240); do l=$(cut -d' ' -f1 /proc/loadavg); if (( $(echo "$l < 6" | bc) )); then break; fi; sleep 10; done
for r in 1 2 3; do
  echo "load=$(cut -d' ' -f1 /proc/loadavg)"
  uv run python export_mh.py latency ../../data/v2/summeval/models/${MODEL:-final_f0} --out ../../data/v2/summeval/lat_${MODEL:-quiet}$r.json 2>&1 | tail -1 | grep -o '"cpu_p50.*'
done
