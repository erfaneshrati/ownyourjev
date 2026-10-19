#!/bin/bash
# sequential job queue with OOM retry. usage: queue.sh jobfile ; each line: name args...
cd /media/erfan/Data/ownyourjev/experiments/helpsteer2
while read -r name args; do
  [ -z "$name" ] && continue
  if grep -q BEST logs/$name.log 2>/dev/null; then continue; fi
  for try in 1 2 3 4 5 6 7 8; do
    NEED_GB=${NEED_GB:-4} uv run python train_mh.py $args --out /media/erfan/Data/ownyourjev/data/v2/helpsteer2/models/$name > logs/$name.log 2>&1
    grep -q BEST logs/$name.log && break
    sleep 60
  done
done < "$1"
