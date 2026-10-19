#!/bin/bash
# usage: queue.sh jobs.txt  (each line: tag args...), runs 2 at a time
L=/tmp/claude-1000/-media-erfan-Data-ownyourjev/887ffe38-400c-4edb-bb16-1a88de00b3d4/scratchpad
cd /media/erfan/Data/ownyourjev/experiments/stsb
cat "$1" | xargs -P 2 -L 1 bash -c 'tag=$0; uv run python train2.py --tag $tag "$@" > '$L'/$tag.log 2>&1; echo done $tag'
