#!/bin/bash
for p in $(pgrep -f "queue2.sh scratch/jobsC"); do kill $p; done
for p in $(pgrep -f "train_dual.py --layers"); do kill $p; done
