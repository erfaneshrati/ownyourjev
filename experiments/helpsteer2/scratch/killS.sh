#!/bin/bash
for p in $(pgrep -f "queue2.sh scratch/jobsS"); do kill $p; done
for p in $(pgrep -f "GF20s"); do kill $p; done
