#!/bin/bash
for p in $(pgrep -f "label_pool.py qwen_silver|label_pool.py qwen_rest"); do [ "$p" != "$$" ] && ps -o args= -p $p | grep -q "^/.*python3\|^uv run" && kill $p; done
