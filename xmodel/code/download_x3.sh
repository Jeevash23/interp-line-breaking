#!/usr/bin/env bash
# D1-X Amendment 3 downloads. Sequential, into the HF cache.
set -u
for m in Qwen/Qwen3-1.7B-Base Qwen/Qwen3-0.6B-Base google/gemma-4-E2B HuggingFaceTB/SmolLM2-1.7B; do
  echo "$(date '+%F %T') start $m"
  "<environment>/bin/hf" download "$m" && echo "$(date '+%F %T') done $m" || echo "$(date '+%F %T') FAILED $m"
done
