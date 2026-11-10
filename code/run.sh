#!/usr/bin/env bash
# Run one D1 script inside the memory cage: ./run.sh <script.py> [args...]
# Output goes to ../outputs/<script>.log
# Since 6 Nov 2026 (parallel jobs) every GPU job holds the shared lock ~/.cache/d1-gpu.lock.
cd "$(dirname "$0")"
V="<environment>/bin/python"
name="$(basename "$1" .py)"
exec flock <home>/.cache/d1-gpu.lock systemd-run --user --scope -q -p MemoryMax=7G -p MemorySwapMax=1G -p TasksMax=256 -- "$V" "$@" > "../outputs/${name}.log" 2>&1
