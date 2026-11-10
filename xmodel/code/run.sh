#!/usr/bin/env bash
# Run one D1-X script inside the memory cage, holding the shared GPU lock: ./run.sh <script.py> [args...]
# Log in ../outputs/<script>_<args>.log.
# Queue rules (6 Nov 2026): at most one D1-X job waits on the GPU lock at a time (local queue
# lock ~/.cache/d1x-queue.lock), and no new GPU job starts while ~/.cache/d1-gpu.round3 exists.
cd "$(dirname "$0")"
V="<environment>/bin/python"
name="$(basename "$1" .py)_$(echo "${@:2}" | tr ' /' '__' | tr -d '.-')"
exec 9><home>/.cache/d1x-queue.lock
flock 9
while [ -e <home>/.cache/d1-gpu.round3 ]; do sleep 20; done
flock <home>/.cache/d1-gpu.lock systemd-run --user --scope -q -p MemoryMax=7G -p MemorySwapMax=1G -p TasksMax=256 -- "$V" "$@" > "../outputs/${name}.log" 2>&1
