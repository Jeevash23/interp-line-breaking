#!/usr/bin/env bash
# Round-5 job queue (from 6 Nov, 17:40 UTC): runs the jobs listed in queue5.txt one at a time, first line first,
# so jobs can be added or reordered while it runs. Each line: <log name> <script> [args...]. A line "STOP" ends it.
# Before each job: the memory gate (>= 6 GB RAM available, checked every 60 s); then the shared GPU lock.
cd "$(dirname "$0")"
V="<environment>/bin/python"
memgate() { until [ "$(free -g | awk '/^Mem:/{print $7}')" -ge 6 ]; do sleep 60; done; }
while true; do
  line=$(head -n 1 queue5.txt 2>/dev/null)
  if [ -z "$line" ]; then sleep 30; continue; fi
  [ "$line" = "STOP" ] && { sed -i 1d queue5.txt; echo "$(date +%T) queue stopped"; exit 0; }
  sed -i 1d queue5.txt
  set -- $line; name=$1; shift
  memgate
  echo "$(date +%T) start $name"
  flock <home>/.cache/d1-gpu.lock systemd-run --user --scope -q -p MemoryMax=7G -p MemorySwapMax=1G -p TasksMax=256 -- "$V" "$@" > "../outputs/${name}.log" 2>&1
  echo "$(date +%T) done $name rc=$?"
done
