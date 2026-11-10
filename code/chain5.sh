#!/usr/bin/env bash
# Round-5 exploration queue (AMENDMENTS.md, Amendment 2). Each job takes the shared GPU lock on its own,
# so at most one D1 job waits on it and other areas' jobs can run in between. Stops at the first failed smoke check.
cd "$(dirname "$0")"
V="<environment>/bin/python"
R() { local name=$1; shift; echo "$(date +%T) start $name"; flock <home>/.cache/d1-gpu.lock systemd-run --user --scope -q -p MemoryMax=7G -p MemorySwapMax=1G -p TasksMax=256 -- "$V" "$@" > "../outputs/${name}.log" 2>&1; local rc=$?; echo "$(date +%T) done $name rc=$rc"; return $rc; }
R exp29_damping exp29_damping.py
until [ -f ../data/de/SHA256SUMS ]; do sleep 30; done
R exp28_tokenbias_smoke exp28_tokenbias.py --smoke || exit 1
R exp28_tokenbias exp28_tokenbias.py
for a in "train 60" "train 40" "explore-code 60" "explore-de 60"; do set -- $a; R "p5_04_general_$1_w$2" p5_04_general.py $1 $2; done
R exp27_suff_search exp27_suff_search.py
for a in "train 20" "train 30" "train 100" "train 120" "explore-code 40" "explore-de 40"; do set -- $a; R "p5_04_general_$1_w$2" p5_04_general.py $1 $2; done
echo "$(date +%T) chain finished"
