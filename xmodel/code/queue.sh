#!/usr/bin/env bash
# Run D1-X jobs one after another (one line per job in the given file, arguments to run.sh).
cd "$(dirname "$0")"
while IFS= read -r line; do
  [ -z "$line" ] && continue
  echo "$(date +%H:%M:%S) start $line"; eval "./run.sh $line"; echo "$(date +%H:%M:%S) end $line"
done < "$1"
