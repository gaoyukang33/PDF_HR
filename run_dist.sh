#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <arg_file> <gpu_id1> [gpu_id2 ...]"
  echo "Example: $0 args/deepmimic_humanoid_ppo_args.txt 0 1"
  exit 1
fi

ARG_FILE="$1"
shift

DEVICES=()
for gpu in "$@"; do
  # optional sanity check: ensure it's an integer
  if ! [[ "$gpu" =~ ^[0-9]+$ ]]; then
    echo "Error: GPU id '$gpu' is not a non-negative integer."
    exit 1
  fi
  DEVICES+=("cuda:${gpu}")
done

exec python mimickit/run.py --arg_file "$ARG_FILE" --devices "${DEVICES[@]}" --visualize false