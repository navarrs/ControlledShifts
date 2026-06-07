#!/usr/bin/env bash
set -euo pipefail

############################
# Usage
############################
usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  -m <models>       Model(s), comma-separated
                    (default: wayformer, scenetransformer, autobot, mtr, naive, cvm)
  -d <devices>      Devices (e.g. 0 or 0,1)
                    (default: 0)
  -b <benchmarks>   Benchmark(s), comma-separated
                    (default: uniform, causal_agents, causal_agents_hard, ego_safeshift, environments)
  -e <extra>        Extra identifier for sweep type (e.g. "test")
                    (default: "")
  -n                Dry run (print commands, do not execute)
  -h                Show this help message

Examples:
  # Run with all defaults
  $0

  # Test specific model and benchmark
  $0 -m wayformer -b causal_agents

  # Multiple models and devices
  $0 -m wayformer,scenetransformer -d 0,1

  # Custom models
  $0 -m wayformer,scenetransformer

  # Dry run to preview commands
  $0 -m wayformer -n
EOF
    exit 1
}

############################
# Defaults
############################
DEFAULT_MODELS=(
    wayformer
    scenetransformer
    autobot
    mtr
    cvm
    naive
)
DEFAULT_DEVICES="0"
DEFAULT_BENCHMARKS=(
  uniform
  causal_agents
  causal_agents_hard
  ego_safeshift
  environments
)
dry_run=false

############################
# Parse arguments
############################
models=()
devices="$DEFAULT_DEVICES"
benchmarks=()
extra=""

while getopts ":m:d:b:e:nh" opt; do
    case $opt in
        m) IFS=',' read -ra models <<< "$OPTARG" ;;
        d) devices="$OPTARG" ;;
        b) IFS=',' read -ra benchmarks <<< "$OPTARG" ;;
        e) extra="$OPTARG" ;;
        n) dry_run=true ;;
        h) usage ;;
        \?) echo "Invalid option: -$OPTARG" >&2; usage ;;
        :) echo "Option -$OPTARG requires an argument." >&2; usage ;;
    esac
done

############################
# Apply defaults if empty
############################
[[ ${#models[@]} -eq 0 ]] && models=("${DEFAULT_MODELS[@]}")
[[ ${#benchmarks[@]} -eq 0 ]] && benchmarks=("${DEFAULT_BENCHMARKS[@]}")

############################
# Run experiments
############################
for model in "${models[@]}"; do
    for benchmark in "${benchmarks[@]}"; do
        sweep_type="_${benchmark}_${extra}"

        cmd=(
            uv run -m controlledshifts.train
            model="$model"
            trainer.devices="[$devices]"
            paths="$benchmark"
        )

        echo "--------------------------------------------------"
        echo "Model:      $model"
        echo "Devices:    $devices"
        echo "Benchmark:  $benchmark"
        echo

        if $dry_run; then
            echo "[DRY RUN] ${cmd[*]}"
        else
            "${cmd[@]}"
        fi
    done
done
