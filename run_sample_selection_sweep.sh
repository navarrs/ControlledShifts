#!/usr/bin/env bash
set -euo pipefail

############################
# Usage
############################
usage() {
    cat <<EOF
Usage: $0 [options]

Sweeps sample selection across strategy x percentage x model x benchmark. Two modes:
  - train (default): train each model on the blacklist produced for (strategy, percentage), pointing
    sample_selection_path at the canonical selection dir.
  - generate (-G):   run controlledshifts.run_sample_selection to (re)create the blacklists first.

Selection files live at: <paths.experiment_cache_path>/sample_selection/<paths.tag>/<gen_folder>/ where gen_folder is
"no_model" for random_drop (model-independent) and the generating run's dated experiment id (via -e) otherwise. The path
is passed to both generation and training as the same Hydra interpolation, so the writer and reader always agree.

Options:
  -m <models>       Model(s), comma-separated. In generate mode each is treated as the embedding-generating model.
                    (default: wayformer, scenetransformer, autobot, mtr, cvm, naive)
  -d <devices>      Devices (e.g. 0 or 0,1)                          (default: 0)
  -b <benchmarks>   Benchmark paths group(s), comma-separated        (default: causal_agents, ego_safeshift, environments)
  -s <strategies>   Strategy/strategies, comma-separated             (default: random_drop)
  -p <percentages>  Retention percentage(s), comma-separated         (default: 0.45, 0.55, 0.65, 0.75, 0.85, 0.95)
  -e <gen_exp>      Generating run's dated experiment id for embedding strategies (ignored for random_drop) (default: no_model)
  -G                Generate mode: run run_sample_selection instead of training
  -c <ckpt_name>    Checkpoint name for generate mode (used to cache training-set outputs)
  -C                Generate mode: build the training batch cache before selecting (create_training_batch_cache=true)
  -n                Dry run (print commands, do not execute)
  -h                Show this help message

Examples:
  # Generate random_drop blacklists for wayformer across the default percentages on causal_agents
  $0 -G -m wayformer -b causal_agents -s random_drop

  # Train all default models on the random_drop selections for causal_agents
  $0 -b causal_agents -s random_drop
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
    causal_agents
    ego_safeshift
    environments
)
DEFAULT_STRATEGIES=(
    random_drop
)
DEFAULT_PERCENTAGES=(0.45 0.55 0.65 0.75 0.85 0.95)

# Valid SampleSelection strategies (must match controlledshifts.utils.constants.SampleSelection; no token strategies).
VALID_STRATEGIES=(
    random_drop
    kmeans_random_drop
    simple_kmeans_cosine_drop
    gumbel_kmeans_cosine_drop
    den_tp
    vocab_cluster_hamming_drop
    vocab_cluster_jaccard_drop
)

# Shared selection-path template. Single-quoted so the '${...}' stay as Hydra interpolations resolved per benchmark.
SEL_TEMPLATE='${paths.experiment_cache_path}/sample_selection/${paths.tag}/'

############################
# Parse arguments
############################
models=()
devices="$DEFAULT_DEVICES"
benchmarks=()
strategies=()
percentages=()
gen_experiment="no_model"
generate=false
ckpt_name=""
create_cache=false
dry_run=false

while getopts ":m:d:b:s:p:e:c:GCnh" opt; do
    case $opt in
        m) IFS=',' read -ra models <<< "$OPTARG" ;;
        d) devices="$OPTARG" ;;
        b) IFS=',' read -ra benchmarks <<< "$OPTARG" ;;
        s) IFS=',' read -ra strategies <<< "$OPTARG" ;;
        p) IFS=',' read -ra percentages <<< "$OPTARG" ;;
        e) gen_experiment="$OPTARG" ;;
        G) generate=true ;;
        c) ckpt_name="$OPTARG" ;;
        C) create_cache=true ;;
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
[[ ${#strategies[@]} -eq 0 ]] && strategies=("${DEFAULT_STRATEGIES[@]}")
[[ ${#percentages[@]} -eq 0 ]] && percentages=("${DEFAULT_PERCENTAGES[@]}")

############################
# Validate strategies
############################
for s in "${strategies[@]}"; do
    if ! printf '%s\n' "${VALID_STRATEGIES[@]}" | grep -qx "$s"; then
        echo "Error: strategy '$s' is not valid. Valid strategies: ${VALID_STRATEGIES[*]}" >&2
        exit 1
    fi
done

############################
# Run sweep
############################
for model in "${models[@]}"; do
    for benchmark in "${benchmarks[@]}"; do
        for strategy in "${strategies[@]}"; do
            # random_drop is model-independent; embedding strategies are scoped to the generating experiment.
            if [[ "$strategy" == "random_drop" ]]; then
                gen_folder="no_model"
            else
                gen_folder="$gen_experiment"
            fi
            selection_path="${SEL_TEMPLATE}${gen_folder}"

            # In generate mode a model-independent strategy (no_model) produces the same file for every model, so
            # generate it once (for the first model) instead of redundantly per model. Training still runs every model.
            if $generate && [[ "$gen_folder" == "no_model" && "$model" != "${models[0]}" ]]; then
                continue
            fi

            for pct in "${percentages[@]}"; do
                if $generate; then
                    cmd=(
                        uv run -m controlledshifts.run_sample_selection
                        model="$model"
                        "trainer.devices=[$devices]"
                        paths="$benchmark"
                        gen_experiment="$gen_folder"
                        "selection_strategies=[$strategy]"
                        "percentages_to_keep=[$pct]"
                        create_training_batch_cache="$create_cache"
                    )
                    [[ -n "$ckpt_name" ]] && cmd+=(ckpt_name="$ckpt_name")
                else
                    cmd=(
                        uv run -m controlledshifts.train
                        model="$model"
                        "trainer.devices=[$devices]"
                        paths="$benchmark"
                        sample_selection_strategy="$strategy"
                        percentage="$pct"
                        sample_selection_path="$selection_path"
                        sweep_type="_${strategy}_${pct}"
                    )
                fi

                echo "--------------------------------------------------"
                echo "Mode:           $([[ $generate == true ]] && echo generate || echo train)"
                echo "Model:          $model"
                echo "Devices:        $devices"
                echo "Benchmark:      $benchmark"
                echo "Strategy:       $strategy"
                echo "Percentage:     $pct"
                echo "Selection path: $selection_path"
                echo

                if $dry_run; then
                    echo "[DRY RUN] ${cmd[*]}"
                else
                    "${cmd[@]}"
                fi
            done
        done
    done
done
