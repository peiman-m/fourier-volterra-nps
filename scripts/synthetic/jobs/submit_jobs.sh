#!/bin/bash

# ============================================================================
# Neural Process Job Submission Script (Phase 6b.3 Hydra CLI)
# ============================================================================
# This script centralizes all SLURM job configuration and submits multiple
# training/evaluation jobs for different model types and configurations.
#
# Usage:
#   1. Make executable: chmod +x submit_jobs.sh
#   2. Edit configuration below, then run: ./submit_jobs.sh
# ============================================================================


# ============================================================================
# CONFIGURATION SECTION - Edit these variables as needed
# ============================================================================

# Job Configuration
CLUSTER="FASTER"           # Options: "FASTER", "GRACE", or "LAUNCH"
DO_TRAIN=false
DO_EVAL=true

# Seed Configuration
NUM_SEEDS=5                # Number of seeds to run (used if SEED_LIST is empty)
SEED_LIST=""               # Optional: List of specific seeds, e.g., "1 30 0" (takes priority over NUM_SEEDS)
SEED_START=0               # Starting seed when using NUM_SEEDS (ignored if SEED_LIST is provided)

# Train SLURM resources
TRAIN_TIME="8:00:00"
TRAIN_NODES=1
TRAIN_NTASKS_PER_NODE=1
TRAIN_CPUS_PER_TASK=4
TRAIN_MEM="50G"
TRAIN_GRES="gpu:a100:1"
TRAIN_PARTITION="gpu"

# Eval SLURM resources (typically lighter)
EVAL_TIME="2:00:00"
EVAL_NODES=1
EVAL_NTASKS_PER_NODE=1
EVAL_CPUS_PER_TASK=4
EVAL_MEM="50G"
EVAL_GRES="gpu:1"        # untyped: let SLURM pick any free GPU (a100/t4/...) — eval fits on t4
EVAL_PARTITION="gpu"

# Per-model EVAL_GRES overrides. Models listed here OOM on small GPUs
# (t4 has 15.5 GiB, a40 has 45 GiB) and must run on a100. The eval loop
# below falls back to $EVAL_GRES (above) for any model not listed here.
declare -A EVAL_GRES_BY_MODEL=(
    ["convcnp-unet"]="gpu:a100:1"
    ["convcnp-fno"]="gpu:a100:1"
    ["convcnp-unet-linear"]="gpu:a100:1"
    ["convcnp-fno-linear"]="gpu:a100:1"
)

SLURM_MAIL_USER="your.email@example.com"  # Email for notifications
MAIL_ENABLED=false                          # Set to false to suppress all job emails

# Extra Hydra overrides appended to every run (smoke test cap; clear for full training)
EXTRA_HYDRA_OVERRIDES="misc.wandb_logging_enabled=false"   # eval-only: W&B off (metrics still saved locally)

# Cluster Account Mappings
declare -A CLUSTER_ACCOUNTS=(
    ["FASTER"]="YOUR_FASTER_ACCOUNT"
    ["GRACE"]="YOUR_GRACE_ACCOUNT"
    # ["LAUNCH"]="YOUR_LAUNCH_ACCOUNT"   # Your LAUNCH cluster account
)

# ============================================================================
# JOB CONFIGURATION - Model types and base configs to run
# ============================================================================

JOB_FILE="job.job"
# Hydra benchmark/synthetic group members (conf/benchmark/synthetic/*.yaml).
# ``base`` is the experiment default — listing it here explicitly lets
# submit_jobs.sh loop over alternative bases (e.g. base-translation-test)
# without special-casing the default.
BASES=(
    "base"
    # "base-translation-test"
)
DATA_TYPE="Synthetic"
OUTPUT_GENERATORS=(
    "sawtooth"
    "squarewave"
    "gp-rbf"
    "gp-matern52"
    "gp-periodic"
)
INPUT_GENERATORS=(
    "uniform"
    # "mixturebeta"
    # translation-shift eval sweep (pair with base-translation-test):
    # "uniform-shift-0"
    # "uniform-shift-3"
    # "uniform-shift-6"
    # "uniform-shift-9"
    # "uniform-shift-12"
    # "uniform-shift-15"
)
MODEL_TYPES=(
    "cnp"
    "acnp"
    "eqtnp"
    "te-eqtnp"
    "te-istnp-m32"
    "convcnp-unet"
    "convcnp-fno"
    "sf-convcnp-f4.9-e288"
    "sf-volterra-convcnp-f4.9-e128-l5-vr4"

    # ### Ablations (-linear)
    # "cnp-linear"
    # "acnp-linear"
    # "eqtnp-linear"
    # "te-eqtnp-linear"
    # "te-istnp-m32-linear"
    # "convcnp-unet-linear"
    # "convcnp-fno-linear"
    # "sf-convcnp-f4.9-e288-linear"

    # ### frequency discretization
    # "sf-volterra-convcnp-f0.98-e128-l5-vr4"
    # "sf-volterra-convcnp-f2.45-e128-l5-vr4"
    # "sf-volterra-convcnp-f7.35-e128-l5-vr4"
    # "sf-volterra-convcnp-f9.8-e128-l5-vr4"
    # "sf-volterra-convcnp-f14.7-e128-l5-vr4"
    # "sf-volterra-convcnp-f19.6-e128-l5-vr4"

    # ### volterra low rank
    # "sf-volterra-convcnp-f4.95-e128-l5-vr1"
    # "sf-volterra-convcnp-f4.895-e128-l5-vr2"
    # "sf-volterra-convcnp-f4.752-e128-l5-vr6"
    # "sf-volterra-convcnp-f4.7-e128-l5-vr8"
    # "sf-volterra-convcnp-f4.66-e128-l5-vr10"

    ### capacity scaling: fixed freq_resolution=0.1, increasing max_freq
    # n_freq = int(max_freq / 0.1): 10, 25, 75, 100, 150, 200
    # "sf-volterra-convcnp-f0.9-e128-l5-vr4-fixedres"
    # "sf-volterra-convcnp-f2.4-e128-l5-vr4-fixedres"
    # "sf-volterra-convcnp-f7.4-e128-l5-vr4-fixedres"
    # "sf-volterra-convcnp-f9.9-e128-l5-vr4-fixedres"
    # "sf-volterra-convcnp-f14.9-e128-l5-vr4-fixedres"
    # "sf-volterra-convcnp-f19.9-e128-l5-vr4-fixedres"
)

# ============================================================================
# FUNCTIONS
# ============================================================================

sbatch_job() {
    local job_file="$1"
    local experiment_name="$2"
    local hydra_overrides="$3"
    local output_gen="$4"
    local seed="$5"
    local operation="$6"
    local account="$7"
    local slurm_time="$8"
    local slurm_nodes="$9"
    local slurm_ntasks_per_node="${10}"
    local slurm_cpus_per_task="${11}"
    local slurm_mem="${12}"
    local slurm_gres="${13}"
    local slurm_partition="${14}"
    local slurm_mail_user="${15}"
    local mail_enabled="${16}"  # "true" or "false"
    local custom_job_name="${17}"
    local dependency="${18}"   # e.g. "afterok:12345678" or ""

    local job_name=${custom_job_name:-$(basename "$job_file" .job)}

    local export_vars="EXPERIMENT_NAME=$experiment_name,HYDRA_OVERRIDES=$hydra_overrides,OUTPUT_GEN=$output_gen,SEED=$seed,OPERATION=$operation,ACCOUNT=$account,SLURM_TIME=$slurm_time,SLURM_NODES=$slurm_nodes,SLURM_NTASKS_PER_NODE=$slurm_ntasks_per_node,SLURM_CPUS_PER_TASK=$slurm_cpus_per_task,SLURM_MEM=$slurm_mem,SLURM_GRES=$slurm_gres,SLURM_PARTITION=$slurm_partition"

    # Pass --output via CLI (not #SBATCH) so the OUTPUT_GEN subdir interpolates
    # at submit time. Slurm on FASTER passes the script *path* to compute nodes
    # (Command=<path> in scontrol), so we hand it the original job_file directly
    # on shared /scratch — no per-job temp file, which also dodges the project
    # inode quota.
    local cmd=(sbatch
        --job-name="$job_name"
        --account="$account"
        --time="$slurm_time"
        --nodes="$slurm_nodes"
        --ntasks-per-node="$slurm_ntasks_per_node"
        --cpus-per-task="$slurm_cpus_per_task"
        --mem="$slurm_mem"
        --gres="$slurm_gres"
        --partition="$slurm_partition"
        --output="logs/${output_gen}/job-output_%x.txt"
        --export="$export_vars"
    )
    if [[ "$mail_enabled" == "true" ]]; then
        cmd+=(--mail-type=ALL --mail-user="$slurm_mail_user")
    fi
    [[ -n "$dependency" ]] && cmd+=(--dependency="$dependency")
    cmd+=("$job_file")

    local result
    local sbatch_rc
    result=$("${cmd[@]}")
    sbatch_rc=$?
    echo "$result" >&2
    [[ $sbatch_rc -ne 0 ]] && return 1
    local job_id
    job_id=$(echo "$result" | grep -F "Submitted batch job" | awk '{print $NF}')
    [[ -z "$job_id" ]] && return 1
    echo "$job_id"
}

# ============================================================================
# VALIDATION AND SETUP
# ============================================================================

# Validate cluster selection
CLUSTER_UPPER=$(echo "$CLUSTER" | tr '[:lower:]' '[:upper:]')
if [[ "$CLUSTER_UPPER" != "FASTER" && "$CLUSTER_UPPER" != "GRACE" && "$CLUSTER_UPPER" != "LAUNCH" ]]; then
    echo "Error: CLUSTER must be 'FASTER', 'GRACE', or 'LAUNCH', got '$CLUSTER'"
    exit 1
fi

# Validate operation flags
if [[ "$DO_TRAIN" != "true" && "$DO_EVAL" != "true" ]]; then
    echo "Error: at least one of DO_TRAIN or DO_EVAL must be true"
    exit 1
fi

# Get account number for the selected cluster
ACCOUNT=${CLUSTER_ACCOUNTS[$CLUSTER_UPPER]}
if [[ -z "$ACCOUNT" ]]; then
    echo "Error: No account number configured for cluster '$CLUSTER_UPPER'"
    exit 1
fi

# Determine seed configuration. SEED_VALUES is the bash array of seeds
# the main loop iterates over — one SLURM job per (cell, seed).
if [[ -n "$SEED_LIST" ]]; then
    SEED_VALUES=($SEED_LIST)
    SEED_INFO="Custom list: ($SEED_LIST)"
else
    SEED_VALUES=()
    for ((i=0; i<NUM_SEEDS; i++)); do
        SEED_VALUES+=($((SEED_START + i)))
    done
    SEED_INFO="Range: $SEED_START to $((SEED_START + NUM_SEEDS - 1))"
fi
SEED_COUNT=${#SEED_VALUES[@]}

# Build operations label
OPS_LABEL=""
[[ "$DO_TRAIN" == "true" ]] && OPS_LABEL+="train"
[[ "$DO_TRAIN" == "true" && "$DO_EVAL" == "true" ]] && OPS_LABEL+=" + "
[[ "$DO_EVAL" == "true" ]] && OPS_LABEL+="eval"

# Display configuration summary
echo "============================================================================"
echo "JOB SUBMISSION CONFIGURATION"
echo "============================================================================"
echo "  Cluster:    $CLUSTER_UPPER"
echo "  Account:    $ACCOUNT"
echo "  Seeds:      $SEED_INFO ($SEED_COUNT seeds)"
echo "  Operations: $OPS_LABEL"
echo ""
if [[ "$DO_TRAIN" == "true" ]]; then
    echo "  Train resources:"
    echo "    Time: $TRAIN_TIME  |  Nodes: $TRAIN_NODES  |  CPUs: $TRAIN_CPUS_PER_TASK  |  Mem: $TRAIN_MEM  |  GPU: $TRAIN_GRES"
    echo ""
fi
if [[ "$DO_EVAL" == "true" ]]; then
    echo "  Eval resources:"
    echo "    Time: $EVAL_TIME   |  Nodes: $EVAL_NODES  |  CPUs: $EVAL_CPUS_PER_TASK  |  Mem: $EVAL_MEM   |  GPU: $EVAL_GRES"
    echo ""
fi
echo "============================================================================"
echo ""

# ============================================================================
# JOB SUBMISSION LOOP
# ============================================================================

for BASE in "${BASES[@]}"; do
    for OUTPUT_GEN in "${OUTPUT_GENERATORS[@]}"; do
        for INPUT_GEN in "${INPUT_GENERATORS[@]}"; do
            for MODEL_TYPE in "${MODEL_TYPES[@]}"; do
                EXPERIMENT_NAME="synthetic/default"
                HYDRA_OVERRIDES="model/synthetic=${MODEL_TYPE} benchmark/synthetic=${BASE} benchmark/synthetic/output_generator=${OUTPUT_GEN} benchmark/synthetic/input_generator=${INPUT_GEN} ${EXTRA_HYDRA_OVERRIDES}"

                for SEED in "${SEED_VALUES[@]}"; do
                    TRAIN_JOB_ID=""

                    if [[ "$DO_TRAIN" == "true" ]]; then
                        CUSTOM_JOB_NAME="train_${MODEL_TYPE}_${OUTPUT_GEN}_${INPUT_GEN}_s${SEED}"
                        echo "Submitting train job: $CUSTOM_JOB_NAME"
                        TRAIN_JOB_ID=$(sbatch_job "$JOB_FILE" \
                            "$EXPERIMENT_NAME" "$HYDRA_OVERRIDES" "$OUTPUT_GEN" \
                            "$SEED" "train" "$ACCOUNT" \
                            "$TRAIN_TIME" "$TRAIN_NODES" "$TRAIN_NTASKS_PER_NODE" "$TRAIN_CPUS_PER_TASK" \
                            "$TRAIN_MEM" "$TRAIN_GRES" "$TRAIN_PARTITION" "$SLURM_MAIL_USER" \
                            "$MAIL_ENABLED" "$CUSTOM_JOB_NAME" "")
                        if [[ -z "$TRAIN_JOB_ID" ]]; then
                            echo "Error: failed to extract job ID for $CUSTOM_JOB_NAME; aborting."
                            exit 1
                        fi
                        sleep 0.5
                    fi

                    if [[ "$DO_EVAL" == "true" ]]; then
                        CUSTOM_JOB_NAME="eval_${MODEL_TYPE}_${OUTPUT_GEN}_${INPUT_GEN}_s${SEED}"
                        DEPENDENCY=""
                        [[ -n "$TRAIN_JOB_ID" ]] && DEPENDENCY="afterok:$TRAIN_JOB_ID"
                        # Per-model GRES override (see EVAL_GRES_BY_MODEL above)
                        EFFECTIVE_EVAL_GRES="${EVAL_GRES_BY_MODEL[$MODEL_TYPE]:-$EVAL_GRES}"
                        echo "Submitting eval job: $CUSTOM_JOB_NAME (gres=$EFFECTIVE_EVAL_GRES)${DEPENDENCY:+ (depends on job $TRAIN_JOB_ID)}"
                        if ! sbatch_job "$JOB_FILE" \
                            "$EXPERIMENT_NAME" "$HYDRA_OVERRIDES" "$OUTPUT_GEN" \
                            "$SEED" "eval" "$ACCOUNT" \
                            "$EVAL_TIME" "$EVAL_NODES" "$EVAL_NTASKS_PER_NODE" "$EVAL_CPUS_PER_TASK" \
                            "$EVAL_MEM" "$EFFECTIVE_EVAL_GRES" "$EVAL_PARTITION" "$SLURM_MAIL_USER" \
                            "$MAIL_ENABLED" "$CUSTOM_JOB_NAME" "$DEPENDENCY" > /dev/null; then
                            echo "Error: failed to submit eval job for $CUSTOM_JOB_NAME; aborting."
                            exit 1
                        fi
                        sleep 0.5
                    fi
                done
            done
        done
    done
done

echo ""
echo "============================================================================"
echo "All jobs submitted successfully!"
echo "============================================================================"
echo "Monitor job status with: squeue --me"
echo "Cancel jobs with: scancel --me"
