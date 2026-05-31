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
CLUSTER="FASTER"          # Options: "FASTER", "GRACE", or "LAUNCH"
DO_TRAIN=false
DO_EVAL_SIM=true        # Evaluate on synthetic test split (base.yaml + test_sim.yaml)
DO_EVAL_REAL=true       # Evaluate on real data (base.yaml + test_real.yaml)

# Seed Configuration
NUM_SEEDS=5                # Number of seeds to run (used if SEED_LIST is empty)
SEED_LIST=""               # Optional: List of specific seeds, e.g., "1 30 0" (takes priority over NUM_SEEDS)
SEED_START=0               # Starting seed when using NUM_SEEDS (ignored if SEED_LIST is provided)

# Train SLURM resources
TRAIN_TIME="10:00:00"
TRAIN_NODES=1
TRAIN_NTASKS_PER_NODE=1
TRAIN_CPUS_PER_TASK=4
TRAIN_MEM="50G"
TRAIN_GRES="gpu:a100:1"
TRAIN_PARTITION="gpu"

# Eval SLURM resources (typically lighter)
EVAL_TIME="1:00:00"
EVAL_NODES=1
EVAL_NTASKS_PER_NODE=1
EVAL_CPUS_PER_TASK=4
EVAL_MEM="50G"
EVAL_GRES="gpu:1"        # untyped: let SLURM pick any free GPU (a100/t4/...) — eval fits on t4
EVAL_PARTITION="gpu"

SLURM_MAIL_USER="your.email@example.com"  # Email for notifications
MAIL_ENABLED=false                          # Set to false to suppress all job emails

# Extra Hydra overrides appended to every run (smoke test cap; clear for full training)
EXTRA_HYDRA_OVERRIDES=""

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
BASES=("base")
DATA_TYPE="predprey"

MODEL_TYPES=(
    "cnp"
    "acnp"
    "eqtnp"
    "te-eqtnp"
    "te-istnp-m32"
    "convcnp-unet"
    "convcnp-fno"
    "sf-convcnp-f5-e288"
    "sf-volterra-convcnp-f5-e128-l5-vr4"
)

# ============================================================================
# FUNCTIONS
# ============================================================================

sbatch_job() {
    local job_file="$1"
    local experiment_name="$2"
    local hydra_overrides="$3"
    local seed="$4"
    local operation="$5"
    local account="$6"
    local slurm_time="$7"
    local slurm_nodes="$8"
    local slurm_ntasks_per_node="$9"
    local slurm_cpus_per_task="${10}"
    local slurm_mem="${11}"
    local slurm_gres="${12}"
    local slurm_partition="${13}"
    local slurm_mail_user="${14}"
    local mail_enabled="${15}"  # "true" or "false"
    local custom_job_name="${16}"
    local dependency="${17}"   # e.g. "afterok:12345678" or ""

    local job_name=${custom_job_name:-$(basename "$job_file" .job)}

    local export_vars="EXPERIMENT_NAME=$experiment_name,HYDRA_OVERRIDES=$hydra_overrides,SEED=$seed,OPERATION=$operation,ACCOUNT=$account,SLURM_TIME=$slurm_time,SLURM_NODES=$slurm_nodes,SLURM_NTASKS_PER_NODE=$slurm_ntasks_per_node,SLURM_CPUS_PER_TASK=$slurm_cpus_per_task,SLURM_MEM=$slurm_mem,SLURM_GRES=$slurm_gres,SLURM_PARTITION=$slurm_partition"

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
if [[ "$DO_TRAIN" != "true" && "$DO_EVAL_SIM" != "true" && "$DO_EVAL_REAL" != "true" ]]; then
    echo "Error: at least one of DO_TRAIN, DO_EVAL_SIM, or DO_EVAL_REAL must be true"
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
[[ "$DO_TRAIN" == "true" && ( "$DO_EVAL_SIM" == "true" || "$DO_EVAL_REAL" == "true" ) ]] && OPS_LABEL+=" + "
[[ "$DO_EVAL_SIM" == "true" ]] && OPS_LABEL+="eval (sim)"
[[ "$DO_EVAL_SIM" == "true" && "$DO_EVAL_REAL" == "true" ]] && OPS_LABEL+=" + "
[[ "$DO_EVAL_REAL" == "true" ]] && OPS_LABEL+="eval (real)"

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
if [[ "$DO_EVAL_SIM" == "true" || "$DO_EVAL_REAL" == "true" ]]; then
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
    for MODEL_TYPE in "${MODEL_TYPES[@]}"; do
        EXPERIMENT_NAME="predprey/default"
        TRAIN_OVERRIDES="model/predprey=${MODEL_TYPE} benchmark/predprey=${BASE} ${EXTRA_HYDRA_OVERRIDES}"
        EVAL_SIM_OVERRIDES="model/predprey=${MODEL_TYPE} benchmark/predprey=test_sim misc.wandb_logging_enabled=false ${EXTRA_HYDRA_OVERRIDES}"
        EVAL_REAL_OVERRIDES="model/predprey=${MODEL_TYPE} benchmark/predprey=test_real misc.wandb_logging_enabled=false ${EXTRA_HYDRA_OVERRIDES}"

        for SEED in "${SEED_VALUES[@]}"; do
            TRAIN_JOB_ID=""

            if [[ "$DO_TRAIN" == "true" ]]; then
                CUSTOM_JOB_NAME="train_${MODEL_TYPE}_${DATA_TYPE}_s${SEED}"
                echo "Submitting train job: $CUSTOM_JOB_NAME"
                TRAIN_JOB_ID=$(sbatch_job "$JOB_FILE" \
                    "$EXPERIMENT_NAME" "$TRAIN_OVERRIDES" "$SEED" "train" "$ACCOUNT" \
                    "$TRAIN_TIME" "$TRAIN_NODES" "$TRAIN_NTASKS_PER_NODE" "$TRAIN_CPUS_PER_TASK" \
                    "$TRAIN_MEM" "$TRAIN_GRES" "$TRAIN_PARTITION" "$SLURM_MAIL_USER" \
                    "$MAIL_ENABLED" "$CUSTOM_JOB_NAME" "")
                if [[ -z "$TRAIN_JOB_ID" ]]; then
                    echo "Error: failed to extract job ID for $CUSTOM_JOB_NAME; aborting."
                    exit 1
                fi
                sleep 0.5
            fi

            if [[ "$DO_EVAL_SIM" == "true" ]]; then
                CUSTOM_JOB_NAME="eval_sim_${MODEL_TYPE}_${DATA_TYPE}_s${SEED}"
                DEPENDENCY=""
                [[ -n "$TRAIN_JOB_ID" ]] && DEPENDENCY="afterok:$TRAIN_JOB_ID"
                echo "Submitting eval (sim) job: $CUSTOM_JOB_NAME${DEPENDENCY:+ (depends on job $TRAIN_JOB_ID)}"
                if ! sbatch_job "$JOB_FILE" \
                    "$EXPERIMENT_NAME" "$EVAL_SIM_OVERRIDES" "$SEED" "eval" "$ACCOUNT" \
                    "$EVAL_TIME" "$EVAL_NODES" "$EVAL_NTASKS_PER_NODE" "$EVAL_CPUS_PER_TASK" \
                    "$EVAL_MEM" "$EVAL_GRES" "$EVAL_PARTITION" "$SLURM_MAIL_USER" \
                    "$MAIL_ENABLED" "$CUSTOM_JOB_NAME" "$DEPENDENCY" > /dev/null; then
                    echo "Error: failed to submit eval (sim) job for $CUSTOM_JOB_NAME; aborting."
                    exit 1
                fi
                sleep 0.5
            fi

            if [[ "$DO_EVAL_REAL" == "true" ]]; then
                CUSTOM_JOB_NAME="eval_real_${MODEL_TYPE}_${DATA_TYPE}_s${SEED}"
                DEPENDENCY=""
                [[ -n "$TRAIN_JOB_ID" ]] && DEPENDENCY="afterok:$TRAIN_JOB_ID"
                echo "Submitting eval (real) job: $CUSTOM_JOB_NAME${DEPENDENCY:+ (depends on job $TRAIN_JOB_ID)}"
                if ! sbatch_job "$JOB_FILE" \
                    "$EXPERIMENT_NAME" "$EVAL_REAL_OVERRIDES" "$SEED" "eval" "$ACCOUNT" \
                    "$EVAL_TIME" "$EVAL_NODES" "$EVAL_NTASKS_PER_NODE" "$EVAL_CPUS_PER_TASK" \
                    "$EVAL_MEM" "$EVAL_GRES" "$EVAL_PARTITION" "$SLURM_MAIL_USER" \
                    "$MAIL_ENABLED" "$CUSTOM_JOB_NAME" "$DEPENDENCY" > /dev/null; then
                    echo "Error: failed to submit eval (real) job for $CUSTOM_JOB_NAME; aborting."
                    exit 1
                fi
                sleep 0.5
            fi
        done
    done
done

echo ""
echo "============================================================================"
echo "All jobs submitted successfully!"
echo "============================================================================"
echo "Monitor job status with: squeue --me"
echo "Cancel jobs with: scancel --me"
