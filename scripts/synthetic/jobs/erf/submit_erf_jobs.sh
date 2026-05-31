#!/bin/bash

# ============================================================================
# Effective Receptive Field Job Submission Script
# ============================================================================
# Submits one SLURM job per (model, task, seed) combo. Each job runs
# `tools/analyze_erf.py` against the existing checkpoint at
# artifacts/<task>/.../seed=<s>/checkpoints/best.ckpt and writes the
# resulting figure to artifacts/<task>/.../seed=<s>/erf/erf.pdf.
#
# Usage:
#   1. Make executable: chmod +x submit_erf_jobs.sh
#   2. Edit configuration below, then run: ./submit_erf_jobs.sh
# ============================================================================


# ============================================================================
# CONFIGURATION SECTION - Edit these variables as needed
# ============================================================================

# Cluster
CLUSTER="FASTER"           # Options: "FASTER", "GRACE", or "LAUNCH"

# Seed Configuration
NUM_SEEDS=5                # Number of seeds to run (used if SEED_LIST is empty)
SEED_LIST=""               # Optional: List of specific seeds, e.g., "1 30 0"
SEED_START=0               # Starting seed when using NUM_SEEDS

# ERF SLURM resources.
#
# ERF runs at small batch_size (default 8, see conf/erf/default.yaml), so
# every model — including the heaviest transformers — fits on a t4
# (15.5 GiB). Using untyped `gpu:1` lets SLURM hand out whatever's free,
# which keeps queue wait times short. No per-model GRES overrides needed
# at this batch size.
ERF_TIME="01:00:00"
ERF_NODES=1
ERF_NTASKS_PER_NODE=1
ERF_CPUS_PER_TASK=4
ERF_MEM="50G"
ERF_GRES="gpu:1"
ERF_PARTITION="gpu"

SLURM_MAIL_USER="your.email@example.com"
MAIL_ENABLED=false

# ERF-specific Hydra overrides on top of `+erf=default`. Empty by default —
# add to override individual ERF knobs without editing conf/erf/default.yaml.
# The +erf=default group loader is added in erf_job.job, so individual
# field overrides here use bare keys (no leading `+`).
# Examples:
#   ERF_OVERRIDES="erf.n_samples=2048 erf.batch_size=16"
#   ERF_OVERRIDES="erf.targets=[0.0]"
ERF_OVERRIDES=""

# Cluster Account Mappings
declare -A CLUSTER_ACCOUNTS=(
    ["FASTER"]="YOUR_FASTER_ACCOUNT"
    ["GRACE"]="YOUR_GRACE_ACCOUNT"
)

# ============================================================================
# JOB CONFIGURATION - Model types and base configs to run
# ============================================================================

JOB_FILE="erf_job.job"
BASES=(
    "base"
)
OUTPUT_GENERATORS=(
    "sawtooth"
    "squarewave"
    "gp-rbf"
    "gp-matern52"
    "gp-periodic"
)
INPUT_GENERATORS=(
    "uniform"
)
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

sbatch_erf_job() {
    local job_file="$1"
    local experiment_name="$2"
    local hydra_overrides="$3"
    local output_gen="$4"
    local seed="$5"
    local account="$6"
    local slurm_time="$7"
    local slurm_nodes="$8"
    local slurm_ntasks_per_node="$9"
    local slurm_cpus_per_task="${10}"
    local slurm_mem="${11}"
    local slurm_gres="${12}"
    local slurm_partition="${13}"
    local slurm_mail_user="${14}"
    local mail_enabled="${15}"
    local custom_job_name="${16}"

    local job_name=${custom_job_name:-$(basename "$job_file" .job)}

    local export_vars="EXPERIMENT_NAME=$experiment_name,HYDRA_OVERRIDES=$hydra_overrides,OUTPUT_GEN=$output_gen,SEED=$seed,ACCOUNT=$account"

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

CLUSTER_UPPER=$(echo "$CLUSTER" | tr '[:lower:]' '[:upper:]')
if [[ "$CLUSTER_UPPER" != "FASTER" && "$CLUSTER_UPPER" != "GRACE" && "$CLUSTER_UPPER" != "LAUNCH" ]]; then
    echo "Error: CLUSTER must be 'FASTER', 'GRACE', or 'LAUNCH', got '$CLUSTER'"
    exit 1
fi

ACCOUNT=${CLUSTER_ACCOUNTS[$CLUSTER_UPPER]}
if [[ -z "$ACCOUNT" ]]; then
    echo "Error: No account number configured for cluster '$CLUSTER_UPPER'"
    exit 1
fi

# Determine seed configuration
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

# Display configuration summary
echo "============================================================================"
echo "ERF JOB SUBMISSION CONFIGURATION"
echo "============================================================================"
echo "  Cluster:    $CLUSTER_UPPER"
echo "  Account:    $ACCOUNT"
echo "  Seeds:      $SEED_INFO ($SEED_COUNT seeds)"
echo ""
echo "  ERF resources:"
echo "    Time: $ERF_TIME  |  Nodes: $ERF_NODES  |  CPUs: $ERF_CPUS_PER_TASK  |  Mem: $ERF_MEM  |  GPU: $ERF_GRES"
echo ""
echo "  ERF overrides (on top of +erf=default):"
echo "    ${ERF_OVERRIDES:-<none>}"
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
                HYDRA_OVERRIDES="model/synthetic=${MODEL_TYPE} benchmark/synthetic=${BASE} benchmark/synthetic/output_generator=${OUTPUT_GEN} benchmark/synthetic/input_generator=${INPUT_GEN} ${ERF_OVERRIDES}"

                for SEED in "${SEED_VALUES[@]}"; do
                    CUSTOM_JOB_NAME="erf_${MODEL_TYPE}_${OUTPUT_GEN}_${INPUT_GEN}_s${SEED}"
                    echo "Submitting ERF job: $CUSTOM_JOB_NAME (gres=$ERF_GRES)"
                    if ! sbatch_erf_job "$JOB_FILE" \
                        "$EXPERIMENT_NAME" "$HYDRA_OVERRIDES" "$OUTPUT_GEN" \
                        "$SEED" "$ACCOUNT" \
                        "$ERF_TIME" "$ERF_NODES" "$ERF_NTASKS_PER_NODE" "$ERF_CPUS_PER_TASK" \
                        "$ERF_MEM" "$ERF_GRES" "$ERF_PARTITION" "$SLURM_MAIL_USER" \
                        "$MAIL_ENABLED" "$CUSTOM_JOB_NAME" > /dev/null; then
                        echo "Error: failed to submit ERF job for $CUSTOM_JOB_NAME; aborting."
                        exit 1
                    fi
                    sleep 0.5
                done
            done
        done
    done
done

echo ""
echo "============================================================================"
echo "All ERF jobs submitted successfully!"
echo "============================================================================"
echo "Monitor job status with: squeue --me"
echo "Cancel jobs with: scancel --me"
