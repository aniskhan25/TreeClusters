#!/bin/bash

#SBATCH --job-name=patch-downloader
#SBATCH --account=project_462000684
#SBATCH --output=output/stdout/%A_%a.out
#SBATCH --error=output/stderr/%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=15:00:00
#SBATCH --partition=small
#SBATCH --mem-per-cpu=6000

if [ -z "$1" ]; then
    echo "Error: HPC_TYPE argument is required (e.g., 'puhti' or 'lumi')."
    exit 1
fi

export HPC_TYPE="$1"

export TREECLUST_REPO_PATH="/users/rahmanan/TreeClusters"

# Set global environment variables based on HPC type.
if [ "$HPC_TYPE" == "puhti" ]; then
    export TREECLUST_VENV_PATH="/projappl/project_2004205/rahmanan/venv"
    export TREECLUST_DATA_PATH="/scratch/project_2008436/rahmanan/tree_clusters/data"
    export TREECLUST_OUTPUT_PATH="/scratch/project_2008436/rahmanan/tree_clusters/output"
elif [ "$HPC_TYPE" == "lumi" ]; then
    export TREECLUST_VENV_PATH="/projappl/project_462000684/rahmanan/venv"
    export TREECLUST_DATA_PATH="/scratch/project_462000684/rahmanan/tree_clusters/data"
    export TREECLUST_OUTPUT_PATH="/scratch/project_462000684/rahmanan/tree_clusters/output"
else
    echo "Error: Unsupported HPC_TYPE '$HPC_TYPE'."
    exit 1
fi


# Batch processing setup
BATCH_INDEX=${2:-0}
BATCH_DIR="$TREECLUST_OUTPUT_PATH/batches"
mkdir -p "$BATCH_DIR"
SPLIT_PREFIX="$BATCH_DIR/batch_"
LINES_PER_BATCH=10000

# Only split if batches don't exist
if [ ! -f "${SPLIT_PREFIX}aa" ]; then
    echo "[INFO] Splitting clusters.csv into batches..."
    tail -n +2 "$TREECLUST_OUTPUT_PATH/sample.csv" | split -l $LINES_PER_BATCH -d --additional-suffix=.csv - "$SPLIT_PREFIX"
    for file in "$BATCH_DIR"/batch_*.csv; do
        sed -i '1i patch_id,x,y,event_type' "$file"
    done
fi

# Resolve current batch file
BATCH_FILE=$(ls "$BATCH_DIR"/batch_*.csv | sed -n "$((BATCH_INDEX + 1))p")
if [ -z "$BATCH_FILE" ]; then
    echo "[ERROR] No batch file found for index $BATCH_INDEX"
    exit 1
fi

echo "[INFO] Processing batch file: $BATCH_FILE"

module use /appl/local/csc/modulefiles/
module load pytorch/2.4

# Reset PATH to minimal system directories
export PATH="/usr/bin:/bin"

# Activate virtual environment
if [ -d "$TREECLUST_VENV_PATH" ]; then
    echo "[INFO] Activating virtual environment at $TREECLUST_VENV_PATH"
    source "$TREECLUST_VENV_PATH/bin/activate"
    # Prepend virtual environment's bin directory to PATH
    export PATH="$TREECLUST_VENV_PATH/bin:$PATH"
else
    echo "[ERROR] Virtual environment not found at $TREECLUST_VENV_PATH"
    exit 1
fi

# Verify PATH (for debugging)
echo "Current PATH: \$PATH"


# Run the Python script using the virtual environment's python3
srun python3 "$TREECLUST_REPO_PATH/src/patches.py" --data-path "$BATCH_FILE" --data-dir "$TREECLUST_DATA_PATH" --output-dir "$TREECLUST_OUTPUT_PATH"

EXIT_STATUS=$?
if [ "${EXIT_STATUS:-0}" -ne 0 ]; then
    echo "[ERROR] Job failed with exit status $EXIT_STATUS"
else
    echo "[INFO] Job completed successfully"
fi

exit $EXIT_STATUS
EOT

echo "Generated SBATCH script:"
cat $SBATCH_SCRIPT

# Submit SLURM Job with minimal environment
sbatch $SBATCH_SCRIPT