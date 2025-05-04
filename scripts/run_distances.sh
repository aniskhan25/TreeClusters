#!/bin/bash

#SBATCH --job-name=compute-distance
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

# Set necessary environment variables for inference.
export DATA_PATH="$TREECLUST_OUTPUT_PATH/clusters.csv"

module use /appl/local/csc/modulefiles/
module load pytorch/2.5

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
srun python3 "$TREECLUST_REPO_PATH/src/distances.py" --data-path "$DATA_PATH" --output-dir "$TREECLUST_OUTPUT_PATH"

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