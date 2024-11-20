#!/bin/sh
#SBATCH --account=ie-idi       # E.g. "ie-idi" if you belong to IDI
#SBATCH --job-name=e10-sweep       # A name for the job
#SBATCH --time=0-0:45:00         # format: D-HH:MM:SS

#SBATCH --partition=GPUQ          # Asking for a GPU
#SBATCH --gres=gpu:1             # Setting the number of GPUs to 1
#SBATCH --mem=16G                 # Asking for 16GB RAM
#SBATCH --nodes=1
#SBATCH --output=output.txt      # Specifying 'stdout'
#SBATCH --error=output.err        # Specifying 'stderr'

#SBATCH --mail-user=jofa@stud.ntnu.no


WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

module purge

# Running your python file
module load Anaconda3/2024.02-1
conda activate tdt17

wandb agent --count 1 $SWEEP_ID

# python sweep.py

# wandb agent jofalck-ntnu/TDT17-mini-project/qioq52yh
# sweep_output=$(wandb sweep config.yaml) 
# sweep_id=$(echo "$sweep_output" | grep "Created sweep with ID:" | awk '{print $5}')
# wandb agent $sweep_id


# SWEEP_ID="your_sweep_id"
# for i in {1..2}; do
#     sbatch --export=SWEEP_ID=$SWEEP_ID run_sweep.sh
# done
