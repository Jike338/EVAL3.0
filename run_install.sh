#!/bin/bash
#SBATCH --job-name=my_job_name                  # Name of the job
#SBATCH --output=slurm_out/output_%j.log        # Output file (%j = job ID)
#SBATCH --time=01:00:00                         # Time limit (HH:MM:SS)
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --ntasks=1                              # Number of tasks
#SBATCH --cpus-per-task=4                       # Number of CPU cores per task
#SBATCH --mem-per-gpu=100G                      # Memory per GPU
#SBATCH --gpus-per-node=4                       # Request 4 GPUs per node

cd /home/jikezhong/EVAL3.0

# Load Python module
module load python/3.10

# Install dependencies (optional for debugging jobs)
pip install -r requirements.txt

# Keep the job alive for inspection (debugging placeholder)
tail -f /dev/null
