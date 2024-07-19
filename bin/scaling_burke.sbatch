#!/bin/bash
#SBATCH -J scaling 			# A single job name for the array
#SBATCH -n 1                # Number of cores
#SBATCH -t 0-23:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p shared           # Partition to submit to
#SBATCH --mem=4000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o logs/scaling_%A_%a.out # Standard output (jobid and array index)
#SBATCH -e logs/scaling_%A_%a.err # Standard error (jobid and array index)

# Load software modules and source conda environment
module load python/3.10.9-fasrc01
source activate pt38

# Run program
srun -c 1 python quick_scaling.py burke_"${SLURM_JOB_ID}"_"${SLURM_ARRAY_TASK_ID}" -t 6 --seed ${SLURM_ARRAY_TASK_ID} --fixed_episode_length 14550 --nepochs 1500
