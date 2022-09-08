#!/bin/bash
#SBATCH -n 1                # Number of cores
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -t 0-04:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p shared           # Partition to submit to
#SBATCH --mem=4000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o logs/pytorch_%j.out   # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e logs/pytorch_%j.err   # File to which STDERR will be written, %j inserts jobid

# Load software modules and source conda environment
module load python/3.8.5-fasrc01
source activate pt38

# Run program
srun -c 1 python quick_train.py test
