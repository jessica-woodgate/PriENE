#! /bin/bash

#SBATCH --job-name=train_PriENE_average
#SBATCH --output=train_PriENE_average.out
#SBATCH --time=24:00:00 
#SBATCH --partition=grace
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

source ~/miniforge3/bin/activate
conda activate PriENE

python3 ~/PriENE/run.py train << EOF
200
average
20
y
EOF