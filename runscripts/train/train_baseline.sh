#! /bin/bash

#SBATCH --job-name=train_PriENE_baseline
#SBATCH --output=train_PriENE_baseline.out
#SBATCH --time=24:00:00 

source ~/miniforge3/bin/activate
conda activate PriENE

python3 ~/PriENE/run.py train --log PriENE << EOF
200
baseline
20
y
EOF