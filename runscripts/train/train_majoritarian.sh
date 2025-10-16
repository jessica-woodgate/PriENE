#! /bin/bash

#SBATCH --job-name=train_PriENE_majoritarian
#SBATCH --output=train_PriENE_majoritarian.out
#SBATCH --time=24:00:00 

source ~/miniforge3/bin/activate
conda activate PriENE

python3 ~/PriENE/run.py train << EOF
200
majoritarian
20
y
EOF