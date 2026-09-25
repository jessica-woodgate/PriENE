#! /bin/bash

#SBATCH --job-name=train_PriENE_majoritarian
#SBATCH --output=train_PriENE_majoritarian.out
#SBATCH --time=24:00:00 
#SBATCH --nodes=1
#SBATCH --gpus=1

source ~/miniforge3/bin/activate
conda activate prenv

python3 ~/PriENE/run.py train << EOF
200
majoritarian
4
y
EOF