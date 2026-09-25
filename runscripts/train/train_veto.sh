#! /bin/bash

#SBATCH --job-name=train_PriENE_veto
#SBATCH --output=train_PriENE_veto.out
#SBATCH --time=24:00:00 
#SBATCH --nodes=1
#SBATCH --gpus=1

source ~/miniforge3/bin/activate
conda activate prenv

python3 ~/PriENE/run.py train << EOF
200
veto
4
y
EOF