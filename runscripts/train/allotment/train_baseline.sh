#! /bin/bash

#SBATCH --job-name=train_PriENE_allotment_baseline
#SBATCH --output=train_PriENE_allotment_baseline.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1

source ~/miniforge3/bin/activate
conda activate prenv

#baseline agents in a society of their own type, in allotment without seasons: 200 days, 4 agents, 2 allotments
python3 ~/PriENE/run.py train << EOF
allotment
n
200
homogeneous
baseline
4
2
y
EOF
