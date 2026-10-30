#! /bin/bash

#SBATCH --job-name=train_PriENE_allotment_egalitarian
#SBATCH --output=train_PriENE_allotment_egalitarian.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1

source ~/miniforge3/bin/activate
conda activate prenv

#egalitarian agents in a society of their own type, in allotment without seasons: 200 days, 4 agents, 2 allotments
python3 ~/PriENE/run.py train << EOF
allotment
n
200
homogeneous
egalitarian
4
2
y
EOF
