#! /bin/bash

#SBATCH --job-name=train_PriENE_allotment_seasons_optimist
#SBATCH --output=train_PriENE_allotment_seasons_optimist.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1

source ~/miniforge3/bin/activate
conda activate prenv

#optimist agents in a society of their own type, in allotment with 25-day seasons: 200 days, 4 agents, 2 allotments
python3 ~/PriENE/run.py train << EOF
allotment
y
25
200
homogeneous
optimist
4
2
y
EOF
