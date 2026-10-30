#! /bin/bash

#SBATCH --job-name=train_PriENE_basic_seasons_utilitarian
#SBATCH --output=train_PriENE_basic_seasons_utilitarian.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1

source ~/miniforge3/bin/activate
conda activate prenv

#utilitarian agents in a society of their own type, in basic with 25-day seasons: 200 days, 4 agents
python3 ~/PriENE/run.py train << EOF
basic
y
25
200
homogeneous
utilitarian
4
y
EOF
