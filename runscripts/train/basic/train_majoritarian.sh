#! /bin/bash

#SBATCH --job-name=train_PriENE_basic_majoritarian
#SBATCH --output=train_PriENE_basic_majoritarian.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1

source ~/miniforge3/bin/activate
conda activate prenv

#majoritarian agents in a society of their own type, in basic without seasons: 200 days, 4 agents
python3 ~/PriENE/run.py train << EOF
basic
n
200
homogeneous
majoritarian
4
y
EOF
