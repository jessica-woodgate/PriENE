#! /bin/bash

#SBATCH --job-name=train_PriENE_colours_maximin
#SBATCH --output=train_PriENE_colours_maximin.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1

source ~/miniforge3/bin/activate
conda activate prenv

#maximin agents in a society of their own type, in colours without seasons: 200 days, 4 agents
python3 ~/PriENE/run.py train << EOF
colours
n
200
homogeneous
maximin
4
y
EOF
