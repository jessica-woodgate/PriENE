#! /bin/bash

#SBATCH --job-name=train_PriENE_capabilities_utilitarian
#SBATCH --output=train_PriENE_capabilities_utilitarian.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1

source ~/miniforge3/bin/activate
conda activate prenv

#utilitarian agents in a society of their own type, in capabilities without seasons: 200 days, 4 agents, 2 capabilities
python3 ~/PriENE/run.py train << EOF
capabilities
n
200
homogeneous
utilitarian
4
2
y
EOF
