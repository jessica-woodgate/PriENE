#! /bin/bash

#SBATCH --job-name=train_PriENE_capabilities_seasons_egalitarian
#SBATCH --output=train_PriENE_capabilities_seasons_egalitarian.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1

source ~/miniforge3/bin/activate
conda activate prenv

#egalitarian agents in a society of their own type, in capabilities with 25-day seasons: 200 days, 4 agents, 2 capabilities
python3 ~/PriENE/run.py train << EOF
capabilities
y
25
200
homogeneous
egalitarian
4
2
y
EOF
