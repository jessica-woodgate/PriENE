#! /bin/bash

#SBATCH --job-name=test_PriENE_capabilities
#SBATCH --output=test_PriENE_capabilities.out
#SBATCH --time=24:00:00

source ~/miniforge3/bin/activate
conda activate prenv

#every agent type in a society of its own type, in capabilities without seasons, with the 200_days checkpoints: 20 agents,
#2 capabilities, writing data and rule-based norms
python3 ~/PriENE/run.py test << EOF
capabilities
n
200_days
homogeneous
all
20
2
y
y
rule_based
n
EOF
