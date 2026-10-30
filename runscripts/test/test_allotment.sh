#! /bin/bash

#SBATCH --job-name=test_PriENE_allotment
#SBATCH --output=test_PriENE_allotment.out
#SBATCH --time=24:00:00

source ~/miniforge3/bin/activate
conda activate prenv

#every agent type in a society of its own type, in allotment without seasons, with the 200_days checkpoints: 20 agents,
#2 allotments, writing data and rule-based norms
python3 ~/PriENE/run.py test << EOF
allotment
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
