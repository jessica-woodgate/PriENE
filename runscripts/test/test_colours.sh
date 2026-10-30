#! /bin/bash

#SBATCH --job-name=test_PriENE_colours
#SBATCH --output=test_PriENE_colours.out
#SBATCH --time=24:00:00

source ~/miniforge3/bin/activate
conda activate prenv

#every agent type in a society of its own type, in colours without seasons, with the 200_days checkpoints: 20 agents,
#writing data and rule-based norms
python3 ~/PriENE/run.py test << EOF
colours
n
200_days
homogeneous
all
20
y
y
rule_based
n
EOF
