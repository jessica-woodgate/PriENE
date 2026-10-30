#! /bin/bash

#SBATCH --job-name=graphs_PriENE_basic_seasons
#SBATCH --output=graphs_PriENE_basic_seasons.out
#SBATCH --time=24:00:00

source ~/miniforge3/bin/activate
conda activate prenv

#graphs for the 200_days results of test_basic_seasons.sh: basic with seasons, 20 agents, homogeneous, with rule-based norms
python3 ~/PriENE/run.py graphs << EOF
200_days
basic
20
y
homogeneous
y
EOF
