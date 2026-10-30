#! /bin/bash

#SBATCH --job-name=graphs_PriENE_colours_seasons
#SBATCH --output=graphs_PriENE_colours_seasons.out
#SBATCH --time=24:00:00

source ~/miniforge3/bin/activate
conda activate prenv

#graphs for the 200_days results of test_colours_seasons.sh: colours with seasons, 20 agents, homogeneous, with rule-based norms
python3 ~/PriENE/run.py graphs << EOF
200_days
colours
20
y
homogeneous
y
EOF
