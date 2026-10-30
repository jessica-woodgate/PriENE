#! /bin/bash

#SBATCH --job-name=graphs_PriENE_allotment_seasons
#SBATCH --output=graphs_PriENE_allotment_seasons.out
#SBATCH --time=24:00:00

source ~/miniforge3/bin/activate
conda activate prenv

#graphs for the 200_days results of test_allotment_seasons.sh: allotment with seasons, 20 agents, homogeneous, with rule-based norms
python3 ~/PriENE/run.py graphs << EOF
200_days
allotment
20
y
homogeneous
y
EOF
