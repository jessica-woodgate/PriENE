#! /bin/bash

#SBATCH --job-name=graphs_PriENE_colours
#SBATCH --output=graphs_PriENE_colours.out
#SBATCH --time=24:00:00 

source ~/miniforge3/bin/activate
conda activate PriENE

python3 ~/PriENE/run.py graphs << EOF
200_days
colours
20
EOF