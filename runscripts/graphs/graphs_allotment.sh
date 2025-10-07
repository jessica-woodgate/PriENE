#! /bin/bash

#SBATCH --job-name=graphs_PriENE_allotment
#SBATCH --output=graphs_PriENE_allotment.out
#SBATCH --time=24:00:00 

source ~/miniforge3/bin/activate
conda activate PriENE

python3 ~/PriENE/run.py graphs << EOF
200_days
allotment
20
EOF