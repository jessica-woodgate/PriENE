#! /bin/bash

#SBATCH --job-name=train_PriENE_veto
#SBATCH --output=train_PriENE_veto.out
#SBATCH --time=24:00:00 

source ~/miniforge3/bin/activate
conda activate PriENE

python3 ~/PriENE/run.py train --log PriENE << EOF
200
veto
20
y
EOF