#! /bin/bash

#SBATCH --job-name=train_PriENE_egalitarian
#SBATCH --output=train_PriENE_egalitarian.out
#SBATCH --time=24:00:00 

source ~/miniforge3/bin/activate
conda activate PriENE

python3 ~/PriENE/run.py train << EOF
200
egalitarian
20
y
EOF