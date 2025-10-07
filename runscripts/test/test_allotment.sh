#! /bin/bash

#SBATCH --job-name=test_PriENE_allotment
#SBATCH --output=test_PriENE_allotment.out
#SBATCH --time=24:00:00 

source ~/miniforge3/bin/activate
conda activate PriENE

python3 ~/PriENE/run.py test << EOF
allotment
200_days
homogeneous
all
20
2
y
y
n
EOF