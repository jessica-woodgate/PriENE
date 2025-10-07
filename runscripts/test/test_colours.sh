#! /bin/bash

#SBATCH --job-name=test_PriENE_colours
#SBATCH --output=test_PriENE_colours.out
#SBATCH --time=24:00:00 

source ~/miniforge3/bin/activate
conda activate PriENE

python3 ~/PriENE/run.py test << EOF
colours
200_days
homogeneous
all
20
y
y
n
EOF