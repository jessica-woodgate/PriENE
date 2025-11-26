# PriENE
This repository includes the codebase for the paper "Combining Normative Ethics Principles to Learn Prosocial Behaviour".


## Table of Contents
- [Introduction](#introduction)
- [Initialisation](#initialisation)
- [Usage](#usage)

## Introduction
PriENE is a framework for implementing multiple normative ethics principles in decision-making capacities of norm-learning agents. This codebase facilitates the creation of agents that aggregate multiple principles to evaluate the effects of their actions on the well-being of other agents. By operationalising principles, PriENE agents learn behaviours that promote prosocial norms which balance individual interests with collective well-being. Evaluations in simulated harvesting scenarios demonstrate that PriENE agents enhance fairness and sustainability compared to agents that implement individual principles.
## Initialisation
To set up the environment, create a virtual environment using:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate renv
```

## Usage

To run the code, use the following command:

```bash
python run.py [train] [test] [graphs]
```

## Arguments

The following arguments can be used with the `run.py` script:

- `train`: Train the norm-learning agent.
- `test`: Evaluate the performance of the trained agent.
- `graphs`: Generate relevant plots for analysis.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{Woodgate+Ajmeri2025Combining
doi = {10.5555/3709347.3744013},
author = {Woodgate, Jessica and Ajmeri, Nirav},
title = {Combining Normative Ethics Principles to Learn Prosocial Behaviour},
year = {2025},
isbn = {9798400714269},
publisher = {International Foundation for Autonomous Agents and Multiagent Systems},
booktitle = {Proceedings of the 24th International Conference on Autonomous Agents and Multiagent Systems ({AAMAS})},
pages = {2789--2791},
numpages = {3},
address = {Detroit},
series = {AAMAS '25}
}
