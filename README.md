# PriENE
This repository includes the codebase for the paper "Operationalising Rawlsian Ethics for Fairness in Norm-Learning Agents."


## Table of Contents
- [Introduction](#introduction)
- [Initialisation](#initialisation)
- [Usage](#usage)

## Introduction
PriENE is a framework for implementing multiple normative ethics principles in decision-making capacities of norm-learning agents. This codebase facilitates the creation of agents that aggregate multiple principles to evaluate the effects of their actions on the well-being of other agents. By operationalising principles, PriENE agents learn behaviours which promote prosocial norms that balance individual interests with collective well-being. Evaluations in simulated harvesting scenarios demonstrate that PriENE agents enhance fairness and sustainability compared to agents that implement individual principles.

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