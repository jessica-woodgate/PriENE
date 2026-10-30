# PriENE
This repository includes the codebase for the paper "Combining Normative Ethics Principles to Learn Prosocial Behaviour".


## Table of Contents
- [Introduction](#introduction)
- [Initialisation](#initialisation)
- [Usage](#usage)
- [Testing](#testing)

## Introduction
PriENE is a framework for implementing multiple normative ethics principles in decision-making capacities of norm-learning agents. This codebase facilitates the creation of agents that aggregate multiple principles to evaluate the effects of their actions on the well-being of other agents. By operationalising principles, PriENE agents learn behaviours that promote prosocial norms which balance individual interests with collective well-being. Evaluations in simulated harvesting scenarios demonstrate that PriENE agents enhance fairness and sustainability compared to agents that implement individual principles.
## Initialisation
To set up the environment, create a virtual environment using:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate prenv
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

Options:

- `-s`, `--seed`: random seed for reproducible runs (default 42).
- `-l`, `--log`: log to the given wandb project.

Everything else about a run is chosen through prompts. With `test`, these include seasons and the number of days in each season (default 25). With seasons, every season is randomly normal, good (one more berry per agent in each group) or bad (one fewer, never below 0); berry allocations always stay with the same group. Output filenames get `_seasons` after the scenario, and `graphs` asks whether the run had seasons so it reads the right files.

In the allotment scenario every allotment starts with an equal share of the berries, and berries in allotment 1 regrow 2 steps after they are foraged (instantly elsewhere). The grid is split into allotments of whole columns (the first allotments are one column wider when the width doesn't divide evenly). With a heterogeneous society, a prompt asks whether the two agent types are `split` (every allotment has both types) or `separate` (each type has half of the allotments, which needs an even number of allotments); separate runs get `_separate` in their output filenames.

Unequal starting health and random swaps of berry allocations between groups have been removed: every agent starts with the default health, and `--initial-healths`, `--season-quality`, `--season-swaps`, `--seasons` and `--season-length` fail with a message saying so.

## Testing

The test suite covers pure logic (exceptions, ethics/norms/movement modules), the DQN/network layer, and full-simulation integration/regression tests. Run it from the repository root with:

```bash
pytest
```

`pytest.ini` points it at the `tests/` directory and adds the repository root to `PYTHONPATH`, so no extra setup is needed once the environment is activated.

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
