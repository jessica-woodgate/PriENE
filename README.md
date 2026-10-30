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

- `train`: Train the norm-learning agents.
- `test`: Evaluate the performance of trained agents.
- `graphs`: Generate relevant plots for analysis.

Options:

- `-s`, `--seed`: random seed for reproducible runs (default 42).
- `-l`, `--log`: log to the given wandb project.

Everything else about a run is chosen through prompts. Both `train` and `test` run in any scenario (`basic`, `colours`, `capabilities`, `allotment`), with or without seasons: the scenario sets the environment, and the mode sets everything about learning.

- **train**: agents learn (exploration starts at 1, so agents first act at random, and decays as `exp(-0.001 × (episode − 1))` until it reaches 0.01, which ends training after about 4,600 episodes), and their networks are saved after every episode. Checkpoints are saved under `data/model_variables/trained/<name>/<n>_agents/<agent type>/agent_<id>/`, where the name holds every setting that changes what is learned, e.g. `allotment_seasons25_2allotments_regrow0-2_200days_seed42`, with a `settings.json` recording them. Training refuses to start if that agent type's checkpoints already exist, so it never overwrites a checkpoint.
- **train, mixed society**: half the agents are the chosen type and half are baseline, learning together (each type only learns from its own type's experiences). The name includes `mixed` (`mixed-separate` for separate allotments), and the agents are saved under `<n>_agents/<agent type>_with_baseline/<agent type or baseline>/agent_<id>/`, with `agent_types.json` saying which agent is which type, e.g. `{"agent_0": "maximin", "agent_1": "maximin", "agent_2": "baseline", "agent_3": "baseline"}`. Testing a mixed society loads these agents if the checkpoint has them, and otherwise each type's own agents (as with the paper's checkpoints). A mixed society must be tested with the same type for each agent as it was trained with (allotment's split layout makes odd agents baseline, the other scenarios and separate allotments the upper half).
- **test**: agents act on a loaded checkpoint without learning or exploring, for 1000 episodes. The prompt takes a run (`200_days`, `50_days`, `current_run`), the name of a trained checkpoint, or a path to a folder holding `<n>_agents` folders; pressing enter uses `200_days`, the agents trained in basic for the paper. The episode length comes from the checkpoint (`200_days` is 200 days; trained checkpoints use the days they were trained with). Agents trained in any scenario can be tested in any other with the same number of agents, since observations and actions only depend on the number of agents; loading a checkpoint for another number of agents fails with a message saying so.

Seasons and the number of days in each season (default 25) are asked in both modes. With seasons, every season is randomly normal, good (one more berry per agent in each group) or bad (one fewer, never below 0); berry allocations always stay with the same group. In basic the whole society is one group, so a good season adds one berry per agent and a bad one removes one per agent. Output filenames get `_seasons` after the scenario (and training outputs start with `train_`), and `graphs` asks whether the run had seasons so it reads the right files.

For example, to train utilitarian agents in colours with seasons, then test them in allotment:

```bash
python run.py train   # colours, y (seasons), enter (25 days), 200 (days), homogeneous, utilitarian, 4, ...
python run.py train   # allotment, n, 200, heterogeneous, maximin, 4, 2 (allotments), separate, ...  (maximin with baseline agents)
python run.py test    # allotment, y, enter, colours_seasons25_200days_seed42, homogeneous, utilitarian, 4, 2, ...
python run.py test    # colours, n, enter (200_days: the paper's basic-trained agents), homogeneous, all, 4, ...
```

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
