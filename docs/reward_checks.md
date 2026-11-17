# Reward checks: scripted policies under candidate reward functions

Step 1 of the reward ablation: before any training, scripted policies (src/agent/modules/scripted_decision_module.py) are run in the basic scenario and every step's outcome is logged (src/reward_checks/simulate.py); each candidate reward function (configs/rewards/) is then evaluated offline on those logs, for every agent type (src/reward_checks/evaluate.py).

## Discounting

The DQN's gamma is 0.95. A reward on an episode's last day (day 200) is worth gamma^199 = 3.69e-05 of the same reward on its first day: a survive reward of 1 adds 3.69e-05 to the discounted return from the start of an episode, so returns from the start barely see whether an agent survives to the end.

## Set-ups

- homogeneous: every agent acts by the policy; focal: agent 0 acts by the policy and the other three are eaters.
- Validation episodes: each episode's environment is reseeded from the validation seed (2000) and the episode, so every set-up faces the same episodes.
- passive: there is no action that does nothing (move is foraging), so passive eats every step, which fails (no_berries) without a berry: it never forages, and its return includes that failed action's penalty every day. passive_neutral scores the same steps with those failed eats as actions with no other effect (neutral_reward): pure inaction.
- An agent that dies on the last day doesn't get the survive reward (as during a run).

## Failed checks

- **suggested_shaping**, average, basic: surviving > dying early (by 1 sd) (6.233 vs -0.381) — difference 6.614, sd 8.265; 4984 survived, 2908 died before day 100
- **suggested_shaping**, baseline, basic: surviving > dying early (by 1 sd) (5.417 vs -0.625) — difference 6.041, sd 8.187; 4984 survived, 2908 died before day 100
- **suggested_shaping**, egalitarian, basic: surviving > dying early (by 1 sd) (5.005 vs -0.562) — difference 5.566, sd 8.032; 4984 survived, 2908 died before day 100
- **suggested_shaping**, majoritarian, basic: surviving > dying early (by 1 sd) (6.512 vs -0.333) — difference 6.845, sd 8.235; 4984 survived, 2908 died before day 100
- **suggested_shaping**, maximin, basic: surviving > dying early (by 1 sd) (6.320 vs -0.366) — difference 6.686, sd 8.183; 4984 survived, 2908 died before day 100
- **suggested_shaping**, optimist, basic: surviving > dying early (by 1 sd) (7.610 vs -0.207) — difference 7.817, sd 8.439; 4984 survived, 2908 died before day 100
- **suggested_shaping**, utilitarian, basic: surviving > dying early (by 1 sd) (7.376 vs -0.215) — difference 7.591, sd 8.643; 4984 survived, 2908 died before day 100
- **suggested_shaping**, veto, basic: surviving > dying early (by 1 sd) (5.044 vs -0.554) — difference 5.598, sd 8.045; 4984 survived, 2908 died before day 100
- **current**, utilitarian, basic: sanction raises sharer relative to eater (-1.569 vs -1.288) — sharer - eaters: -1.569 with the sanction, -1.288 on base rewards alone; the sanction adds -0.281
- **current**, optimist, basic: sanction raises sharer relative to eater (-1.429 vs -1.288) — sharer - eaters: -1.429 with the sanction, -1.288 on base rewards alone; the sanction adds -0.141
- **suggested_ethics_as_current**, utilitarian, basic: sanction raises sharer relative to eater (-1.589 vs -1.308) — sharer - eaters: -1.589 with the sanction, -1.308 on base rewards alone; the sanction adds -0.281
- **suggested_ethics_as_current**, optimist, basic: sanction raises sharer relative to eater (-1.449 vs -1.308) — sharer - eaters: -1.449 with the sanction, -1.308 on base rewards alone; the sanction adds -0.141
- **suggested**, utilitarian, basic: sanction raises sharer relative to eater (-1.881 vs -1.600) — sharer - eaters: -1.881 with the sanction, -1.600 on base rewards alone; the sanction adds -0.281
- **suggested**, optimist, basic: sanction raises sharer relative to eater (-1.741 vs -1.600) — sharer - eaters: -1.741 with the sanction, -1.600 on base rewards alone; the sanction adds -0.141
- **suggested_alive**, utilitarian, basic: sanction raises sharer relative to eater (-1.937 vs -1.655) — sharer - eaters: -1.937 with the sanction, -1.655 on base rewards alone; the sanction adds -0.281
- **suggested_alive**, optimist, basic: sanction raises sharer relative to eater (-1.797 vs -1.655) — sharer - eaters: -1.797 with the sanction, -1.655 on base rewards alone; the sanction adds -0.141
- **suggested_death5**, utilitarian, basic: sanction raises sharer relative to eater (-2.437 vs -2.155) — sharer - eaters: -2.437 with the sanction, -2.155 on base rewards alone; the sanction adds -0.281
- **suggested_death5**, optimist, basic: sanction raises sharer relative to eater (-2.297 vs -2.155) — sharer - eaters: -2.297 with the sanction, -2.155 on base rewards alone; the sanction adds -0.141
- **suggested_forage05**, utilitarian, basic: sanction raises sharer relative to eater (-2.233 vs -1.951) — sharer - eaters: -2.233 with the sanction, -1.951 on base rewards alone; the sanction adds -0.281
- **suggested_forage05**, optimist, basic: sanction raises sharer relative to eater (-2.093 vs -1.951) — sharer - eaters: -2.093 with the sanction, -1.951 on base rewards alone; the sanction adds -0.141
- **suggested_forage08**, utilitarian, basic: sanction raises sharer relative to eater (-2.444 vs -2.162) — sharer - eaters: -2.444 with the sanction, -2.162 on base rewards alone; the sanction adds -0.281
- **suggested_forage08**, optimist, basic: sanction raises sharer relative to eater (-2.304 vs -2.162) — sharer - eaters: -2.304 with the sanction, -2.162 on base rewards alone; the sanction adds -0.141
- **suggested_shaping**, utilitarian, basic: sanction raises sharer relative to eater (-1.606 vs -1.325) — sharer - eaters: -1.606 with the sanction, -1.325 on base rewards alone; the sanction adds -0.281
- **suggested_shaping**, optimist, basic: sanction raises sharer relative to eater (-1.466 vs -1.325) — sharer - eaters: -1.466 with the sanction, -1.325 on base rewards alone; the sanction adds -0.141
- **suggested_throw05**, utilitarian, basic: sanction raises sharer relative to eater (-1.509 vs -1.228) — sharer - eaters: -1.509 with the sanction, -1.228 on base rewards alone; the sanction adds -0.281
- **suggested_throw05**, optimist, basic: sanction raises sharer relative to eater (-1.370 vs -1.228) — sharer - eaters: -1.370 with the sanction, -1.228 on base rewards alone; the sanction adds -0.141
- **suggested_throw08**, utilitarian, basic: sanction raises sharer relative to eater (-1.286 vs -1.005) — sharer - eaters: -1.286 with the sanction, -1.005 on base rewards alone; the sanction adds -0.281
- **suggested_throw08**, optimist, basic: sanction raises sharer relative to eater (-1.146 vs -1.005) — sharer - eaters: -1.146 with the sanction, -1.005 on base rewards alone; the sanction adds -0.141

## Checks passed per reward function

| reward function | checks passed | failed | not checked |
|---|---|---|---|
| current | 47 | 2 | 0 |
| suggested | 47 | 2 | 0 |
| suggested_alive | 47 | 2 | 0 |
| suggested_death5 | 47 | 2 | 0 |
| suggested_ethics_as_current | 47 | 2 | 0 |
| suggested_forage05 | 47 | 2 | 0 |
| suggested_forage08 | 47 | 2 | 0 |
| suggested_shaping | 39 | 10 | 0 |
| suggested_throw05 | 47 | 2 | 0 |
| suggested_throw08 | 47 | 2 | 0 |

## The sanction's effect on a sharer (focal set-up)

Sharer's discounted return minus the eaters', with the sanction and on base rewards alone, and the sanction's effect (the difference):

| reward function | agent type | with sanction | base only | effect |
|---|---|---|---|---|
| current | utilitarian | -1.569 | -1.288 | -0.281 |
| current | maximin | -0.544 | -1.288 | +0.743 |
| current | egalitarian | 0.334 | -1.288 | +1.622 |
| current | average | -0.593 | -1.288 | +0.695 |
| current | majoritarian | -0.596 | -1.288 | +0.692 |
| current | optimist | -1.429 | -1.288 | -0.141 |
| current | veto | 0.317 | -1.288 | +1.605 |
| suggested_ethics_as_current | utilitarian | -1.589 | -1.308 | -0.281 |
| suggested_ethics_as_current | maximin | -0.564 | -1.308 | +0.743 |
| suggested_ethics_as_current | egalitarian | 0.314 | -1.308 | +1.622 |
| suggested_ethics_as_current | average | -0.613 | -1.308 | +0.695 |
| suggested_ethics_as_current | majoritarian | -0.616 | -1.308 | +0.692 |
| suggested_ethics_as_current | optimist | -1.449 | -1.308 | -0.141 |
| suggested_ethics_as_current | veto | 0.297 | -1.308 | +1.605 |
| suggested | utilitarian | -1.881 | -1.600 | -0.281 |
| suggested | maximin | -0.857 | -1.600 | +0.743 |
| suggested | egalitarian | 0.022 | -1.600 | +1.622 |
| suggested | average | -0.905 | -1.600 | +0.695 |
| suggested | majoritarian | -0.908 | -1.600 | +0.692 |
| suggested | optimist | -1.741 | -1.600 | -0.141 |
| suggested | veto | 0.005 | -1.600 | +1.605 |
| suggested_alive | utilitarian | -1.937 | -1.655 | -0.281 |
| suggested_alive | maximin | -0.912 | -1.655 | +0.743 |
| suggested_alive | egalitarian | -0.034 | -1.655 | +1.622 |
| suggested_alive | average | -0.961 | -1.655 | +0.695 |
| suggested_alive | majoritarian | -0.964 | -1.655 | +0.692 |
| suggested_alive | optimist | -1.797 | -1.655 | -0.141 |
| suggested_alive | veto | -0.050 | -1.655 | +1.605 |
| suggested_death5 | utilitarian | -2.437 | -2.155 | -0.281 |
| suggested_death5 | maximin | -1.412 | -2.155 | +0.743 |
| suggested_death5 | egalitarian | -0.534 | -2.155 | +1.622 |
| suggested_death5 | average | -1.461 | -2.155 | +0.695 |
| suggested_death5 | majoritarian | -1.464 | -2.155 | +0.692 |
| suggested_death5 | optimist | -2.297 | -2.155 | -0.141 |
| suggested_death5 | veto | -0.550 | -2.155 | +1.605 |
| suggested_forage05 | utilitarian | -2.233 | -1.951 | -0.281 |
| suggested_forage05 | maximin | -1.208 | -1.951 | +0.743 |
| suggested_forage05 | egalitarian | -0.330 | -1.951 | +1.622 |
| suggested_forage05 | average | -1.257 | -1.951 | +0.695 |
| suggested_forage05 | majoritarian | -1.260 | -1.951 | +0.692 |
| suggested_forage05 | optimist | -2.093 | -1.951 | -0.141 |
| suggested_forage05 | veto | -0.346 | -1.951 | +1.605 |
| suggested_forage08 | utilitarian | -2.444 | -2.162 | -0.281 |
| suggested_forage08 | maximin | -1.419 | -2.162 | +0.743 |
| suggested_forage08 | egalitarian | -0.541 | -2.162 | +1.622 |
| suggested_forage08 | average | -1.468 | -2.162 | +0.695 |
| suggested_forage08 | majoritarian | -1.471 | -2.162 | +0.692 |
| suggested_forage08 | optimist | -2.304 | -2.162 | -0.141 |
| suggested_forage08 | veto | -0.557 | -2.162 | +1.605 |
| suggested_shaping | utilitarian | -1.606 | -1.325 | -0.281 |
| suggested_shaping | maximin | -0.581 | -1.325 | +0.743 |
| suggested_shaping | egalitarian | 0.297 | -1.325 | +1.622 |
| suggested_shaping | average | -0.630 | -1.325 | +0.695 |
| suggested_shaping | majoritarian | -0.633 | -1.325 | +0.692 |
| suggested_shaping | optimist | -1.466 | -1.325 | -0.141 |
| suggested_shaping | veto | 0.280 | -1.325 | +1.605 |
| suggested_throw05 | utilitarian | -1.509 | -1.228 | -0.281 |
| suggested_throw05 | maximin | -0.485 | -1.228 | +0.743 |
| suggested_throw05 | egalitarian | 0.394 | -1.228 | +1.622 |
| suggested_throw05 | average | -0.534 | -1.228 | +0.695 |
| suggested_throw05 | majoritarian | -0.537 | -1.228 | +0.692 |
| suggested_throw05 | optimist | -1.370 | -1.228 | -0.141 |
| suggested_throw05 | veto | 0.377 | -1.228 | +1.605 |
| suggested_throw08 | utilitarian | -1.286 | -1.005 | -0.281 |
| suggested_throw08 | maximin | -0.262 | -1.005 | +0.743 |
| suggested_throw08 | egalitarian | 0.617 | -1.005 | +1.622 |
| suggested_throw08 | average | -0.311 | -1.005 | +0.695 |
| suggested_throw08 | majoritarian | -0.314 | -1.005 | +0.692 |
| suggested_throw08 | optimist | -1.146 | -1.005 | -0.141 |
| suggested_throw08 | veto | 0.600 | -1.005 | +1.605 |

## Discounted returns (homogeneous set-ups, mean over agent-episodes)

### basic, baseline

| reward_function | altruist | eater | hoarder | passive | passive_neutral | random | sharer |
|---|---|---|---|---|---|---|---|
| current | 4.434 | 6.332 | 4.210 | -2.045 | -0.698 | -1.438 | 5.733 |
| suggested | 0.186 | 4.122 | 2.056 | -1.371 | -0.698 | -1.097 | 2.625 |
| suggested_alive | 0.577 | 4.500 | 2.408 | -1.251 | -0.578 | -0.958 | 3.019 |
| suggested_death5 | 0.101 | 3.895 | 1.571 | -4.165 | -3.492 | -3.700 | 2.567 |
| suggested_ethics_as_current | 0.186 | 4.122 | 2.056 | -1.371 | -0.698 | -1.097 | 2.625 |
| suggested_forage05 | 2.438 | 6.332 | 4.210 | -1.371 | -0.698 | -0.884 | 4.896 |
| suggested_forage08 | 3.789 | 7.657 | 5.502 | -1.371 | -0.698 | -0.756 | 6.259 |
| suggested_shaping | -12.637 | 9.866 | 8.573 | -3.640 | -2.967 | -3.016 | 0.973 |
| suggested_throw05 | 2.182 | 4.122 | 2.056 | -1.371 | -0.698 | -1.062 | 3.462 |
| suggested_throw08 | 3.380 | 4.122 | 2.056 | -1.371 | -0.698 | -1.041 | 3.964 |

### basic, egalitarian

| reward_function | altruist | eater | hoarder | passive | passive_neutral | random | sharer |
|---|---|---|---|---|---|---|---|
| current | 4.502 | 5.179 | 3.323 | -1.371 | -0.698 | -0.976 | 5.114 |
| suggested | 0.295 | 3.806 | 1.605 | -1.371 | -0.698 | -1.196 | 2.533 |
| suggested_alive | 0.687 | 4.183 | 1.957 | -1.251 | -0.578 | -1.056 | 2.927 |
| suggested_death5 | 0.210 | 3.579 | 1.120 | -4.165 | -3.492 | -3.799 | 2.475 |
| suggested_ethics_as_current | 0.254 | 2.970 | 1.170 | -1.371 | -0.698 | -1.224 | 2.005 |
| suggested_forage05 | 2.547 | 6.015 | 3.759 | -1.371 | -0.698 | -0.982 | 4.805 |
| suggested_forage08 | 3.899 | 7.341 | 5.051 | -1.371 | -0.698 | -0.854 | 6.168 |
| suggested_shaping | -12.527 | 9.550 | 8.122 | -3.640 | -2.967 | -3.114 | 0.881 |
| suggested_throw05 | 2.292 | 3.806 | 1.605 | -1.371 | -0.698 | -1.160 | 3.370 |
| suggested_throw08 | 3.490 | 3.806 | 1.605 | -1.371 | -0.698 | -1.139 | 3.872 |

### basic, maximin

| reward_function | altruist | eater | hoarder | passive | passive_neutral | random | sharer |
|---|---|---|---|---|---|---|---|
| current | 5.890 | 6.224 | 4.403 | -1.371 | -0.698 | -0.793 | 6.365 |
| suggested | 1.683 | 4.850 | 2.685 | -1.371 | -0.698 | -1.013 | 3.784 |
| suggested_alive | 2.075 | 5.228 | 3.036 | -1.251 | -0.578 | -0.874 | 4.178 |
| suggested_death5 | 1.598 | 4.623 | 2.199 | -4.165 | -3.492 | -3.617 | 3.726 |
| suggested_ethics_as_current | 1.642 | 4.014 | 2.249 | -1.371 | -0.698 | -1.042 | 3.256 |
| suggested_forage05 | 3.935 | 7.059 | 4.838 | -1.371 | -0.698 | -0.800 | 6.055 |
| suggested_forage08 | 5.286 | 8.385 | 6.130 | -1.371 | -0.698 | -0.672 | 7.418 |
| suggested_shaping | -11.139 | 10.594 | 9.202 | -3.640 | -2.967 | -2.932 | 2.132 |
| suggested_throw05 | 3.680 | 4.850 | 2.685 | -1.371 | -0.698 | -0.978 | 4.621 |
| suggested_throw08 | 4.877 | 4.850 | 2.685 | -1.371 | -0.698 | -0.957 | 5.123 |

### basic, utilitarian

| reward_function | altruist | eater | hoarder | passive | passive_neutral | random | sharer |
|---|---|---|---|---|---|---|---|
| current | 6.194 | 7.263 | 5.497 | -1.371 | -0.698 | -0.706 | 7.023 |
| suggested | 1.987 | 5.890 | 3.779 | -1.371 | -0.698 | -0.926 | 4.442 |
| suggested_alive | 2.379 | 6.267 | 4.131 | -1.251 | -0.578 | -0.787 | 4.836 |
| suggested_death5 | 1.902 | 5.663 | 3.294 | -4.165 | -3.492 | -3.530 | 4.384 |
| suggested_ethics_as_current | 1.946 | 5.054 | 3.344 | -1.371 | -0.698 | -0.955 | 3.914 |
| suggested_forage05 | 4.239 | 8.099 | 5.933 | -1.371 | -0.698 | -0.713 | 6.713 |
| suggested_forage08 | 5.591 | 9.424 | 7.225 | -1.371 | -0.698 | -0.585 | 8.076 |
| suggested_shaping | -10.835 | 11.633 | 10.296 | -3.640 | -2.967 | -2.845 | 2.790 |
| suggested_throw05 | 3.984 | 5.890 | 3.779 | -1.371 | -0.698 | -0.891 | 5.279 |
| suggested_throw08 | 5.181 | 5.890 | 3.779 | -1.371 | -0.698 | -0.870 | 5.781 |

## What the policies do (whatever the rewards)

| scenario | setup | policy | role | survival_time | survived_share | berries_eaten | berries_carried | berries_thrown | failed_actions |
|---|---|---|---|---|---|---|---|---|---|
| basic | focal | altruist | eaters | 186.10 | 0.92 | 42.97 | 0.23 | 0.00 | 0.00 |
| basic | focal | altruist | focal | 41.45 | 0.01 | 5.58 | 0.36 | 3.48 | 0.00 |
| basic | focal | hoarder | eaters | 168.98 | 0.81 | 37.45 | 0.21 | 0.00 | 0.00 |
| basic | focal | hoarder | focal | 154.76 | 0.70 | 25.16 | 9.93 | 0.00 | 0.00 |
| basic | focal | random | eaters | 184.30 | 0.91 | 42.75 | 0.25 | 0.00 | 0.00 |
| basic | focal | random | focal | 8.63 | 0.00 | 0.10 | 0.20 | 0.04 | 6.79 |
| basic | focal | sharer | eaters | 187.84 | 0.93 | 41.93 | 0.23 | 0.00 | 0.00 |
| basic | focal | sharer | focal | 125.90 | 0.56 | 25.94 | 0.28 | 1.33 | 0.00 |
| basic | homogeneous | altruist | all | 189.42 | 0.93 | 0.72 | 0.50 | 40.47 | 0.00 |
| basic | homogeneous | eater | all | 171.59 | 0.83 | 38.31 | 0.25 | 0.00 | 0.00 |
| basic | homogeneous | hoarder | all | 129.23 | 0.53 | 20.73 | 7.16 | 0.00 | 0.00 |
| basic | homogeneous | passive | all | 8.00 | 0.00 | 0.00 | 0.00 | 0.00 | 8.00 |
| basic | homogeneous | passive_neutral | all | 8.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| basic | homogeneous | random | all | 9.67 | 0.00 | 0.20 | 0.27 | 0.08 | 7.47 |
| basic | homogeneous | sharer | all | 192.85 | 0.95 | 38.72 | 0.24 | 3.31 | 0.00 |

Full results: data/reward_checks/results.csv; every check: data/reward_checks/checks.csv.
