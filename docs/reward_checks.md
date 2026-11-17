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

- **suggested_shaping**, average, basic: surviving > dying early (by 1 sd) (4.576 vs 0.186) — difference 4.390, sd 8.672; 7635 survived, 3378 died before day 100
- **suggested_shaping**, baseline, basic: surviving > dying early (by 1 sd) (3.741 vs -0.125) — difference 3.866, sd 8.678; 7635 survived, 3378 died before day 100
- **suggested_shaping**, egalitarian, basic: surviving > dying early (by 1 sd) (3.361 vs -0.021) — difference 3.382, sd 8.521; 7635 survived, 3378 died before day 100
- **suggested_shaping**, majoritarian, basic: surviving > dying early (by 1 sd) (4.878 vs 0.243) — difference 4.635, sd 8.603; 7635 survived, 3378 died before day 100
- **suggested_shaping**, maximin, basic: surviving > dying early (by 1 sd) (4.684 vs 0.202) — difference 4.481, sd 8.562; 7635 survived, 3378 died before day 100
- **suggested_shaping**, optimist, basic: surviving > dying early (by 1 sd) (5.987 vs 0.388) — difference 5.599, sd 8.717; 7635 survived, 3378 died before day 100
- **suggested_shaping**, utilitarian, basic: surviving > dying early (by 1 sd) (5.683 vs 0.376) — difference 5.307, sd 8.994; 7635 survived, 3378 died before day 100
- **suggested_shaping**, veto, basic: surviving > dying early (by 1 sd) (3.400 vs -0.014) — difference 3.413, sd 8.531; 7635 survived, 3378 died before day 100
- **current**, utilitarian, basic: sanction raises sharer relative to eater (-1.569 vs -1.288) — sharer - eaters: -1.569 with the sanction, -1.288 on base rewards alone; the sanction adds -0.281
- **current**, optimist, basic: sanction raises sharer relative to eater (-1.429 vs -1.288) — sharer - eaters: -1.429 with the sanction, -1.288 on base rewards alone; the sanction adds -0.141
- **suggested_ethics_as_current**, utilitarian, basic: sanction raises sharer relative to eater (-1.589 vs -1.308) — sharer - eaters: -1.589 with the sanction, -1.308 on base rewards alone; the sanction adds -0.281
- **suggested_ethics_as_current**, optimist, basic: sanction raises sharer relative to eater (-1.449 vs -1.308) — sharer - eaters: -1.449 with the sanction, -1.308 on base rewards alone; the sanction adds -0.141
- **nonnegative**, utilitarian, basic: sanction raises sharer relative to eater (-1.742 vs -1.461) — sharer - eaters: -1.742 with the sanction, -1.461 on base rewards alone; the sanction adds -0.281
- **nonnegative**, optimist, basic: sanction raises sharer relative to eater (-1.602 vs -1.461) — sharer - eaters: -1.602 with the sanction, -1.461 on base rewards alone; the sanction adds -0.141
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
- **suggested_positive_sanction**, utilitarian, basic: sanction raises sharer relative to eater (-1.881 vs -1.600) — sharer - eaters: -1.881 with the sanction, -1.600 on base rewards alone; the sanction adds -0.281
- **suggested_positive_sanction**, optimist, basic: sanction raises sharer relative to eater (-1.741 vs -1.600) — sharer - eaters: -1.741 with the sanction, -1.600 on base rewards alone; the sanction adds -0.141
- **suggested_shaping**, utilitarian, basic: sanction raises sharer relative to eater (-1.606 vs -1.325) — sharer - eaters: -1.606 with the sanction, -1.325 on base rewards alone; the sanction adds -0.281
- **suggested_shaping**, optimist, basic: sanction raises sharer relative to eater (-1.466 vs -1.325) — sharer - eaters: -1.466 with the sanction, -1.325 on base rewards alone; the sanction adds -0.141
- **suggested_throw05**, utilitarian, basic: sanction raises sharer relative to eater (-1.509 vs -1.228) — sharer - eaters: -1.509 with the sanction, -1.228 on base rewards alone; the sanction adds -0.281
- **suggested_throw05**, optimist, basic: sanction raises sharer relative to eater (-1.370 vs -1.228) — sharer - eaters: -1.370 with the sanction, -1.228 on base rewards alone; the sanction adds -0.141
- **suggested_throw08**, utilitarian, basic: sanction raises sharer relative to eater (-1.286 vs -1.005) — sharer - eaters: -1.286 with the sanction, -1.005 on base rewards alone; the sanction adds -0.281
- **suggested_throw08**, optimist, basic: sanction raises sharer relative to eater (-1.146 vs -1.005) — sharer - eaters: -1.146 with the sanction, -1.005 on base rewards alone; the sanction adds -0.141
- **current**, utilitarian, basic: sanction raises altruist relative to eater (-2.703 vs -2.281) — altruist - eaters: -2.703 with the sanction, -2.281 on base rewards alone; the sanction adds -0.422
- **current**, optimist, basic: sanction raises altruist relative to eater (-2.861 vs -2.281) — altruist - eaters: -2.861 with the sanction, -2.281 on base rewards alone; the sanction adds -0.580
- **suggested_ethics_as_current**, utilitarian, basic: sanction raises altruist relative to eater (-2.815 vs -2.392) — altruist - eaters: -2.815 with the sanction, -2.392 on base rewards alone; the sanction adds -0.422
- **suggested_ethics_as_current**, optimist, basic: sanction raises altruist relative to eater (-2.973 vs -2.392) — altruist - eaters: -2.973 with the sanction, -2.392 on base rewards alone; the sanction adds -0.580
- **nonnegative**, utilitarian, basic: sanction raises altruist relative to eater (-3.117 vs -2.695) — altruist - eaters: -3.117 with the sanction, -2.695 on base rewards alone; the sanction adds -0.422
- **nonnegative**, optimist, basic: sanction raises altruist relative to eater (-3.008 vs -2.695) — altruist - eaters: -3.008 with the sanction, -2.695 on base rewards alone; the sanction adds -0.313
- **suggested**, utilitarian, basic: sanction raises altruist relative to eater (-3.354 vs -2.931) — altruist - eaters: -3.354 with the sanction, -2.931 on base rewards alone; the sanction adds -0.422
- **suggested**, optimist, basic: sanction raises altruist relative to eater (-3.512 vs -2.931) — altruist - eaters: -3.512 with the sanction, -2.931 on base rewards alone; the sanction adds -0.580
- **suggested_alive**, utilitarian, basic: sanction raises altruist relative to eater (-3.448 vs -3.026) — altruist - eaters: -3.448 with the sanction, -3.026 on base rewards alone; the sanction adds -0.422
- **suggested_alive**, optimist, basic: sanction raises altruist relative to eater (-3.606 vs -3.026) — altruist - eaters: -3.606 with the sanction, -3.026 on base rewards alone; the sanction adds -0.580
- **suggested_death5**, utilitarian, basic: sanction raises altruist relative to eater (-4.299 vs -3.876) — altruist - eaters: -4.299 with the sanction, -3.876 on base rewards alone; the sanction adds -0.422
- **suggested_death5**, optimist, basic: sanction raises altruist relative to eater (-4.457 vs -3.876) — altruist - eaters: -4.457 with the sanction, -3.876 on base rewards alone; the sanction adds -0.580
- **suggested_forage05**, utilitarian, basic: sanction raises altruist relative to eater (-3.881 vs -3.459) — altruist - eaters: -3.881 with the sanction, -3.459 on base rewards alone; the sanction adds -0.422
- **suggested_forage05**, optimist, basic: sanction raises altruist relative to eater (-4.039 vs -3.459) — altruist - eaters: -4.039 with the sanction, -3.459 on base rewards alone; the sanction adds -0.580
- **suggested_forage08**, utilitarian, basic: sanction raises altruist relative to eater (-4.198 vs -3.776) — altruist - eaters: -4.198 with the sanction, -3.776 on base rewards alone; the sanction adds -0.422
- **suggested_forage08**, optimist, basic: sanction raises altruist relative to eater (-4.356 vs -3.776) — altruist - eaters: -4.356 with the sanction, -3.776 on base rewards alone; the sanction adds -0.580
- **suggested_positive_sanction**, utilitarian, basic: sanction raises altruist relative to eater (-3.354 vs -2.931) — altruist - eaters: -3.354 with the sanction, -2.931 on base rewards alone; the sanction adds -0.422
- **suggested_positive_sanction**, optimist, basic: sanction raises altruist relative to eater (-3.244 vs -2.931) — altruist - eaters: -3.244 with the sanction, -2.931 on base rewards alone; the sanction adds -0.313
- **suggested_shaping**, utilitarian, basic: sanction raises altruist relative to eater (-2.662 vs -2.240) — altruist - eaters: -2.662 with the sanction, -2.240 on base rewards alone; the sanction adds -0.422
- **suggested_shaping**, optimist, basic: sanction raises altruist relative to eater (-2.821 vs -2.240) — altruist - eaters: -2.821 with the sanction, -2.240 on base rewards alone; the sanction adds -0.580
- **suggested_throw05**, utilitarian, basic: sanction raises altruist relative to eater (-2.714 vs -2.292) — altruist - eaters: -2.714 with the sanction, -2.292 on base rewards alone; the sanction adds -0.422
- **suggested_throw05**, optimist, basic: sanction raises altruist relative to eater (-2.873 vs -2.292) — altruist - eaters: -2.873 with the sanction, -2.292 on base rewards alone; the sanction adds -0.580
- **suggested_throw08**, utilitarian, basic: sanction raises altruist relative to eater (-2.331 vs -1.909) — altruist - eaters: -2.331 with the sanction, -1.909 on base rewards alone; the sanction adds -0.422
- **suggested_throw08**, optimist, basic: sanction raises altruist relative to eater (-2.489 vs -1.909) — altruist - eaters: -2.489 with the sanction, -1.909 on base rewards alone; the sanction adds -0.580
- **current**, utilitarian, basic: sanction raises sharer_hoarder relative to eater (-2.124 vs -1.927) — sharer_hoarder - eaters: -2.124 with the sanction, -1.927 on base rewards alone; the sanction adds -0.197
- **current**, optimist, basic: sanction raises sharer_hoarder relative to eater (-2.018 vs -1.927) — sharer_hoarder - eaters: -2.018 with the sanction, -1.927 on base rewards alone; the sanction adds -0.091
- **suggested_ethics_as_current**, utilitarian, basic: sanction raises sharer_hoarder relative to eater (-2.170 vs -1.973) — sharer_hoarder - eaters: -2.170 with the sanction, -1.973 on base rewards alone; the sanction adds -0.197
- **suggested_ethics_as_current**, optimist, basic: sanction raises sharer_hoarder relative to eater (-2.064 vs -1.973) — sharer_hoarder - eaters: -2.064 with the sanction, -1.973 on base rewards alone; the sanction adds -0.091
- **nonnegative**, utilitarian, basic: sanction raises sharer_hoarder relative to eater (-2.487 vs -2.290) — sharer_hoarder - eaters: -2.487 with the sanction, -2.290 on base rewards alone; the sanction adds -0.197
- **nonnegative**, optimist, basic: sanction raises sharer_hoarder relative to eater (-2.381 vs -2.290) — sharer_hoarder - eaters: -2.381 with the sanction, -2.290 on base rewards alone; the sanction adds -0.091
- **suggested**, utilitarian, basic: sanction raises sharer_hoarder relative to eater (-2.628 vs -2.431) — sharer_hoarder - eaters: -2.628 with the sanction, -2.431 on base rewards alone; the sanction adds -0.197
- **suggested**, optimist, basic: sanction raises sharer_hoarder relative to eater (-2.522 vs -2.431) — sharer_hoarder - eaters: -2.522 with the sanction, -2.431 on base rewards alone; the sanction adds -0.091
- **suggested_alive**, utilitarian, basic: sanction raises sharer_hoarder relative to eater (-2.685 vs -2.488) — sharer_hoarder - eaters: -2.685 with the sanction, -2.488 on base rewards alone; the sanction adds -0.197
- **suggested_alive**, optimist, basic: sanction raises sharer_hoarder relative to eater (-2.579 vs -2.488) — sharer_hoarder - eaters: -2.579 with the sanction, -2.488 on base rewards alone; the sanction adds -0.091
- **suggested_death5**, utilitarian, basic: sanction raises sharer_hoarder relative to eater (-3.194 vs -2.997) — sharer_hoarder - eaters: -3.194 with the sanction, -2.997 on base rewards alone; the sanction adds -0.197
- **suggested_death5**, optimist, basic: sanction raises sharer_hoarder relative to eater (-3.088 vs -2.997) — sharer_hoarder - eaters: -3.088 with the sanction, -2.997 on base rewards alone; the sanction adds -0.091
- **suggested_forage05**, utilitarian, basic: sanction raises sharer_hoarder relative to eater (-2.874 vs -2.677) — sharer_hoarder - eaters: -2.874 with the sanction, -2.677 on base rewards alone; the sanction adds -0.197
- **suggested_forage05**, optimist, basic: sanction raises sharer_hoarder relative to eater (-2.768 vs -2.677) — sharer_hoarder - eaters: -2.768 with the sanction, -2.677 on base rewards alone; the sanction adds -0.091
- **suggested_forage08**, utilitarian, basic: sanction raises sharer_hoarder relative to eater (-3.022 vs -2.825) — sharer_hoarder - eaters: -3.022 with the sanction, -2.825 on base rewards alone; the sanction adds -0.197
- **suggested_forage08**, optimist, basic: sanction raises sharer_hoarder relative to eater (-2.916 vs -2.825) — sharer_hoarder - eaters: -2.916 with the sanction, -2.825 on base rewards alone; the sanction adds -0.091
- **suggested_positive_sanction**, utilitarian, basic: sanction raises sharer_hoarder relative to eater (-2.628 vs -2.431) — sharer_hoarder - eaters: -2.628 with the sanction, -2.431 on base rewards alone; the sanction adds -0.197
- **suggested_positive_sanction**, optimist, basic: sanction raises sharer_hoarder relative to eater (-2.522 vs -2.431) — sharer_hoarder - eaters: -2.522 with the sanction, -2.431 on base rewards alone; the sanction adds -0.091
- **suggested_shaping**, utilitarian, basic: sanction raises sharer_hoarder relative to eater (-1.883 vs -1.686) — sharer_hoarder - eaters: -1.883 with the sanction, -1.686 on base rewards alone; the sanction adds -0.197
- **suggested_shaping**, optimist, basic: sanction raises sharer_hoarder relative to eater (-1.778 vs -1.686) — sharer_hoarder - eaters: -1.778 with the sanction, -1.686 on base rewards alone; the sanction adds -0.091
- **suggested_throw05**, utilitarian, basic: sanction raises sharer_hoarder relative to eater (-2.336 vs -2.139) — sharer_hoarder - eaters: -2.336 with the sanction, -2.139 on base rewards alone; the sanction adds -0.197
- **suggested_throw05**, optimist, basic: sanction raises sharer_hoarder relative to eater (-2.230 vs -2.139) — sharer_hoarder - eaters: -2.230 with the sanction, -2.139 on base rewards alone; the sanction adds -0.091
- **suggested_throw08**, utilitarian, basic: sanction raises sharer_hoarder relative to eater (-2.161 vs -1.964) — sharer_hoarder - eaters: -2.161 with the sanction, -1.964 on base rewards alone; the sanction adds -0.197
- **suggested_throw08**, optimist, basic: sanction raises sharer_hoarder relative to eater (-2.055 vs -1.964) — sharer_hoarder - eaters: -2.055 with the sanction, -1.964 on base rewards alone; the sanction adds -0.091
- **current**, utilitarian, basic: sanction raises altruist_eater relative to eater (-2.430 vs -2.008) — altruist_eater - eaters: -2.430 with the sanction, -2.008 on base rewards alone; the sanction adds -0.422
- **current**, optimist, basic: sanction raises altruist_eater relative to eater (-2.621 vs -2.008) — altruist_eater - eaters: -2.621 with the sanction, -2.008 on base rewards alone; the sanction adds -0.614
- **suggested_ethics_as_current**, utilitarian, basic: sanction raises altruist_eater relative to eater (-2.576 vs -2.154) — altruist_eater - eaters: -2.576 with the sanction, -2.154 on base rewards alone; the sanction adds -0.422
- **suggested_ethics_as_current**, optimist, basic: sanction raises altruist_eater relative to eater (-2.768 vs -2.154) — altruist_eater - eaters: -2.768 with the sanction, -2.154 on base rewards alone; the sanction adds -0.614
- **nonnegative**, utilitarian, basic: sanction raises altruist_eater relative to eater (-2.843 vs -2.421) — altruist_eater - eaters: -2.843 with the sanction, -2.421 on base rewards alone; the sanction adds -0.422
- **nonnegative**, optimist, basic: sanction raises altruist_eater relative to eater (-2.745 vs -2.421) — altruist_eater - eaters: -2.745 with the sanction, -2.421 on base rewards alone; the sanction adds -0.324
- **suggested**, utilitarian, basic: sanction raises altruist_eater relative to eater (-3.060 vs -2.638) — altruist_eater - eaters: -3.060 with the sanction, -2.638 on base rewards alone; the sanction adds -0.422
- **suggested**, optimist, basic: sanction raises altruist_eater relative to eater (-3.252 vs -2.638) — altruist_eater - eaters: -3.252 with the sanction, -2.638 on base rewards alone; the sanction adds -0.614
- **suggested_alive**, utilitarian, basic: sanction raises altruist_eater relative to eater (-3.147 vs -2.725) — altruist_eater - eaters: -3.147 with the sanction, -2.725 on base rewards alone; the sanction adds -0.422
- **suggested_alive**, optimist, basic: sanction raises altruist_eater relative to eater (-3.339 vs -2.725) — altruist_eater - eaters: -3.339 with the sanction, -2.725 on base rewards alone; the sanction adds -0.614
- **suggested_death5**, utilitarian, basic: sanction raises altruist_eater relative to eater (-3.931 vs -3.509) — altruist_eater - eaters: -3.931 with the sanction, -3.509 on base rewards alone; the sanction adds -0.422
- **suggested_death5**, optimist, basic: sanction raises altruist_eater relative to eater (-4.123 vs -3.509) — altruist_eater - eaters: -4.123 with the sanction, -3.509 on base rewards alone; the sanction adds -0.614
- **suggested_forage05**, utilitarian, basic: sanction raises altruist_eater relative to eater (-3.588 vs -3.166) — altruist_eater - eaters: -3.588 with the sanction, -3.166 on base rewards alone; the sanction adds -0.422
- **suggested_forage05**, optimist, basic: sanction raises altruist_eater relative to eater (-3.779 vs -3.166) — altruist_eater - eaters: -3.779 with the sanction, -3.166 on base rewards alone; the sanction adds -0.614
- **suggested_forage08**, utilitarian, basic: sanction raises altruist_eater relative to eater (-3.904 vs -3.482) — altruist_eater - eaters: -3.904 with the sanction, -3.482 on base rewards alone; the sanction adds -0.422
- **suggested_forage08**, optimist, basic: sanction raises altruist_eater relative to eater (-4.096 vs -3.482) — altruist_eater - eaters: -4.096 with the sanction, -3.482 on base rewards alone; the sanction adds -0.614
- **suggested_positive_sanction**, utilitarian, basic: sanction raises altruist_eater relative to eater (-3.060 vs -2.638) — altruist_eater - eaters: -3.060 with the sanction, -2.638 on base rewards alone; the sanction adds -0.422
- **suggested_positive_sanction**, optimist, basic: sanction raises altruist_eater relative to eater (-2.962 vs -2.638) — altruist_eater - eaters: -2.962 with the sanction, -2.638 on base rewards alone; the sanction adds -0.324
- **suggested_shaping**, utilitarian, basic: sanction raises altruist_eater relative to eater (-2.151 vs -1.729) — altruist_eater - eaters: -2.151 with the sanction, -1.729 on base rewards alone; the sanction adds -0.422
- **suggested_shaping**, optimist, basic: sanction raises altruist_eater relative to eater (-2.343 vs -1.729) — altruist_eater - eaters: -2.343 with the sanction, -1.729 on base rewards alone; the sanction adds -0.614
- **suggested_throw05**, utilitarian, basic: sanction raises altruist_eater relative to eater (-2.386 vs -1.965) — altruist_eater - eaters: -2.386 with the sanction, -1.965 on base rewards alone; the sanction adds -0.422
- **suggested_throw05**, optimist, basic: sanction raises altruist_eater relative to eater (-2.578 vs -1.965) — altruist_eater - eaters: -2.578 with the sanction, -1.965 on base rewards alone; the sanction adds -0.614
- **suggested_throw08**, utilitarian, basic: sanction raises altruist_eater relative to eater (-1.982 vs -1.560) — altruist_eater - eaters: -1.982 with the sanction, -1.560 on base rewards alone; the sanction adds -0.422
- **suggested_throw08**, optimist, basic: sanction raises altruist_eater relative to eater (-2.174 vs -1.560) — altruist_eater - eaters: -2.174 with the sanction, -1.560 on base rewards alone; the sanction adds -0.614

## Checks passed per reward function

| reward function | checks passed | failed | not checked |
|---|---|---|---|
| current | 64 | 8 | 0 |
| nonnegative | 64 | 8 | 0 |
| suggested | 64 | 8 | 0 |
| suggested_alive | 64 | 8 | 0 |
| suggested_death5 | 64 | 8 | 0 |
| suggested_ethics_as_current | 64 | 8 | 0 |
| suggested_forage05 | 64 | 8 | 0 |
| suggested_forage08 | 64 | 8 | 0 |
| suggested_positive_sanction | 64 | 8 | 0 |
| suggested_shaping | 56 | 16 | 0 |
| suggested_throw05 | 64 | 8 | 0 |
| suggested_throw08 | 64 | 8 | 0 |

## The sanction's effect on sharing agents (focal set-ups)

The focal agent's discounted return minus the eaters', with the sanction and on base rewards alone, and the sanction's effect (the difference):

| policy | reward function | agent type | with sanction | base only | effect |
|---|---|---|---|---|---|
| sharer | current | utilitarian | -1.569 | -1.288 | -0.281 |
| sharer | current | maximin | -0.544 | -1.288 | +0.743 |
| sharer | current | egalitarian | 0.334 | -1.288 | +1.622 |
| sharer | current | average | -0.593 | -1.288 | +0.695 |
| sharer | current | majoritarian | -0.596 | -1.288 | +0.692 |
| sharer | current | optimist | -1.429 | -1.288 | -0.141 |
| sharer | current | veto | 0.317 | -1.288 | +1.605 |
| sharer | suggested_ethics_as_current | utilitarian | -1.589 | -1.308 | -0.281 |
| sharer | suggested_ethics_as_current | maximin | -0.564 | -1.308 | +0.743 |
| sharer | suggested_ethics_as_current | egalitarian | 0.314 | -1.308 | +1.622 |
| sharer | suggested_ethics_as_current | average | -0.613 | -1.308 | +0.695 |
| sharer | suggested_ethics_as_current | majoritarian | -0.616 | -1.308 | +0.692 |
| sharer | suggested_ethics_as_current | optimist | -1.449 | -1.308 | -0.141 |
| sharer | suggested_ethics_as_current | veto | 0.297 | -1.308 | +1.605 |
| sharer | nonnegative | utilitarian | -1.742 | -1.461 | -0.281 |
| sharer | nonnegative | maximin | -0.718 | -1.461 | +0.743 |
| sharer | nonnegative | egalitarian | -0.713 | -1.461 | +0.748 |
| sharer | nonnegative | average | -0.766 | -1.461 | +0.695 |
| sharer | nonnegative | majoritarian | -0.769 | -1.461 | +0.692 |
| sharer | nonnegative | optimist | -1.602 | -1.461 | -0.141 |
| sharer | nonnegative | veto | -0.729 | -1.461 | +0.732 |
| sharer | suggested | utilitarian | -1.881 | -1.600 | -0.281 |
| sharer | suggested | maximin | -0.857 | -1.600 | +0.743 |
| sharer | suggested | egalitarian | 0.022 | -1.600 | +1.622 |
| sharer | suggested | average | -0.905 | -1.600 | +0.695 |
| sharer | suggested | majoritarian | -0.908 | -1.600 | +0.692 |
| sharer | suggested | optimist | -1.741 | -1.600 | -0.141 |
| sharer | suggested | veto | 0.005 | -1.600 | +1.605 |
| sharer | suggested_alive | utilitarian | -1.937 | -1.655 | -0.281 |
| sharer | suggested_alive | maximin | -0.912 | -1.655 | +0.743 |
| sharer | suggested_alive | egalitarian | -0.034 | -1.655 | +1.622 |
| sharer | suggested_alive | average | -0.961 | -1.655 | +0.695 |
| sharer | suggested_alive | majoritarian | -0.964 | -1.655 | +0.692 |
| sharer | suggested_alive | optimist | -1.797 | -1.655 | -0.141 |
| sharer | suggested_alive | veto | -0.050 | -1.655 | +1.605 |
| sharer | suggested_death5 | utilitarian | -2.437 | -2.155 | -0.281 |
| sharer | suggested_death5 | maximin | -1.412 | -2.155 | +0.743 |
| sharer | suggested_death5 | egalitarian | -0.534 | -2.155 | +1.622 |
| sharer | suggested_death5 | average | -1.461 | -2.155 | +0.695 |
| sharer | suggested_death5 | majoritarian | -1.464 | -2.155 | +0.692 |
| sharer | suggested_death5 | optimist | -2.297 | -2.155 | -0.141 |
| sharer | suggested_death5 | veto | -0.550 | -2.155 | +1.605 |
| sharer | suggested_forage05 | utilitarian | -2.233 | -1.951 | -0.281 |
| sharer | suggested_forage05 | maximin | -1.208 | -1.951 | +0.743 |
| sharer | suggested_forage05 | egalitarian | -0.330 | -1.951 | +1.622 |
| sharer | suggested_forage05 | average | -1.257 | -1.951 | +0.695 |
| sharer | suggested_forage05 | majoritarian | -1.260 | -1.951 | +0.692 |
| sharer | suggested_forage05 | optimist | -2.093 | -1.951 | -0.141 |
| sharer | suggested_forage05 | veto | -0.346 | -1.951 | +1.605 |
| sharer | suggested_forage08 | utilitarian | -2.444 | -2.162 | -0.281 |
| sharer | suggested_forage08 | maximin | -1.419 | -2.162 | +0.743 |
| sharer | suggested_forage08 | egalitarian | -0.541 | -2.162 | +1.622 |
| sharer | suggested_forage08 | average | -1.468 | -2.162 | +0.695 |
| sharer | suggested_forage08 | majoritarian | -1.471 | -2.162 | +0.692 |
| sharer | suggested_forage08 | optimist | -2.304 | -2.162 | -0.141 |
| sharer | suggested_forage08 | veto | -0.557 | -2.162 | +1.605 |
| sharer | suggested_positive_sanction | utilitarian | -1.881 | -1.600 | -0.281 |
| sharer | suggested_positive_sanction | maximin | -0.857 | -1.600 | +0.743 |
| sharer | suggested_positive_sanction | egalitarian | -0.851 | -1.600 | +0.748 |
| sharer | suggested_positive_sanction | average | -0.905 | -1.600 | +0.695 |
| sharer | suggested_positive_sanction | majoritarian | -0.908 | -1.600 | +0.692 |
| sharer | suggested_positive_sanction | optimist | -1.741 | -1.600 | -0.141 |
| sharer | suggested_positive_sanction | veto | -0.868 | -1.600 | +0.732 |
| sharer | suggested_shaping | utilitarian | -1.606 | -1.325 | -0.281 |
| sharer | suggested_shaping | maximin | -0.581 | -1.325 | +0.743 |
| sharer | suggested_shaping | egalitarian | 0.297 | -1.325 | +1.622 |
| sharer | suggested_shaping | average | -0.630 | -1.325 | +0.695 |
| sharer | suggested_shaping | majoritarian | -0.633 | -1.325 | +0.692 |
| sharer | suggested_shaping | optimist | -1.466 | -1.325 | -0.141 |
| sharer | suggested_shaping | veto | 0.280 | -1.325 | +1.605 |
| sharer | suggested_throw05 | utilitarian | -1.509 | -1.228 | -0.281 |
| sharer | suggested_throw05 | maximin | -0.485 | -1.228 | +0.743 |
| sharer | suggested_throw05 | egalitarian | 0.394 | -1.228 | +1.622 |
| sharer | suggested_throw05 | average | -0.534 | -1.228 | +0.695 |
| sharer | suggested_throw05 | majoritarian | -0.537 | -1.228 | +0.692 |
| sharer | suggested_throw05 | optimist | -1.370 | -1.228 | -0.141 |
| sharer | suggested_throw05 | veto | 0.377 | -1.228 | +1.605 |
| sharer | suggested_throw08 | utilitarian | -1.286 | -1.005 | -0.281 |
| sharer | suggested_throw08 | maximin | -0.262 | -1.005 | +0.743 |
| sharer | suggested_throw08 | egalitarian | 0.617 | -1.005 | +1.622 |
| sharer | suggested_throw08 | average | -0.311 | -1.005 | +0.695 |
| sharer | suggested_throw08 | majoritarian | -0.314 | -1.005 | +0.692 |
| sharer | suggested_throw08 | optimist | -1.146 | -1.005 | -0.141 |
| sharer | suggested_throw08 | veto | 0.600 | -1.005 | +1.605 |
| altruist | current | utilitarian | -2.703 | -2.281 | -0.422 |
| altruist | current | maximin | -1.843 | -2.281 | +0.438 |
| altruist | current | egalitarian | -0.857 | -2.281 | +1.424 |
| altruist | current | average | -1.801 | -2.281 | +0.480 |
| altruist | current | majoritarian | -1.907 | -2.281 | +0.374 |
| altruist | current | optimist | -2.861 | -2.281 | -0.580 |
| altruist | current | veto | -0.890 | -2.281 | +1.391 |
| altruist | suggested_ethics_as_current | utilitarian | -2.815 | -2.392 | -0.422 |
| altruist | suggested_ethics_as_current | maximin | -1.955 | -2.392 | +0.438 |
| altruist | suggested_ethics_as_current | egalitarian | -0.968 | -2.392 | +1.424 |
| altruist | suggested_ethics_as_current | average | -1.912 | -2.392 | +0.480 |
| altruist | suggested_ethics_as_current | majoritarian | -2.018 | -2.392 | +0.374 |
| altruist | suggested_ethics_as_current | optimist | -2.973 | -2.392 | -0.580 |
| altruist | suggested_ethics_as_current | veto | -1.001 | -2.392 | +1.391 |
| altruist | nonnegative | utilitarian | -3.117 | -2.695 | -0.422 |
| altruist | nonnegative | maximin | -1.990 | -2.695 | +0.705 |
| altruist | nonnegative | egalitarian | -2.002 | -2.695 | +0.693 |
| altruist | nonnegative | average | -2.041 | -2.695 | +0.654 |
| altruist | nonnegative | majoritarian | -2.053 | -2.695 | +0.642 |
| altruist | nonnegative | optimist | -3.008 | -2.695 | -0.313 |
| altruist | nonnegative | veto | -2.022 | -2.695 | +0.673 |
| altruist | suggested | utilitarian | -3.354 | -2.931 | -0.422 |
| altruist | suggested | maximin | -2.494 | -2.931 | +0.438 |
| altruist | suggested | egalitarian | -1.507 | -2.931 | +1.424 |
| altruist | suggested | average | -2.451 | -2.931 | +0.480 |
| altruist | suggested | majoritarian | -2.557 | -2.931 | +0.374 |
| altruist | suggested | optimist | -3.512 | -2.931 | -0.580 |
| altruist | suggested | veto | -1.540 | -2.931 | +1.391 |
| altruist | suggested_alive | utilitarian | -3.448 | -3.026 | -0.422 |
| altruist | suggested_alive | maximin | -2.588 | -3.026 | +0.438 |
| altruist | suggested_alive | egalitarian | -1.602 | -3.026 | +1.424 |
| altruist | suggested_alive | average | -2.546 | -3.026 | +0.480 |
| altruist | suggested_alive | majoritarian | -2.652 | -3.026 | +0.374 |
| altruist | suggested_alive | optimist | -3.606 | -3.026 | -0.580 |
| altruist | suggested_alive | veto | -1.635 | -3.026 | +1.391 |
| altruist | suggested_death5 | utilitarian | -4.299 | -3.876 | -0.422 |
| altruist | suggested_death5 | maximin | -3.439 | -3.876 | +0.438 |
| altruist | suggested_death5 | egalitarian | -2.452 | -3.876 | +1.424 |
| altruist | suggested_death5 | average | -3.397 | -3.876 | +0.480 |
| altruist | suggested_death5 | majoritarian | -3.503 | -3.876 | +0.374 |
| altruist | suggested_death5 | optimist | -4.457 | -3.876 | -0.580 |
| altruist | suggested_death5 | veto | -2.485 | -3.876 | +1.391 |
| altruist | suggested_forage05 | utilitarian | -3.881 | -3.459 | -0.422 |
| altruist | suggested_forage05 | maximin | -3.021 | -3.459 | +0.438 |
| altruist | suggested_forage05 | egalitarian | -2.035 | -3.459 | +1.424 |
| altruist | suggested_forage05 | average | -2.979 | -3.459 | +0.480 |
| altruist | suggested_forage05 | majoritarian | -3.085 | -3.459 | +0.374 |
| altruist | suggested_forage05 | optimist | -4.039 | -3.459 | -0.580 |
| altruist | suggested_forage05 | veto | -2.068 | -3.459 | +1.391 |
| altruist | suggested_forage08 | utilitarian | -4.198 | -3.776 | -0.422 |
| altruist | suggested_forage08 | maximin | -3.338 | -3.776 | +0.438 |
| altruist | suggested_forage08 | egalitarian | -2.352 | -3.776 | +1.424 |
| altruist | suggested_forage08 | average | -3.296 | -3.776 | +0.480 |
| altruist | suggested_forage08 | majoritarian | -3.402 | -3.776 | +0.374 |
| altruist | suggested_forage08 | optimist | -4.356 | -3.776 | -0.580 |
| altruist | suggested_forage08 | veto | -2.385 | -3.776 | +1.391 |
| altruist | suggested_positive_sanction | utilitarian | -3.354 | -2.931 | -0.422 |
| altruist | suggested_positive_sanction | maximin | -2.226 | -2.931 | +0.705 |
| altruist | suggested_positive_sanction | egalitarian | -2.238 | -2.931 | +0.693 |
| altruist | suggested_positive_sanction | average | -2.277 | -2.931 | +0.654 |
| altruist | suggested_positive_sanction | majoritarian | -2.290 | -2.931 | +0.642 |
| altruist | suggested_positive_sanction | optimist | -3.244 | -2.931 | -0.313 |
| altruist | suggested_positive_sanction | veto | -2.258 | -2.931 | +0.673 |
| altruist | suggested_shaping | utilitarian | -2.662 | -2.240 | -0.422 |
| altruist | suggested_shaping | maximin | -1.803 | -2.240 | +0.438 |
| altruist | suggested_shaping | egalitarian | -0.816 | -2.240 | +1.424 |
| altruist | suggested_shaping | average | -1.760 | -2.240 | +0.480 |
| altruist | suggested_shaping | majoritarian | -1.866 | -2.240 | +0.374 |
| altruist | suggested_shaping | optimist | -2.821 | -2.240 | -0.580 |
| altruist | suggested_shaping | veto | -0.849 | -2.240 | +1.391 |
| altruist | suggested_throw05 | utilitarian | -2.714 | -2.292 | -0.422 |
| altruist | suggested_throw05 | maximin | -1.855 | -2.292 | +0.438 |
| altruist | suggested_throw05 | egalitarian | -0.868 | -2.292 | +1.424 |
| altruist | suggested_throw05 | average | -1.812 | -2.292 | +0.480 |
| altruist | suggested_throw05 | majoritarian | -1.918 | -2.292 | +0.374 |
| altruist | suggested_throw05 | optimist | -2.873 | -2.292 | -0.580 |
| altruist | suggested_throw05 | veto | -0.901 | -2.292 | +1.391 |
| altruist | suggested_throw08 | utilitarian | -2.331 | -1.909 | -0.422 |
| altruist | suggested_throw08 | maximin | -1.471 | -1.909 | +0.438 |
| altruist | suggested_throw08 | egalitarian | -0.485 | -1.909 | +1.424 |
| altruist | suggested_throw08 | average | -1.429 | -1.909 | +0.480 |
| altruist | suggested_throw08 | majoritarian | -1.535 | -1.909 | +0.374 |
| altruist | suggested_throw08 | optimist | -2.489 | -1.909 | -0.580 |
| altruist | suggested_throw08 | veto | -0.518 | -1.909 | +1.391 |
| sharer_hoarder | current | utilitarian | -2.124 | -1.927 | -0.197 |
| sharer_hoarder | current | maximin | -1.532 | -1.927 | +0.395 |
| sharer_hoarder | current | egalitarian | -1.143 | -1.927 | +0.784 |
| sharer_hoarder | current | average | -1.600 | -1.927 | +0.327 |
| sharer_hoarder | current | majoritarian | -1.587 | -1.927 | +0.340 |
| sharer_hoarder | current | optimist | -2.018 | -1.927 | -0.091 |
| sharer_hoarder | current | veto | -1.159 | -1.927 | +0.769 |
| sharer_hoarder | suggested_ethics_as_current | utilitarian | -2.170 | -1.973 | -0.197 |
| sharer_hoarder | suggested_ethics_as_current | maximin | -1.578 | -1.973 | +0.395 |
| sharer_hoarder | suggested_ethics_as_current | egalitarian | -1.189 | -1.973 | +0.784 |
| sharer_hoarder | suggested_ethics_as_current | average | -1.646 | -1.973 | +0.327 |
| sharer_hoarder | suggested_ethics_as_current | majoritarian | -1.633 | -1.973 | +0.340 |
| sharer_hoarder | suggested_ethics_as_current | optimist | -2.064 | -1.973 | -0.091 |
| sharer_hoarder | suggested_ethics_as_current | veto | -1.205 | -1.973 | +0.769 |
| sharer_hoarder | nonnegative | utilitarian | -2.487 | -2.290 | -0.197 |
| sharer_hoarder | nonnegative | maximin | -1.895 | -2.290 | +0.395 |
| sharer_hoarder | nonnegative | egalitarian | -1.936 | -2.290 | +0.354 |
| sharer_hoarder | nonnegative | average | -1.962 | -2.290 | +0.327 |
| sharer_hoarder | nonnegative | majoritarian | -1.950 | -2.290 | +0.340 |
| sharer_hoarder | nonnegative | optimist | -2.381 | -2.290 | -0.091 |
| sharer_hoarder | nonnegative | veto | -1.951 | -2.290 | +0.339 |
| sharer_hoarder | suggested | utilitarian | -2.628 | -2.431 | -0.197 |
| sharer_hoarder | suggested | maximin | -2.036 | -2.431 | +0.395 |
| sharer_hoarder | suggested | egalitarian | -1.647 | -2.431 | +0.784 |
| sharer_hoarder | suggested | average | -2.104 | -2.431 | +0.327 |
| sharer_hoarder | suggested | majoritarian | -2.091 | -2.431 | +0.340 |
| sharer_hoarder | suggested | optimist | -2.522 | -2.431 | -0.091 |
| sharer_hoarder | suggested | veto | -1.663 | -2.431 | +0.769 |
| sharer_hoarder | suggested_alive | utilitarian | -2.685 | -2.488 | -0.197 |
| sharer_hoarder | suggested_alive | maximin | -2.093 | -2.488 | +0.395 |
| sharer_hoarder | suggested_alive | egalitarian | -1.704 | -2.488 | +0.784 |
| sharer_hoarder | suggested_alive | average | -2.160 | -2.488 | +0.327 |
| sharer_hoarder | suggested_alive | majoritarian | -2.148 | -2.488 | +0.340 |
| sharer_hoarder | suggested_alive | optimist | -2.579 | -2.488 | -0.091 |
| sharer_hoarder | suggested_alive | veto | -1.719 | -2.488 | +0.769 |
| sharer_hoarder | suggested_death5 | utilitarian | -3.194 | -2.997 | -0.197 |
| sharer_hoarder | suggested_death5 | maximin | -2.602 | -2.997 | +0.395 |
| sharer_hoarder | suggested_death5 | egalitarian | -2.213 | -2.997 | +0.784 |
| sharer_hoarder | suggested_death5 | average | -2.670 | -2.997 | +0.327 |
| sharer_hoarder | suggested_death5 | majoritarian | -2.657 | -2.997 | +0.340 |
| sharer_hoarder | suggested_death5 | optimist | -3.088 | -2.997 | -0.091 |
| sharer_hoarder | suggested_death5 | veto | -2.229 | -2.997 | +0.769 |
| sharer_hoarder | suggested_forage05 | utilitarian | -2.874 | -2.677 | -0.197 |
| sharer_hoarder | suggested_forage05 | maximin | -2.282 | -2.677 | +0.395 |
| sharer_hoarder | suggested_forage05 | egalitarian | -1.893 | -2.677 | +0.784 |
| sharer_hoarder | suggested_forage05 | average | -2.350 | -2.677 | +0.327 |
| sharer_hoarder | suggested_forage05 | majoritarian | -2.337 | -2.677 | +0.340 |
| sharer_hoarder | suggested_forage05 | optimist | -2.768 | -2.677 | -0.091 |
| sharer_hoarder | suggested_forage05 | veto | -1.909 | -2.677 | +0.769 |
| sharer_hoarder | suggested_forage08 | utilitarian | -3.022 | -2.825 | -0.197 |
| sharer_hoarder | suggested_forage08 | maximin | -2.430 | -2.825 | +0.395 |
| sharer_hoarder | suggested_forage08 | egalitarian | -2.041 | -2.825 | +0.784 |
| sharer_hoarder | suggested_forage08 | average | -2.497 | -2.825 | +0.327 |
| sharer_hoarder | suggested_forage08 | majoritarian | -2.485 | -2.825 | +0.340 |
| sharer_hoarder | suggested_forage08 | optimist | -2.916 | -2.825 | -0.091 |
| sharer_hoarder | suggested_forage08 | veto | -2.056 | -2.825 | +0.769 |
| sharer_hoarder | suggested_positive_sanction | utilitarian | -2.628 | -2.431 | -0.197 |
| sharer_hoarder | suggested_positive_sanction | maximin | -2.036 | -2.431 | +0.395 |
| sharer_hoarder | suggested_positive_sanction | egalitarian | -2.077 | -2.431 | +0.354 |
| sharer_hoarder | suggested_positive_sanction | average | -2.104 | -2.431 | +0.327 |
| sharer_hoarder | suggested_positive_sanction | majoritarian | -2.091 | -2.431 | +0.340 |
| sharer_hoarder | suggested_positive_sanction | optimist | -2.522 | -2.431 | -0.091 |
| sharer_hoarder | suggested_positive_sanction | veto | -2.093 | -2.431 | +0.339 |
| sharer_hoarder | suggested_shaping | utilitarian | -1.883 | -1.686 | -0.197 |
| sharer_hoarder | suggested_shaping | maximin | -1.291 | -1.686 | +0.395 |
| sharer_hoarder | suggested_shaping | egalitarian | -0.903 | -1.686 | +0.784 |
| sharer_hoarder | suggested_shaping | average | -1.359 | -1.686 | +0.327 |
| sharer_hoarder | suggested_shaping | majoritarian | -1.346 | -1.686 | +0.340 |
| sharer_hoarder | suggested_shaping | optimist | -1.778 | -1.686 | -0.091 |
| sharer_hoarder | suggested_shaping | veto | -0.918 | -1.686 | +0.769 |
| sharer_hoarder | suggested_throw05 | utilitarian | -2.336 | -2.139 | -0.197 |
| sharer_hoarder | suggested_throw05 | maximin | -1.744 | -2.139 | +0.395 |
| sharer_hoarder | suggested_throw05 | egalitarian | -1.355 | -2.139 | +0.784 |
| sharer_hoarder | suggested_throw05 | average | -1.812 | -2.139 | +0.327 |
| sharer_hoarder | suggested_throw05 | majoritarian | -1.799 | -2.139 | +0.340 |
| sharer_hoarder | suggested_throw05 | optimist | -2.230 | -2.139 | -0.091 |
| sharer_hoarder | suggested_throw05 | veto | -1.371 | -2.139 | +0.769 |
| sharer_hoarder | suggested_throw08 | utilitarian | -2.161 | -1.964 | -0.197 |
| sharer_hoarder | suggested_throw08 | maximin | -1.569 | -1.964 | +0.395 |
| sharer_hoarder | suggested_throw08 | egalitarian | -1.180 | -1.964 | +0.784 |
| sharer_hoarder | suggested_throw08 | average | -1.637 | -1.964 | +0.327 |
| sharer_hoarder | suggested_throw08 | majoritarian | -1.624 | -1.964 | +0.340 |
| sharer_hoarder | suggested_throw08 | optimist | -2.055 | -1.964 | -0.091 |
| sharer_hoarder | suggested_throw08 | veto | -1.195 | -1.964 | +0.769 |
| altruist_eater | current | utilitarian | -2.430 | -2.008 | -0.422 |
| altruist_eater | current | maximin | -1.454 | -2.008 | +0.554 |
| altruist_eater | current | egalitarian | -0.365 | -2.008 | +1.643 |
| altruist_eater | current | average | -1.416 | -2.008 | +0.592 |
| altruist_eater | current | majoritarian | -1.560 | -2.008 | +0.448 |
| altruist_eater | current | optimist | -2.621 | -2.008 | -0.614 |
| altruist_eater | current | veto | -0.411 | -2.008 | +1.597 |
| altruist_eater | suggested_ethics_as_current | utilitarian | -2.576 | -2.154 | -0.422 |
| altruist_eater | suggested_ethics_as_current | maximin | -1.600 | -2.154 | +0.554 |
| altruist_eater | suggested_ethics_as_current | egalitarian | -0.512 | -2.154 | +1.643 |
| altruist_eater | suggested_ethics_as_current | average | -1.563 | -2.154 | +0.592 |
| altruist_eater | suggested_ethics_as_current | majoritarian | -1.707 | -2.154 | +0.448 |
| altruist_eater | suggested_ethics_as_current | optimist | -2.768 | -2.154 | -0.614 |
| altruist_eater | suggested_ethics_as_current | veto | -0.558 | -2.154 | +1.597 |
| altruist_eater | nonnegative | utilitarian | -2.843 | -2.421 | -0.422 |
| altruist_eater | nonnegative | maximin | -1.577 | -2.421 | +0.844 |
| altruist_eater | nonnegative | egalitarian | -1.614 | -2.421 | +0.807 |
| altruist_eater | nonnegative | average | -1.643 | -2.421 | +0.778 |
| altruist_eater | nonnegative | majoritarian | -1.684 | -2.421 | +0.737 |
| altruist_eater | nonnegative | optimist | -2.745 | -2.421 | -0.324 |
| altruist_eater | nonnegative | veto | -1.640 | -2.421 | +0.781 |
| altruist_eater | suggested | utilitarian | -3.060 | -2.638 | -0.422 |
| altruist_eater | suggested | maximin | -2.084 | -2.638 | +0.554 |
| altruist_eater | suggested | egalitarian | -0.996 | -2.638 | +1.643 |
| altruist_eater | suggested | average | -2.047 | -2.638 | +0.592 |
| altruist_eater | suggested | majoritarian | -2.191 | -2.638 | +0.448 |
| altruist_eater | suggested | optimist | -3.252 | -2.638 | -0.614 |
| altruist_eater | suggested | veto | -1.042 | -2.638 | +1.597 |
| altruist_eater | suggested_alive | utilitarian | -3.147 | -2.725 | -0.422 |
| altruist_eater | suggested_alive | maximin | -2.171 | -2.725 | +0.554 |
| altruist_eater | suggested_alive | egalitarian | -1.083 | -2.725 | +1.643 |
| altruist_eater | suggested_alive | average | -2.134 | -2.725 | +0.592 |
| altruist_eater | suggested_alive | majoritarian | -2.278 | -2.725 | +0.448 |
| altruist_eater | suggested_alive | optimist | -3.339 | -2.725 | -0.614 |
| altruist_eater | suggested_alive | veto | -1.129 | -2.725 | +1.597 |
| altruist_eater | suggested_death5 | utilitarian | -3.931 | -3.509 | -0.422 |
| altruist_eater | suggested_death5 | maximin | -2.955 | -3.509 | +0.554 |
| altruist_eater | suggested_death5 | egalitarian | -1.866 | -3.509 | +1.643 |
| altruist_eater | suggested_death5 | average | -2.917 | -3.509 | +0.592 |
| altruist_eater | suggested_death5 | majoritarian | -3.061 | -3.509 | +0.448 |
| altruist_eater | suggested_death5 | optimist | -4.123 | -3.509 | -0.614 |
| altruist_eater | suggested_death5 | veto | -1.912 | -3.509 | +1.597 |
| altruist_eater | suggested_forage05 | utilitarian | -3.588 | -3.166 | -0.422 |
| altruist_eater | suggested_forage05 | maximin | -2.612 | -3.166 | +0.554 |
| altruist_eater | suggested_forage05 | egalitarian | -1.523 | -3.166 | +1.643 |
| altruist_eater | suggested_forage05 | average | -2.574 | -3.166 | +0.592 |
| altruist_eater | suggested_forage05 | majoritarian | -2.718 | -3.166 | +0.448 |
| altruist_eater | suggested_forage05 | optimist | -3.779 | -3.166 | -0.614 |
| altruist_eater | suggested_forage05 | veto | -1.569 | -3.166 | +1.597 |
| altruist_eater | suggested_forage08 | utilitarian | -3.904 | -3.482 | -0.422 |
| altruist_eater | suggested_forage08 | maximin | -2.928 | -3.482 | +0.554 |
| altruist_eater | suggested_forage08 | egalitarian | -1.840 | -3.482 | +1.643 |
| altruist_eater | suggested_forage08 | average | -2.891 | -3.482 | +0.592 |
| altruist_eater | suggested_forage08 | majoritarian | -3.035 | -3.482 | +0.448 |
| altruist_eater | suggested_forage08 | optimist | -4.096 | -3.482 | -0.614 |
| altruist_eater | suggested_forage08 | veto | -1.886 | -3.482 | +1.597 |
| altruist_eater | suggested_positive_sanction | utilitarian | -3.060 | -2.638 | -0.422 |
| altruist_eater | suggested_positive_sanction | maximin | -1.795 | -2.638 | +0.844 |
| altruist_eater | suggested_positive_sanction | egalitarian | -1.832 | -2.638 | +0.807 |
| altruist_eater | suggested_positive_sanction | average | -1.860 | -2.638 | +0.778 |
| altruist_eater | suggested_positive_sanction | majoritarian | -1.901 | -2.638 | +0.737 |
| altruist_eater | suggested_positive_sanction | optimist | -2.962 | -2.638 | -0.324 |
| altruist_eater | suggested_positive_sanction | veto | -1.857 | -2.638 | +0.781 |
| altruist_eater | suggested_shaping | utilitarian | -2.151 | -1.729 | -0.422 |
| altruist_eater | suggested_shaping | maximin | -1.175 | -1.729 | +0.554 |
| altruist_eater | suggested_shaping | egalitarian | -0.087 | -1.729 | +1.643 |
| altruist_eater | suggested_shaping | average | -1.137 | -1.729 | +0.592 |
| altruist_eater | suggested_shaping | majoritarian | -1.281 | -1.729 | +0.448 |
| altruist_eater | suggested_shaping | optimist | -2.343 | -1.729 | -0.614 |
| altruist_eater | suggested_shaping | veto | -0.132 | -1.729 | +1.597 |
| altruist_eater | suggested_throw05 | utilitarian | -2.386 | -1.965 | -0.422 |
| altruist_eater | suggested_throw05 | maximin | -1.410 | -1.965 | +0.554 |
| altruist_eater | suggested_throw05 | egalitarian | -0.322 | -1.965 | +1.643 |
| altruist_eater | suggested_throw05 | average | -1.373 | -1.965 | +0.592 |
| altruist_eater | suggested_throw05 | majoritarian | -1.517 | -1.965 | +0.448 |
| altruist_eater | suggested_throw05 | optimist | -2.578 | -1.965 | -0.614 |
| altruist_eater | suggested_throw05 | veto | -0.368 | -1.965 | +1.597 |
| altruist_eater | suggested_throw08 | utilitarian | -1.982 | -1.560 | -0.422 |
| altruist_eater | suggested_throw08 | maximin | -1.006 | -1.560 | +0.554 |
| altruist_eater | suggested_throw08 | egalitarian | 0.082 | -1.560 | +1.643 |
| altruist_eater | suggested_throw08 | average | -0.969 | -1.560 | +0.592 |
| altruist_eater | suggested_throw08 | majoritarian | -1.113 | -1.560 | +0.448 |
| altruist_eater | suggested_throw08 | optimist | -2.174 | -1.560 | -0.614 |
| altruist_eater | suggested_throw08 | veto | 0.036 | -1.560 | +1.597 |

## Discounted returns (homogeneous set-ups, mean over agent-episodes)

### basic, baseline

| reward_function | altruist | altruist_eater | eater | hoarder | passive | passive_neutral | random | sharer | sharer_hoarder |
|---|---|---|---|---|---|---|---|---|---|
| current | 4.434 | 4.561 | 6.332 | 4.210 | -2.045 | -0.698 | -1.438 | 5.733 | 4.395 |
| nonnegative | 0.207 | 0.296 | 4.179 | 2.178 | 0.000 | 0.000 | 0.143 | 2.639 | 1.261 |
| suggested | 0.186 | 0.281 | 4.122 | 2.056 | -1.371 | -0.698 | -1.097 | 2.625 | 1.238 |
| suggested_alive | 0.577 | 0.675 | 4.500 | 2.408 | -1.251 | -0.578 | -0.958 | 3.019 | 1.629 |
| suggested_death5 | 0.101 | 0.223 | 3.895 | 1.571 | -4.165 | -3.492 | -3.700 | 2.567 | 1.147 |
| suggested_ethics_as_current | 0.186 | 0.281 | 4.122 | 2.056 | -1.371 | -0.698 | -1.097 | 2.625 | 1.238 |
| suggested_forage05 | 2.438 | 2.552 | 6.332 | 4.210 | -1.371 | -0.698 | -0.884 | 4.896 | 3.534 |
| suggested_forage08 | 3.789 | 3.915 | 7.657 | 5.502 | -1.371 | -0.698 | -0.756 | 6.259 | 4.911 |
| suggested_positive_sanction | 0.186 | 0.281 | 4.122 | 2.056 | -1.371 | -0.698 | -1.097 | 2.625 | 1.238 |
| suggested_shaping | -12.637 | -12.649 | 9.866 | 8.573 | -3.640 | -2.967 | -3.016 | 0.973 | 0.531 |
| suggested_throw05 | 2.182 | 2.290 | 4.122 | 2.056 | -1.371 | -0.698 | -1.062 | 3.462 | 2.099 |
| suggested_throw08 | 3.380 | 3.495 | 4.122 | 2.056 | -1.371 | -0.698 | -1.041 | 3.964 | 2.615 |

### basic, egalitarian

| reward_function | altruist | altruist_eater | eater | hoarder | passive | passive_neutral | random | sharer | sharer_hoarder |
|---|---|---|---|---|---|---|---|---|---|
| current | 4.502 | 4.608 | 5.179 | 3.323 | -1.371 | -0.698 | -0.976 | 5.114 | 4.006 |
| nonnegative | 1.787 | 1.887 | 4.891 | 2.794 | 0.000 | 0.000 | 0.180 | 3.752 | 2.353 |
| suggested | 0.295 | 0.387 | 3.806 | 1.605 | -1.371 | -0.698 | -1.196 | 2.533 | 1.103 |
| suggested_alive | 0.687 | 0.781 | 4.183 | 1.957 | -1.251 | -0.578 | -1.056 | 2.927 | 1.493 |
| suggested_death5 | 0.210 | 0.329 | 3.579 | 1.120 | -4.165 | -3.492 | -3.799 | 2.475 | 1.011 |
| suggested_ethics_as_current | 0.254 | 0.328 | 2.970 | 1.170 | -1.371 | -0.698 | -1.224 | 2.005 | 0.850 |
| suggested_forage05 | 2.547 | 2.659 | 6.015 | 3.759 | -1.371 | -0.698 | -0.982 | 4.805 | 3.398 |
| suggested_forage08 | 3.899 | 4.021 | 7.341 | 5.051 | -1.371 | -0.698 | -0.854 | 6.168 | 4.775 |
| suggested_positive_sanction | 1.765 | 1.872 | 4.834 | 2.672 | -1.371 | -0.698 | -1.060 | 3.738 | 2.330 |
| suggested_shaping | -12.527 | -12.542 | 9.550 | 8.122 | -3.640 | -2.967 | -3.114 | 0.881 | 0.395 |
| suggested_throw05 | 2.292 | 2.396 | 3.806 | 1.605 | -1.371 | -0.698 | -1.160 | 3.370 | 1.963 |
| suggested_throw08 | 3.490 | 3.601 | 3.806 | 1.605 | -1.371 | -0.698 | -1.139 | 3.872 | 2.479 |

### basic, maximin

| reward_function | altruist | altruist_eater | eater | hoarder | passive | passive_neutral | random | sharer | sharer_hoarder |
|---|---|---|---|---|---|---|---|---|---|
| current | 5.890 | 6.024 | 6.224 | 4.403 | -1.371 | -0.698 | -0.793 | 6.365 | 5.272 |
| nonnegative | 1.862 | 1.963 | 4.907 | 2.806 | 0.000 | 0.000 | 0.227 | 3.798 | 2.391 |
| suggested | 1.683 | 1.803 | 4.850 | 2.685 | -1.371 | -0.698 | -1.013 | 3.784 | 2.368 |
| suggested_alive | 2.075 | 2.197 | 5.228 | 3.036 | -1.251 | -0.578 | -0.874 | 4.178 | 2.759 |
| suggested_death5 | 1.598 | 1.745 | 4.623 | 2.199 | -4.165 | -3.492 | -3.617 | 3.726 | 2.277 |
| suggested_ethics_as_current | 1.642 | 1.744 | 4.014 | 2.249 | -1.371 | -0.698 | -1.042 | 3.256 | 2.116 |
| suggested_forage05 | 3.935 | 4.074 | 7.059 | 4.838 | -1.371 | -0.698 | -0.800 | 6.055 | 4.664 |
| suggested_forage08 | 5.286 | 5.437 | 8.385 | 6.130 | -1.371 | -0.698 | -0.672 | 7.418 | 6.041 |
| suggested_positive_sanction | 1.841 | 1.948 | 4.850 | 2.685 | -1.371 | -0.698 | -1.013 | 3.784 | 2.368 |
| suggested_shaping | -11.139 | -11.127 | 10.594 | 9.202 | -3.640 | -2.967 | -2.932 | 2.132 | 1.660 |
| suggested_throw05 | 3.680 | 3.812 | 4.850 | 2.685 | -1.371 | -0.698 | -0.978 | 4.621 | 3.229 |
| suggested_throw08 | 4.877 | 5.017 | 4.850 | 2.685 | -1.371 | -0.698 | -0.957 | 5.123 | 3.745 |

### basic, utilitarian

| reward_function | altruist | altruist_eater | eater | hoarder | passive | passive_neutral | random | sharer | sharer_hoarder |
|---|---|---|---|---|---|---|---|---|---|
| current | 6.194 | 6.319 | 7.263 | 5.497 | -1.371 | -0.698 | -0.706 | 7.023 | 5.979 |
| nonnegative | 2.009 | 2.113 | 5.947 | 3.901 | 0.000 | 0.000 | 0.314 | 4.456 | 3.098 |
| suggested | 1.987 | 2.098 | 5.890 | 3.779 | -1.371 | -0.698 | -0.926 | 4.442 | 3.075 |
| suggested_alive | 2.379 | 2.492 | 6.267 | 4.131 | -1.251 | -0.578 | -0.787 | 4.836 | 3.466 |
| suggested_death5 | 1.902 | 2.040 | 5.663 | 3.294 | -4.165 | -3.492 | -3.530 | 4.384 | 2.984 |
| suggested_ethics_as_current | 1.946 | 2.039 | 5.054 | 3.344 | -1.371 | -0.698 | -0.955 | 3.914 | 2.823 |
| suggested_forage05 | 4.239 | 4.370 | 8.099 | 5.933 | -1.371 | -0.698 | -0.713 | 6.713 | 5.370 |
| suggested_forage08 | 5.591 | 5.732 | 9.424 | 7.225 | -1.371 | -0.698 | -0.585 | 8.076 | 6.748 |
| suggested_positive_sanction | 1.987 | 2.098 | 5.890 | 3.779 | -1.371 | -0.698 | -0.926 | 4.442 | 3.075 |
| suggested_shaping | -10.835 | -10.831 | 11.633 | 10.296 | -3.640 | -2.967 | -2.845 | 2.790 | 2.367 |
| suggested_throw05 | 3.984 | 4.107 | 5.890 | 3.779 | -1.371 | -0.698 | -0.891 | 5.279 | 3.935 |
| suggested_throw08 | 5.181 | 5.312 | 5.890 | 3.779 | -1.371 | -0.698 | -0.870 | 5.781 | 4.452 |

## What the policies do (whatever the rewards)

| scenario | setup | policy | role | survival_time | survived_share | berries_eaten | berries_carried | berries_thrown | failed_actions |
|---|---|---|---|---|---|---|---|---|---|
| basic | focal | altruist | eaters | 186.10 | 0.92 | 42.97 | 0.23 | 0.00 | 0.00 |
| basic | focal | altruist | focal | 41.45 | 0.01 | 5.58 | 0.36 | 3.48 | 0.00 |
| basic | focal | altruist_eater | eaters | 188.19 | 0.93 | 43.05 | 0.24 | 0.00 | 0.00 |
| basic | focal | altruist_eater | focal | 47.03 | 0.00 | 6.50 | 0.30 | 3.92 | 0.00 |
| basic | focal | hoarder | eaters | 168.98 | 0.81 | 37.45 | 0.21 | 0.00 | 0.00 |
| basic | focal | hoarder | focal | 154.76 | 0.70 | 25.16 | 9.93 | 0.00 | 0.00 |
| basic | focal | random | eaters | 184.30 | 0.91 | 42.75 | 0.25 | 0.00 | 0.00 |
| basic | focal | random | focal | 8.63 | 0.00 | 0.10 | 0.20 | 0.04 | 6.79 |
| basic | focal | sharer | eaters | 187.84 | 0.93 | 41.93 | 0.23 | 0.00 | 0.00 |
| basic | focal | sharer | focal | 125.90 | 0.56 | 25.94 | 0.28 | 1.33 | 0.00 |
| basic | focal | sharer_hoarder | eaters | 185.70 | 0.92 | 41.10 | 0.21 | 0.00 | 0.00 |
| basic | focal | sharer_hoarder | focal | 113.06 | 0.43 | 17.94 | 5.20 | 1.05 | 0.00 |
| basic | homogeneous | altruist | all | 189.42 | 0.93 | 0.72 | 0.50 | 40.47 | 0.00 |
| basic | homogeneous | altruist_eater | all | 192.73 | 0.95 | 0.70 | 0.24 | 41.30 | 0.00 |
| basic | homogeneous | eater | all | 171.59 | 0.83 | 38.31 | 0.25 | 0.00 | 0.00 |
| basic | homogeneous | hoarder | all | 129.23 | 0.53 | 20.73 | 7.16 | 0.00 | 0.00 |
| basic | homogeneous | passive | all | 8.00 | 0.00 | 0.00 | 0.00 | 0.00 | 8.00 |
| basic | homogeneous | passive_neutral | all | 8.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| basic | homogeneous | random | all | 9.67 | 0.00 | 0.20 | 0.27 | 0.08 | 7.47 |
| basic | homogeneous | sharer | all | 192.85 | 0.95 | 38.72 | 0.24 | 3.31 | 0.00 |
| basic | homogeneous | sharer_hoarder | all | 184.18 | 0.87 | 24.25 | 5.24 | 6.03 | 0.00 |

Full results: data/reward_checks/results.csv; every check: data/reward_checks/checks.csv.
