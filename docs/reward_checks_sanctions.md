# Reward checks: varying the ethics sanction

The scripted-policy check (docs/reward_checks.md) with every reward function in configs/rewards/ at each sanction 0, 0.1, 0.2, 0.4, 0.8, 1, 2, 5 (the reward functions named <file>_s<sanction>). The scripted policies don't depend on the rewards, so the same logs (basic, 200 validation episodes per set-up) are evaluated afresh for each.

Gamma is 0.95; a reward on day 200 is worth 3.69e-05 of one on the first day.

A sanction of s is s times a sanction of 1 at every step (every principle gives +s, -s or 0, and every aggregation combines those linearly or clips them to +/- s), so its effect on any return is linear in s: it can't change sign as the sanction grows.

## The sharer against the eaters (focal set-up)

The focal sharer's discounted return minus the eaters' at each sanction; the sanction's effect for a sanction of 1; and the sanction at which the sharer catches up with the eaters (blank: it never does, as the sanction lowers it or it isn't behind).

| base_reward_function | agent_type | sharer - eaters at s=0 | sharer - eaters at s=0.1 | sharer - eaters at s=0.2 | sharer - eaters at s=0.4 | sharer - eaters at s=0.8 | sharer - eaters at s=1 | sharer - eaters at s=2 | sharer - eaters at s=5 | effect_per_unit | break_even_sanction |
|---|---|---|---|---|---|---|---|---|---|---|---|
| current | average | -1.288 | -1.114 | -0.940 | -0.593 | 0.101 | 0.449 | 2.185 | 7.394 | 1.736 | 0.742 |
| current | egalitarian | -1.288 | -0.882 | -0.477 | 0.334 | 1.956 | 2.766 | 6.820 | 18.982 | 4.054 | 0.318 |
| current | majoritarian | -1.288 | -1.115 | -0.942 | -0.596 | 0.095 | 0.441 | 2.170 | 7.356 | 1.729 | 0.745 |
| current | maximin | -1.288 | -1.102 | -0.916 | -0.544 | 0.199 | 0.571 | 2.429 | 8.004 | 1.858 | 0.693 |
| current | optimist | -1.288 | -1.323 | -1.358 | -1.429 | -1.571 | -1.641 | -1.995 | -3.056 | -0.354 |  |
| current | utilitarian | -1.288 | -1.358 | -1.428 | -1.569 | -1.850 | -1.991 | -2.694 | -4.804 | -0.703 |  |
| current | veto | -1.288 | -0.886 | -0.485 | 0.317 | 1.922 | 2.725 | 6.737 | 18.775 | 4.012 | 0.321 |
| suggested | average | -1.600 | -1.426 | -1.253 | -0.905 | -0.211 | 0.136 | 1.873 | 7.082 | 1.736 | 0.921 |
| suggested | egalitarian | -1.600 | -1.194 | -0.789 | 0.022 | 1.643 | 2.454 | 6.508 | 18.670 | 4.054 | 0.395 |
| suggested | majoritarian | -1.600 | -1.427 | -1.254 | -0.908 | -0.217 | 0.129 | 1.858 | 7.044 | 1.729 | 0.925 |
| suggested | maximin | -1.600 | -1.414 | -1.228 | -0.857 | -0.113 | 0.258 | 2.117 | 7.691 | 1.858 | 0.861 |
| suggested | optimist | -1.600 | -1.635 | -1.671 | -1.741 | -1.883 | -1.954 | -2.307 | -3.368 | -0.354 |  |
| suggested | utilitarian | -1.600 | -1.670 | -1.740 | -1.881 | -2.162 | -2.303 | -3.006 | -5.116 | -0.703 |  |
| suggested | veto | -1.600 | -1.199 | -0.797 | 0.005 | 1.610 | 2.413 | 6.425 | 18.463 | 4.012 | 0.399 |
| suggested_alive | average | -1.655 | -1.482 | -1.308 | -0.961 | -0.266 | 0.081 | 1.817 | 7.026 | 1.736 | 0.953 |
| suggested_alive | egalitarian | -1.655 | -1.250 | -0.845 | -0.034 | 1.588 | 2.399 | 6.453 | 18.615 | 4.054 | 0.408 |
| suggested_alive | majoritarian | -1.655 | -1.482 | -1.310 | -0.964 | -0.272 | 0.073 | 1.802 | 6.988 | 1.729 | 0.958 |
| suggested_alive | maximin | -1.655 | -1.470 | -1.284 | -0.912 | -0.169 | 0.203 | 2.061 | 7.636 | 1.858 | 0.891 |
| suggested_alive | optimist | -1.655 | -1.691 | -1.726 | -1.797 | -1.938 | -2.009 | -2.363 | -3.424 | -0.354 |  |
| suggested_alive | utilitarian | -1.655 | -1.726 | -1.796 | -1.937 | -2.218 | -2.359 | -3.062 | -5.172 | -0.703 |  |
| suggested_alive | veto | -1.655 | -1.254 | -0.853 | -0.050 | 1.555 | 2.357 | 6.370 | 18.407 | 4.012 | 0.413 |
| suggested_death5 | average | -2.155 | -1.982 | -1.808 | -1.461 | -0.766 | -0.419 | 1.317 | 6.526 | 1.736 | 1.241 |
| suggested_death5 | egalitarian | -2.155 | -1.750 | -1.345 | -0.534 | 1.088 | 1.899 | 5.953 | 18.115 | 4.054 | 0.532 |
| suggested_death5 | majoritarian | -2.155 | -1.983 | -1.810 | -1.464 | -0.772 | -0.427 | 1.302 | 6.488 | 1.729 | 1.247 |
| suggested_death5 | maximin | -2.155 | -1.970 | -1.784 | -1.412 | -0.669 | -0.297 | 1.561 | 7.136 | 1.858 | 1.160 |
| suggested_death5 | optimist | -2.155 | -2.191 | -2.226 | -2.297 | -2.438 | -2.509 | -2.863 | -3.924 | -0.354 |  |
| suggested_death5 | utilitarian | -2.155 | -2.226 | -2.296 | -2.437 | -2.718 | -2.859 | -3.562 | -5.672 | -0.703 |  |
| suggested_death5 | veto | -2.155 | -1.754 | -1.353 | -0.550 | 1.055 | 1.857 | 5.870 | 17.907 | 4.012 | 0.537 |
| suggested_ethics_as_current | average | -1.308 | -1.134 | -0.960 | -0.613 | 0.081 | 0.429 | 2.165 | 7.374 | 1.736 | 0.753 |
| suggested_ethics_as_current | egalitarian | -1.308 | -0.902 | -0.497 | 0.314 | 1.936 | 2.746 | 6.800 | 18.962 | 4.054 | 0.323 |
| suggested_ethics_as_current | majoritarian | -1.308 | -1.135 | -0.962 | -0.616 | 0.075 | 0.421 | 2.150 | 7.336 | 1.729 | 0.756 |
| suggested_ethics_as_current | maximin | -1.308 | -1.122 | -0.936 | -0.564 | 0.179 | 0.551 | 2.409 | 7.983 | 1.858 | 0.704 |
| suggested_ethics_as_current | optimist | -1.308 | -1.343 | -1.378 | -1.449 | -1.591 | -1.661 | -2.015 | -3.076 | -0.354 |  |
| suggested_ethics_as_current | utilitarian | -1.308 | -1.378 | -1.448 | -1.589 | -1.870 | -2.011 | -2.714 | -4.824 | -0.703 |  |
| suggested_ethics_as_current | veto | -1.308 | -0.906 | -0.505 | 0.297 | 1.902 | 2.705 | 6.717 | 18.755 | 4.012 | 0.326 |
| suggested_forage05 | average | -1.951 | -1.778 | -1.604 | -1.257 | -0.562 | -0.215 | 1.521 | 6.730 | 1.736 | 1.124 |
| suggested_forage05 | egalitarian | -1.951 | -1.546 | -1.141 | -0.330 | 1.292 | 2.103 | 6.157 | 18.319 | 4.054 | 0.481 |
| suggested_forage05 | majoritarian | -1.951 | -1.779 | -1.606 | -1.260 | -0.568 | -0.223 | 1.506 | 6.692 | 1.729 | 1.129 |
| suggested_forage05 | maximin | -1.951 | -1.766 | -1.580 | -1.208 | -0.465 | -0.093 | 1.765 | 7.340 | 1.858 | 1.050 |
| suggested_forage05 | optimist | -1.951 | -1.987 | -2.022 | -2.093 | -2.234 | -2.305 | -2.659 | -3.720 | -0.354 |  |
| suggested_forage05 | utilitarian | -1.951 | -2.022 | -2.092 | -2.233 | -2.514 | -2.655 | -3.358 | -5.468 | -0.703 |  |
| suggested_forage05 | veto | -1.951 | -1.550 | -1.149 | -0.346 | 1.259 | 2.061 | 6.074 | 18.111 | 4.012 | 0.486 |
| suggested_forage08 | average | -2.162 | -1.989 | -1.815 | -1.468 | -0.773 | -0.426 | 1.310 | 6.519 | 1.736 | 1.245 |
| suggested_forage08 | egalitarian | -2.162 | -1.757 | -1.352 | -0.541 | 1.081 | 1.892 | 5.946 | 18.108 | 4.054 | 0.533 |
| suggested_forage08 | majoritarian | -2.162 | -1.990 | -1.817 | -1.471 | -0.779 | -0.434 | 1.295 | 6.481 | 1.729 | 1.251 |
| suggested_forage08 | maximin | -2.162 | -1.977 | -1.791 | -1.419 | -0.676 | -0.304 | 1.554 | 7.129 | 1.858 | 1.164 |
| suggested_forage08 | optimist | -2.162 | -2.198 | -2.233 | -2.304 | -2.445 | -2.516 | -2.870 | -3.931 | -0.354 |  |
| suggested_forage08 | utilitarian | -2.162 | -2.233 | -2.303 | -2.444 | -2.725 | -2.866 | -3.569 | -5.679 | -0.703 |  |
| suggested_forage08 | veto | -2.162 | -1.761 | -1.360 | -0.557 | 1.048 | 1.850 | 5.863 | 17.900 | 4.012 | 0.539 |
| suggested_shaping | average | -1.325 | -1.151 | -0.977 | -0.630 | 0.064 | 0.412 | 2.148 | 7.357 | 1.736 | 0.763 |
| suggested_shaping | egalitarian | -1.325 | -0.919 | -0.514 | 0.297 | 1.919 | 2.729 | 6.783 | 18.945 | 4.054 | 0.327 |
| suggested_shaping | majoritarian | -1.325 | -1.152 | -0.979 | -0.633 | 0.058 | 0.404 | 2.133 | 7.319 | 1.729 | 0.766 |
| suggested_shaping | maximin | -1.325 | -1.139 | -0.953 | -0.581 | 0.162 | 0.534 | 2.392 | 7.967 | 1.858 | 0.713 |
| suggested_shaping | optimist | -1.325 | -1.360 | -1.395 | -1.466 | -1.608 | -1.678 | -2.032 | -3.093 | -0.354 |  |
| suggested_shaping | utilitarian | -1.325 | -1.395 | -1.465 | -1.606 | -1.887 | -2.028 | -2.731 | -4.841 | -0.703 |  |
| suggested_shaping | veto | -1.325 | -0.923 | -0.522 | 0.280 | 1.885 | 2.688 | 6.700 | 18.738 | 4.012 | 0.330 |
| suggested_throw05 | average | -1.228 | -1.054 | -0.881 | -0.534 | 0.161 | 0.508 | 2.245 | 7.453 | 1.736 | 0.707 |
| suggested_throw05 | egalitarian | -1.228 | -0.823 | -0.417 | 0.394 | 2.015 | 2.826 | 6.880 | 19.042 | 4.054 | 0.303 |
| suggested_throw05 | majoritarian | -1.228 | -1.055 | -0.882 | -0.537 | 0.155 | 0.501 | 2.229 | 7.416 | 1.729 | 0.710 |
| suggested_throw05 | maximin | -1.228 | -1.042 | -0.856 | -0.485 | 0.259 | 0.630 | 2.488 | 8.063 | 1.858 | 0.661 |
| suggested_throw05 | optimist | -1.228 | -1.263 | -1.299 | -1.370 | -1.511 | -1.582 | -1.935 | -2.997 | -0.354 |  |
| suggested_throw05 | utilitarian | -1.228 | -1.298 | -1.369 | -1.509 | -1.791 | -1.931 | -2.635 | -4.745 | -0.703 |  |
| suggested_throw05 | veto | -1.228 | -0.827 | -0.426 | 0.377 | 1.982 | 2.784 | 6.797 | 18.834 | 4.012 | 0.306 |
| suggested_throw08 | average | -1.005 | -0.831 | -0.658 | -0.311 | 0.384 | 0.731 | 2.468 | 7.676 | 1.736 | 0.579 |
| suggested_throw08 | egalitarian | -1.005 | -0.600 | -0.194 | 0.617 | 2.238 | 3.049 | 7.103 | 19.265 | 4.054 | 0.248 |
| suggested_throw08 | majoritarian | -1.005 | -0.832 | -0.659 | -0.314 | 0.378 | 0.724 | 2.453 | 7.639 | 1.729 | 0.581 |
| suggested_throw08 | maximin | -1.005 | -0.819 | -0.633 | -0.262 | 0.482 | 0.853 | 2.711 | 8.286 | 1.858 | 0.541 |
| suggested_throw08 | optimist | -1.005 | -1.040 | -1.076 | -1.146 | -1.288 | -1.359 | -1.712 | -2.773 | -0.354 |  |
| suggested_throw08 | utilitarian | -1.005 | -1.075 | -1.146 | -1.286 | -1.568 | -1.708 | -2.412 | -4.522 | -0.703 |  |
| suggested_throw08 | veto | -1.005 | -0.604 | -0.203 | 0.600 | 2.205 | 3.007 | 7.020 | 19.057 | 4.012 | 0.250 |

## Failed checks by sanction

Failed checks (reward functions x agent types) at each sanction:

| check | s=0 | s=0.1 | s=0.2 | s=0.4 | s=0.8 | s=1 | s=2 | s=5 |
|---|---|---|---|---|---|---|---|---|
| sanction raises sharer relative to eater | 0 | 20 | 20 | 20 | 20 | 20 | 20 | 20 |
| surviving > dying early (by 1 sd) | 8 | 8 | 8 | 8 | 9 | 12 | 21 | 21 |

By agent type:

| check | average | baseline | egalitarian | majoritarian | maximin | optimist | utilitarian | veto |
|---|---|---|---|---|---|---|---|---|
| sanction raises sharer relative to eater | 0 | 0 | 0 | 0 | 0 | 70 | 70 | 0 |
| surviving > dying early (by 1 sd) | 6 | 8 | 30 | 6 | 6 | 4 | 5 | 30 |

## Checks passed per reward function

| reward function | checks passed | failed | not checked |
|---|---|---|---|
| current_s0 | 42 | 0 | 7 |
| current_s0.1 | 47 | 2 | 0 |
| current_s0.2 | 47 | 2 | 0 |
| current_s0.4 | 47 | 2 | 0 |
| current_s0.8 | 47 | 2 | 0 |
| current_s1 | 47 | 2 | 0 |
| current_s2 | 45 | 4 | 0 |
| current_s5 | 45 | 4 | 0 |
| suggested_alive_s0 | 42 | 0 | 7 |
| suggested_alive_s0.1 | 47 | 2 | 0 |
| suggested_alive_s0.2 | 47 | 2 | 0 |
| suggested_alive_s0.4 | 47 | 2 | 0 |
| suggested_alive_s0.8 | 47 | 2 | 0 |
| suggested_alive_s1 | 45 | 4 | 0 |
| suggested_alive_s2 | 45 | 4 | 0 |
| suggested_alive_s5 | 45 | 4 | 0 |
| suggested_death5_s0 | 42 | 0 | 7 |
| suggested_death5_s0.1 | 47 | 2 | 0 |
| suggested_death5_s0.2 | 47 | 2 | 0 |
| suggested_death5_s0.4 | 47 | 2 | 0 |
| suggested_death5_s0.8 | 47 | 2 | 0 |
| suggested_death5_s1 | 47 | 2 | 0 |
| suggested_death5_s2 | 45 | 4 | 0 |
| suggested_death5_s5 | 45 | 4 | 0 |
| suggested_ethics_as_current_s0 | 42 | 0 | 7 |
| suggested_ethics_as_current_s0.1 | 47 | 2 | 0 |
| suggested_ethics_as_current_s0.2 | 47 | 2 | 0 |
| suggested_ethics_as_current_s0.4 | 47 | 2 | 0 |
| suggested_ethics_as_current_s0.8 | 45 | 4 | 0 |
| suggested_ethics_as_current_s1 | 45 | 4 | 0 |
| suggested_ethics_as_current_s2 | 45 | 4 | 0 |
| suggested_ethics_as_current_s5 | 45 | 4 | 0 |
| suggested_forage05_s0 | 42 | 0 | 7 |
| suggested_forage05_s0.1 | 47 | 2 | 0 |
| suggested_forage05_s0.2 | 47 | 2 | 0 |
| suggested_forage05_s0.4 | 47 | 2 | 0 |
| suggested_forage05_s0.8 | 47 | 2 | 0 |
| suggested_forage05_s1 | 47 | 2 | 0 |
| suggested_forage05_s2 | 45 | 4 | 0 |
| suggested_forage05_s5 | 45 | 4 | 0 |
| suggested_forage08_s0 | 42 | 0 | 7 |
| suggested_forage08_s0.1 | 47 | 2 | 0 |
| suggested_forage08_s0.2 | 47 | 2 | 0 |
| suggested_forage08_s0.4 | 47 | 2 | 0 |
| suggested_forage08_s0.8 | 47 | 2 | 0 |
| suggested_forage08_s1 | 47 | 2 | 0 |
| suggested_forage08_s2 | 45 | 4 | 0 |
| suggested_forage08_s5 | 45 | 4 | 0 |
| suggested_s0 | 42 | 0 | 7 |
| suggested_s0.1 | 47 | 2 | 0 |
| suggested_s0.2 | 47 | 2 | 0 |
| suggested_s0.4 | 47 | 2 | 0 |
| suggested_s0.8 | 47 | 2 | 0 |
| suggested_s1 | 45 | 4 | 0 |
| suggested_s2 | 45 | 4 | 0 |
| suggested_s5 | 45 | 4 | 0 |
| suggested_shaping_s0 | 34 | 8 | 7 |
| suggested_shaping_s0.1 | 39 | 10 | 0 |
| suggested_shaping_s0.2 | 39 | 10 | 0 |
| suggested_shaping_s0.4 | 39 | 10 | 0 |
| suggested_shaping_s0.8 | 40 | 9 | 0 |
| suggested_shaping_s1 | 41 | 8 | 0 |
| suggested_shaping_s2 | 44 | 5 | 0 |
| suggested_shaping_s5 | 44 | 5 | 0 |
| suggested_throw05_s0 | 42 | 0 | 7 |
| suggested_throw05_s0.1 | 47 | 2 | 0 |
| suggested_throw05_s0.2 | 47 | 2 | 0 |
| suggested_throw05_s0.4 | 47 | 2 | 0 |
| suggested_throw05_s0.8 | 47 | 2 | 0 |
| suggested_throw05_s1 | 47 | 2 | 0 |
| suggested_throw05_s2 | 45 | 4 | 0 |
| suggested_throw05_s5 | 45 | 4 | 0 |
| suggested_throw08_s0 | 42 | 0 | 7 |
| suggested_throw08_s0.1 | 47 | 2 | 0 |
| suggested_throw08_s0.2 | 47 | 2 | 0 |
| suggested_throw08_s0.4 | 47 | 2 | 0 |
| suggested_throw08_s0.8 | 47 | 2 | 0 |
| suggested_throw08_s1 | 47 | 2 | 0 |
| suggested_throw08_s2 | 45 | 4 | 0 |
| suggested_throw08_s5 | 45 | 4 | 0 |

Full results: data/reward_checks/sanctions/results.csv, checks.csv and sweep.csv.
