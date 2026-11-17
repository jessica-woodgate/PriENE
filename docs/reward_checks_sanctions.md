# Reward checks: varying the ethics sanction

The scripted-policy check (docs/reward_checks.md) with every reward function in configs/rewards/ at each sanction 0, 0.1, 0.2, 0.4, 0.8, 1, 2, 5 (the reward functions named <file>_s<sanction>). The scripted policies don't depend on the rewards, so the same logs (basic, 200 validation episodes per set-up) are evaluated afresh for each.

Gamma is 0.95; a reward on day 200 is worth 3.69e-05 of one on the first day.

A sanction of s is s times a sanction of 1 at every step (every principle gives +s, -s or 0, and every aggregation combines those linearly or clips them to +/- s), so its effect on any return is linear in s: it can't change sign as the sanction grows.

## Sharing agents against the eaters (focal set-ups)

For each sharing policy as the focal agent among eaters: its discounted return minus the eaters' at each sanction; the sanction's effect for a sanction of 1; and the sanction at which it catches up with the eaters (blank: it never does, as the sanction lowers it or it isn't behind).

| policy | base_reward_function | agent_type | focal - eaters at s=0 | focal - eaters at s=0.1 | focal - eaters at s=0.2 | focal - eaters at s=0.4 | focal - eaters at s=0.8 | focal - eaters at s=1 | focal - eaters at s=2 | focal - eaters at s=5 | effect_per_unit | break_even_sanction |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| altruist | current | average | -2.281 | -2.161 | -2.041 | -1.801 | -1.321 | -1.081 | 0.118 | 3.718 | 1.200 | 1.901 |
| altruist | current | egalitarian | -2.281 | -1.925 | -1.569 | -0.857 | 0.567 | 1.279 | 4.840 | 15.521 | 3.560 | 0.641 |
| altruist | current | majoritarian | -2.281 | -2.188 | -2.094 | -1.907 | -1.533 | -1.346 | -0.412 | 2.393 | 0.935 | 2.440 |
| altruist | current | maximin | -2.281 | -2.172 | -2.062 | -1.843 | -1.406 | -1.187 | -0.093 | 3.190 | 1.094 | 2.085 |
| altruist | current | optimist | -2.281 | -2.426 | -2.571 | -2.861 | -3.442 | -3.732 | -5.183 | -9.535 | -1.451 |  |
| altruist | current | utilitarian | -2.281 | -2.387 | -2.492 | -2.703 | -3.125 | -3.337 | -4.392 | -7.559 | -1.056 |  |
| altruist | current | veto | -2.281 | -1.933 | -1.586 | -0.890 | 0.501 | 1.197 | 4.674 | 15.107 | 3.478 | 0.656 |
| altruist | nonnegative | average | -2.695 | -2.532 | -2.368 | -2.041 | -1.387 | -1.060 | 0.575 | 5.479 | 1.635 | 1.648 |
| altruist | nonnegative | egalitarian | -2.695 | -2.522 | -2.348 | -2.002 | -1.309 | -0.962 | 0.770 | 5.968 | 1.733 | 1.555 |
| altruist | nonnegative | majoritarian | -2.695 | -2.535 | -2.374 | -2.053 | -1.412 | -1.091 | 0.513 | 5.324 | 1.604 | 1.680 |
| altruist | nonnegative | maximin | -2.695 | -2.519 | -2.342 | -1.990 | -1.284 | -0.932 | 0.832 | 6.122 | 1.763 | 1.528 |
| altruist | nonnegative | optimist | -2.695 | -2.773 | -2.851 | -3.008 | -3.320 | -3.477 | -4.258 | -6.603 | -0.782 |  |
| altruist | nonnegative | utilitarian | -2.695 | -2.801 | -2.906 | -3.117 | -3.539 | -3.751 | -4.806 | -7.973 | -1.056 |  |
| altruist | nonnegative | veto | -2.695 | -2.527 | -2.358 | -2.022 | -1.349 | -1.012 | 0.670 | 5.718 | 1.683 | 1.602 |
| altruist | suggested | average | -2.931 | -2.811 | -2.691 | -2.451 | -1.972 | -1.732 | -0.532 | 3.067 | 1.200 | 2.443 |
| altruist | suggested | egalitarian | -2.931 | -2.575 | -2.219 | -1.507 | -0.083 | 0.629 | 4.190 | 14.871 | 3.560 | 0.823 |
| altruist | suggested | majoritarian | -2.931 | -2.838 | -2.744 | -2.557 | -2.183 | -1.997 | -1.062 | 1.742 | 0.935 | 3.136 |
| altruist | suggested | maximin | -2.931 | -2.822 | -2.712 | -2.494 | -2.056 | -1.837 | -0.743 | 2.540 | 1.094 | 2.679 |
| altruist | suggested | optimist | -2.931 | -3.076 | -3.221 | -3.512 | -4.092 | -4.382 | -5.833 | -10.185 | -1.451 |  |
| altruist | suggested | utilitarian | -2.931 | -3.037 | -3.142 | -3.354 | -3.776 | -3.987 | -5.042 | -8.209 | -1.056 |  |
| altruist | suggested | veto | -2.931 | -2.584 | -2.236 | -1.540 | -0.149 | 0.546 | 4.024 | 14.457 | 3.478 | 0.843 |
| altruist | suggested_alive | average | -3.026 | -2.906 | -2.786 | -2.546 | -2.066 | -1.826 | -0.626 | 2.973 | 1.200 | 2.522 |
| altruist | suggested_alive | egalitarian | -3.026 | -2.670 | -2.314 | -1.602 | -0.177 | 0.535 | 4.095 | 14.777 | 3.560 | 0.850 |
| altruist | suggested_alive | majoritarian | -3.026 | -2.932 | -2.839 | -2.652 | -2.278 | -2.091 | -1.156 | 1.648 | 0.935 | 3.237 |
| altruist | suggested_alive | maximin | -3.026 | -2.916 | -2.807 | -2.588 | -2.150 | -1.932 | -0.837 | 2.445 | 1.094 | 2.765 |
| altruist | suggested_alive | optimist | -3.026 | -3.171 | -3.316 | -3.606 | -4.186 | -4.477 | -5.927 | -10.280 | -1.451 |  |
| altruist | suggested_alive | utilitarian | -3.026 | -3.131 | -3.237 | -3.448 | -3.870 | -4.081 | -5.137 | -8.303 | -1.056 |  |
| altruist | suggested_alive | veto | -3.026 | -2.678 | -2.330 | -1.635 | -0.244 | 0.452 | 3.929 | 14.362 | 3.478 | 0.870 |
| altruist | suggested_death5 | average | -3.876 | -3.757 | -3.637 | -3.397 | -2.917 | -2.677 | -1.477 | 2.122 | 1.200 | 3.231 |
| altruist | suggested_death5 | egalitarian | -3.876 | -3.520 | -3.164 | -2.452 | -1.028 | -0.316 | 3.244 | 13.926 | 3.560 | 1.089 |
| altruist | suggested_death5 | majoritarian | -3.876 | -3.783 | -3.690 | -3.503 | -3.129 | -2.942 | -2.007 | 0.797 | 0.935 | 4.147 |
| altruist | suggested_death5 | maximin | -3.876 | -3.767 | -3.658 | -3.439 | -3.001 | -2.782 | -1.688 | 1.595 | 1.094 | 3.543 |
| altruist | suggested_death5 | optimist | -3.876 | -4.022 | -4.167 | -4.457 | -5.037 | -5.327 | -6.778 | -11.130 | -1.451 |  |
| altruist | suggested_death5 | utilitarian | -3.876 | -3.982 | -4.088 | -4.299 | -4.721 | -4.932 | -5.988 | -9.154 | -1.056 |  |
| altruist | suggested_death5 | veto | -3.876 | -3.529 | -3.181 | -2.485 | -1.094 | -0.399 | 3.079 | 13.511 | 3.478 | 1.115 |
| altruist | suggested_ethics_as_current | average | -2.392 | -2.272 | -2.152 | -1.912 | -1.433 | -1.193 | 0.007 | 3.606 | 1.200 | 1.994 |
| altruist | suggested_ethics_as_current | egalitarian | -2.392 | -2.036 | -1.680 | -0.968 | 0.456 | 1.168 | 4.729 | 15.410 | 3.560 | 0.672 |
| altruist | suggested_ethics_as_current | majoritarian | -2.392 | -2.299 | -2.205 | -2.018 | -1.645 | -1.458 | -0.523 | 2.281 | 0.935 | 2.559 |
| altruist | suggested_ethics_as_current | maximin | -2.392 | -2.283 | -2.173 | -1.955 | -1.517 | -1.298 | -0.204 | 3.079 | 1.094 | 2.186 |
| altruist | suggested_ethics_as_current | optimist | -2.392 | -2.537 | -2.682 | -2.973 | -3.553 | -3.843 | -5.294 | -9.646 | -1.451 |  |
| altruist | suggested_ethics_as_current | utilitarian | -2.392 | -2.498 | -2.603 | -2.815 | -3.237 | -3.448 | -4.503 | -7.670 | -1.056 |  |
| altruist | suggested_ethics_as_current | veto | -2.392 | -2.045 | -1.697 | -1.001 | 0.390 | 1.085 | 4.563 | 14.996 | 3.478 | 0.688 |
| altruist | suggested_forage05 | average | -3.459 | -3.339 | -3.219 | -2.979 | -2.499 | -2.259 | -1.060 | 2.540 | 1.200 | 2.883 |
| altruist | suggested_forage05 | egalitarian | -3.459 | -3.103 | -2.747 | -2.035 | -0.611 | 0.101 | 3.662 | 14.343 | 3.560 | 0.972 |
| altruist | suggested_forage05 | majoritarian | -3.459 | -3.366 | -3.272 | -3.085 | -2.711 | -2.524 | -1.590 | 1.215 | 0.935 | 3.700 |
| altruist | suggested_forage05 | maximin | -3.459 | -3.350 | -3.240 | -3.021 | -2.584 | -2.365 | -1.271 | 2.012 | 1.094 | 3.161 |
| altruist | suggested_forage05 | optimist | -3.459 | -3.604 | -3.749 | -4.039 | -4.620 | -4.910 | -6.361 | -10.713 | -1.451 |  |
| altruist | suggested_forage05 | utilitarian | -3.459 | -3.565 | -3.670 | -3.881 | -4.303 | -4.515 | -5.570 | -8.737 | -1.056 |  |
| altruist | suggested_forage05 | veto | -3.459 | -3.111 | -2.764 | -2.068 | -0.677 | 0.019 | 3.496 | 13.929 | 3.478 | 0.995 |
| altruist | suggested_forage08 | average | -3.776 | -3.656 | -3.536 | -3.296 | -2.816 | -2.576 | -1.376 | 2.223 | 1.200 | 3.147 |
| altruist | suggested_forage08 | egalitarian | -3.776 | -3.420 | -3.064 | -2.352 | -0.927 | -0.215 | 3.345 | 14.027 | 3.560 | 1.060 |
| altruist | suggested_forage08 | majoritarian | -3.776 | -3.682 | -3.589 | -3.402 | -3.028 | -2.841 | -1.906 | 0.898 | 0.935 | 4.039 |
| altruist | suggested_forage08 | maximin | -3.776 | -3.666 | -3.557 | -3.338 | -2.900 | -2.682 | -1.587 | 1.695 | 1.094 | 3.451 |
| altruist | suggested_forage08 | optimist | -3.776 | -3.921 | -4.066 | -4.356 | -4.936 | -5.226 | -6.677 | -11.030 | -1.451 |  |
| altruist | suggested_forage08 | utilitarian | -3.776 | -3.881 | -3.987 | -4.198 | -4.620 | -4.831 | -5.887 | -9.053 | -1.056 |  |
| altruist | suggested_forage08 | veto | -3.776 | -3.428 | -3.080 | -2.385 | -0.994 | -0.298 | 3.179 | 13.612 | 3.478 | 1.086 |
| altruist | suggested_positive_sanction | average | -2.931 | -2.768 | -2.604 | -2.277 | -1.623 | -1.296 | 0.338 | 5.243 | 1.635 | 1.793 |
| altruist | suggested_positive_sanction | egalitarian | -2.931 | -2.758 | -2.585 | -2.238 | -1.545 | -1.199 | 0.534 | 5.732 | 1.733 | 1.692 |
| altruist | suggested_positive_sanction | majoritarian | -2.931 | -2.771 | -2.611 | -2.290 | -1.648 | -1.327 | 0.276 | 5.088 | 1.604 | 1.828 |
| altruist | suggested_positive_sanction | maximin | -2.931 | -2.755 | -2.579 | -2.226 | -1.521 | -1.168 | 0.595 | 5.885 | 1.763 | 1.662 |
| altruist | suggested_positive_sanction | optimist | -2.931 | -3.009 | -3.088 | -3.244 | -3.557 | -3.713 | -4.495 | -6.840 | -0.782 |  |
| altruist | suggested_positive_sanction | utilitarian | -2.931 | -3.037 | -3.142 | -3.354 | -3.776 | -3.987 | -5.042 | -8.209 | -1.056 |  |
| altruist | suggested_positive_sanction | veto | -2.931 | -2.763 | -2.595 | -2.258 | -1.585 | -1.249 | 0.434 | 5.481 | 1.683 | 1.742 |
| altruist | suggested_shaping | average | -2.240 | -2.120 | -2.000 | -1.760 | -1.280 | -1.040 | 0.159 | 3.758 | 1.200 | 1.867 |
| altruist | suggested_shaping | egalitarian | -2.240 | -1.884 | -1.528 | -0.816 | 0.608 | 1.320 | 4.881 | 15.562 | 3.560 | 0.629 |
| altruist | suggested_shaping | majoritarian | -2.240 | -2.147 | -2.053 | -1.866 | -1.492 | -1.305 | -0.371 | 2.434 | 0.935 | 2.397 |
| altruist | suggested_shaping | maximin | -2.240 | -2.131 | -2.021 | -1.803 | -1.365 | -1.146 | -0.052 | 3.231 | 1.094 | 2.047 |
| altruist | suggested_shaping | optimist | -2.240 | -2.385 | -2.530 | -2.821 | -3.401 | -3.691 | -5.142 | -9.494 | -1.451 |  |
| altruist | suggested_shaping | utilitarian | -2.240 | -2.346 | -2.451 | -2.662 | -3.085 | -3.296 | -4.351 | -7.518 | -1.056 |  |
| altruist | suggested_shaping | veto | -2.240 | -1.892 | -1.545 | -0.849 | 0.542 | 1.237 | 4.715 | 15.148 | 3.478 | 0.644 |
| altruist | suggested_throw05 | average | -2.292 | -2.172 | -2.052 | -1.812 | -1.332 | -1.093 | 0.107 | 3.706 | 1.200 | 1.911 |
| altruist | suggested_throw05 | egalitarian | -2.292 | -1.936 | -1.580 | -0.868 | 0.556 | 1.268 | 4.829 | 15.510 | 3.560 | 0.644 |
| altruist | suggested_throw05 | majoritarian | -2.292 | -2.199 | -2.105 | -1.918 | -1.544 | -1.358 | -0.423 | 2.382 | 0.935 | 2.452 |
| altruist | suggested_throw05 | maximin | -2.292 | -2.183 | -2.073 | -1.855 | -1.417 | -1.198 | -0.104 | 3.179 | 1.094 | 2.095 |
| altruist | suggested_throw05 | optimist | -2.292 | -2.437 | -2.582 | -2.873 | -3.453 | -3.743 | -5.194 | -9.546 | -1.451 |  |
| altruist | suggested_throw05 | utilitarian | -2.292 | -2.398 | -2.503 | -2.714 | -3.137 | -3.348 | -4.403 | -7.570 | -1.056 |  |
| altruist | suggested_throw05 | veto | -2.292 | -1.945 | -1.597 | -0.901 | 0.490 | 1.185 | 4.663 | 15.096 | 3.478 | 0.659 |
| altruist | suggested_throw08 | average | -1.909 | -1.789 | -1.669 | -1.429 | -0.949 | -0.709 | 0.491 | 4.090 | 1.200 | 1.591 |
| altruist | suggested_throw08 | egalitarian | -1.909 | -1.553 | -1.197 | -0.485 | 0.940 | 1.652 | 5.212 | 15.894 | 3.560 | 0.536 |
| altruist | suggested_throw08 | majoritarian | -1.909 | -1.815 | -1.722 | -1.535 | -1.161 | -0.974 | -0.039 | 2.765 | 0.935 | 2.042 |
| altruist | suggested_throw08 | maximin | -1.909 | -1.799 | -1.690 | -1.471 | -1.033 | -0.815 | 0.280 | 3.562 | 1.094 | 1.744 |
| altruist | suggested_throw08 | optimist | -1.909 | -2.054 | -2.199 | -2.489 | -3.069 | -3.360 | -4.810 | -9.163 | -1.451 |  |
| altruist | suggested_throw08 | utilitarian | -1.909 | -2.014 | -2.120 | -2.331 | -2.753 | -2.964 | -4.020 | -7.186 | -1.056 |  |
| altruist | suggested_throw08 | veto | -1.909 | -1.561 | -1.213 | -0.518 | 0.873 | 1.569 | 5.046 | 15.479 | 3.478 | 0.549 |
| altruist_eater | current | average | -2.008 | -1.860 | -1.712 | -1.416 | -0.825 | -0.529 | 0.950 | 5.387 | 1.479 | 1.358 |
| altruist_eater | current | egalitarian | -2.008 | -1.597 | -1.187 | -0.365 | 1.277 | 2.099 | 6.205 | 18.524 | 4.106 | 0.489 |
| altruist_eater | current | majoritarian | -2.008 | -1.896 | -1.784 | -1.560 | -1.112 | -0.889 | 0.231 | 3.588 | 1.119 | 1.794 |
| altruist_eater | current | maximin | -2.008 | -1.869 | -1.731 | -1.454 | -0.899 | -0.622 | 0.763 | 4.920 | 1.386 | 1.449 |
| altruist_eater | current | optimist | -2.008 | -2.161 | -2.315 | -2.621 | -3.235 | -3.542 | -5.076 | -9.678 | -1.534 |  |
| altruist_eater | current | utilitarian | -2.008 | -2.113 | -2.219 | -2.430 | -2.852 | -3.063 | -4.118 | -7.282 | -1.055 |  |
| altruist_eater | current | veto | -2.008 | -1.609 | -1.209 | -0.411 | 1.185 | 1.984 | 5.975 | 17.950 | 3.992 | 0.503 |
| altruist_eater | nonnegative | average | -2.421 | -2.226 | -2.032 | -1.643 | -0.865 | -0.476 | 1.469 | 7.303 | 1.945 | 1.245 |
| altruist_eater | nonnegative | egalitarian | -2.421 | -2.219 | -2.017 | -1.614 | -0.807 | -0.404 | 1.613 | 7.663 | 2.017 | 1.200 |
| altruist_eater | nonnegative | majoritarian | -2.421 | -2.236 | -2.052 | -1.684 | -0.946 | -0.578 | 1.265 | 6.794 | 1.843 | 1.313 |
| altruist_eater | nonnegative | maximin | -2.421 | -2.210 | -1.999 | -1.577 | -0.733 | -0.311 | 1.798 | 8.126 | 2.109 | 1.148 |
| altruist_eater | nonnegative | optimist | -2.421 | -2.502 | -2.583 | -2.745 | -3.069 | -3.231 | -4.041 | -6.472 | -0.810 |  |
| altruist_eater | nonnegative | utilitarian | -2.421 | -2.526 | -2.632 | -2.843 | -3.265 | -3.476 | -4.530 | -7.695 | -1.055 |  |
| altruist_eater | nonnegative | veto | -2.421 | -2.225 | -2.030 | -1.640 | -0.859 | -0.468 | 1.484 | 7.342 | 1.953 | 1.240 |
| altruist_eater | suggested | average | -2.638 | -2.490 | -2.343 | -2.047 | -1.455 | -1.159 | 0.320 | 4.757 | 1.479 | 1.784 |
| altruist_eater | suggested | egalitarian | -2.638 | -2.228 | -1.817 | -0.996 | 0.647 | 1.468 | 5.574 | 17.894 | 4.106 | 0.642 |
| altruist_eater | suggested | majoritarian | -2.638 | -2.526 | -2.415 | -2.191 | -1.743 | -1.519 | -0.400 | 2.958 | 1.119 | 2.357 |
| altruist_eater | suggested | maximin | -2.638 | -2.500 | -2.361 | -2.084 | -1.530 | -1.253 | 0.133 | 4.289 | 1.386 | 1.904 |
| altruist_eater | suggested | optimist | -2.638 | -2.792 | -2.945 | -3.252 | -3.866 | -4.172 | -5.707 | -10.309 | -1.534 |  |
| altruist_eater | suggested | utilitarian | -2.638 | -2.744 | -2.849 | -3.060 | -3.482 | -3.693 | -4.748 | -7.913 | -1.055 |  |
| altruist_eater | suggested | veto | -2.638 | -2.239 | -1.840 | -1.042 | 0.555 | 1.353 | 5.345 | 17.319 | 3.992 | 0.661 |
| altruist_eater | suggested_alive | average | -2.725 | -2.577 | -2.430 | -2.134 | -1.542 | -1.246 | 0.233 | 4.670 | 1.479 | 1.843 |
| altruist_eater | suggested_alive | egalitarian | -2.725 | -2.315 | -1.904 | -1.083 | 0.560 | 1.381 | 5.487 | 17.807 | 4.106 | 0.664 |
| altruist_eater | suggested_alive | majoritarian | -2.725 | -2.613 | -2.502 | -2.278 | -1.830 | -1.606 | -0.487 | 2.871 | 1.119 | 2.435 |
| altruist_eater | suggested_alive | maximin | -2.725 | -2.587 | -2.448 | -2.171 | -1.617 | -1.340 | 0.046 | 4.202 | 1.386 | 1.967 |
| altruist_eater | suggested_alive | optimist | -2.725 | -2.879 | -3.032 | -3.339 | -3.953 | -4.259 | -5.794 | -10.396 | -1.534 |  |
| altruist_eater | suggested_alive | utilitarian | -2.725 | -2.831 | -2.936 | -3.147 | -3.569 | -3.780 | -4.835 | -8.000 | -1.055 |  |
| altruist_eater | suggested_alive | veto | -2.725 | -2.326 | -1.927 | -1.129 | 0.468 | 1.266 | 5.258 | 17.232 | 3.992 | 0.683 |
| altruist_eater | suggested_death5 | average | -3.509 | -3.361 | -3.213 | -2.917 | -2.326 | -2.030 | -0.551 | 3.886 | 1.479 | 2.372 |
| altruist_eater | suggested_death5 | egalitarian | -3.509 | -3.098 | -2.688 | -1.866 | -0.224 | 0.598 | 4.704 | 17.023 | 4.106 | 0.854 |
| altruist_eater | suggested_death5 | majoritarian | -3.509 | -3.397 | -3.285 | -3.061 | -2.614 | -2.390 | -1.271 | 2.087 | 1.119 | 3.135 |
| altruist_eater | suggested_death5 | maximin | -3.509 | -3.370 | -3.232 | -2.955 | -2.400 | -2.123 | -0.738 | 3.419 | 1.386 | 2.532 |
| altruist_eater | suggested_death5 | optimist | -3.509 | -3.662 | -3.816 | -4.123 | -4.736 | -5.043 | -6.577 | -11.180 | -1.534 |  |
| altruist_eater | suggested_death5 | utilitarian | -3.509 | -3.614 | -3.720 | -3.931 | -4.353 | -4.564 | -5.619 | -8.783 | -1.055 |  |
| altruist_eater | suggested_death5 | veto | -3.509 | -3.110 | -2.711 | -1.912 | -0.316 | 0.483 | 4.474 | 16.449 | 3.992 | 0.879 |
| altruist_eater | suggested_ethics_as_current | average | -2.154 | -2.006 | -1.858 | -1.563 | -0.971 | -0.675 | 0.804 | 5.241 | 1.479 | 1.457 |
| altruist_eater | suggested_ethics_as_current | egalitarian | -2.154 | -1.744 | -1.333 | -0.512 | 1.131 | 1.952 | 6.059 | 18.378 | 4.106 | 0.525 |
| altruist_eater | suggested_ethics_as_current | majoritarian | -2.154 | -2.042 | -1.930 | -1.707 | -1.259 | -1.035 | 0.084 | 3.442 | 1.119 | 1.925 |
| altruist_eater | suggested_ethics_as_current | maximin | -2.154 | -2.016 | -1.877 | -1.600 | -1.046 | -0.769 | 0.617 | 4.774 | 1.386 | 1.555 |
| altruist_eater | suggested_ethics_as_current | optimist | -2.154 | -2.308 | -2.461 | -2.768 | -3.382 | -3.688 | -5.222 | -9.825 | -1.534 |  |
| altruist_eater | suggested_ethics_as_current | utilitarian | -2.154 | -2.260 | -2.365 | -2.576 | -2.998 | -3.209 | -4.264 | -7.429 | -1.055 |  |
| altruist_eater | suggested_ethics_as_current | veto | -2.154 | -1.755 | -1.356 | -0.558 | 1.039 | 1.837 | 5.829 | 17.804 | 3.992 | 0.540 |
| altruist_eater | suggested_forage05 | average | -3.166 | -3.018 | -2.870 | -2.574 | -1.983 | -1.687 | -0.208 | 4.229 | 1.479 | 2.140 |
| altruist_eater | suggested_forage05 | egalitarian | -3.166 | -2.755 | -2.345 | -1.523 | 0.119 | 0.941 | 5.047 | 17.366 | 4.106 | 0.771 |
| altruist_eater | suggested_forage05 | majoritarian | -3.166 | -3.054 | -2.942 | -2.718 | -2.270 | -2.047 | -0.927 | 2.430 | 1.119 | 2.829 |
| altruist_eater | suggested_forage05 | maximin | -3.166 | -3.027 | -2.889 | -2.612 | -2.057 | -1.780 | -0.395 | 3.762 | 1.386 | 2.285 |
| altruist_eater | suggested_forage05 | optimist | -3.166 | -3.319 | -3.473 | -3.779 | -4.393 | -4.700 | -6.234 | -10.836 | -1.534 |  |
| altruist_eater | suggested_forage05 | utilitarian | -3.166 | -3.271 | -3.377 | -3.588 | -4.010 | -4.221 | -5.276 | -8.440 | -1.055 |  |
| altruist_eater | suggested_forage05 | veto | -3.166 | -2.767 | -2.367 | -1.569 | 0.027 | 0.826 | 4.817 | 16.792 | 3.992 | 0.793 |
| altruist_eater | suggested_forage08 | average | -3.482 | -3.334 | -3.186 | -2.891 | -2.299 | -2.003 | -0.524 | 3.913 | 1.479 | 2.354 |
| altruist_eater | suggested_forage08 | egalitarian | -3.482 | -3.072 | -2.661 | -1.840 | -0.197 | 0.624 | 4.731 | 17.050 | 4.106 | 0.848 |
| altruist_eater | suggested_forage08 | majoritarian | -3.482 | -3.370 | -3.258 | -3.035 | -2.587 | -2.363 | -1.244 | 2.114 | 1.119 | 3.111 |
| altruist_eater | suggested_forage08 | maximin | -3.482 | -3.344 | -3.205 | -2.928 | -2.374 | -2.097 | -0.711 | 3.446 | 1.386 | 2.513 |
| altruist_eater | suggested_forage08 | optimist | -3.482 | -3.636 | -3.789 | -4.096 | -4.710 | -5.016 | -6.551 | -11.153 | -1.534 |  |
| altruist_eater | suggested_forage08 | utilitarian | -3.482 | -3.588 | -3.693 | -3.904 | -4.326 | -4.537 | -5.592 | -8.757 | -1.055 |  |
| altruist_eater | suggested_forage08 | veto | -3.482 | -3.083 | -2.684 | -1.886 | -0.289 | 0.509 | 4.501 | 16.475 | 3.992 | 0.872 |
| altruist_eater | suggested_positive_sanction | average | -2.638 | -2.444 | -2.249 | -1.860 | -1.083 | -0.694 | 1.251 | 7.085 | 1.945 | 1.357 |
| altruist_eater | suggested_positive_sanction | egalitarian | -2.638 | -2.437 | -2.235 | -1.832 | -1.025 | -0.622 | 1.395 | 7.445 | 2.017 | 1.308 |
| altruist_eater | suggested_positive_sanction | majoritarian | -2.638 | -2.454 | -2.270 | -1.901 | -1.164 | -0.795 | 1.048 | 6.577 | 1.843 | 1.432 |
| altruist_eater | suggested_positive_sanction | maximin | -2.638 | -2.427 | -2.216 | -1.795 | -0.951 | -0.529 | 1.580 | 7.908 | 2.109 | 1.251 |
| altruist_eater | suggested_positive_sanction | optimist | -2.638 | -2.719 | -2.800 | -2.962 | -3.287 | -3.449 | -4.259 | -6.690 | -0.810 |  |
| altruist_eater | suggested_positive_sanction | utilitarian | -2.638 | -2.744 | -2.849 | -3.060 | -3.482 | -3.693 | -4.748 | -7.913 | -1.055 |  |
| altruist_eater | suggested_positive_sanction | veto | -2.638 | -2.443 | -2.248 | -1.857 | -1.076 | -0.686 | 1.267 | 7.124 | 1.953 | 1.351 |
| altruist_eater | suggested_shaping | average | -1.729 | -1.581 | -1.433 | -1.137 | -0.546 | -0.250 | 1.229 | 5.666 | 1.479 | 1.169 |
| altruist_eater | suggested_shaping | egalitarian | -1.729 | -1.318 | -0.908 | -0.087 | 1.556 | 2.377 | 6.484 | 18.803 | 4.106 | 0.421 |
| altruist_eater | suggested_shaping | majoritarian | -1.729 | -1.617 | -1.505 | -1.281 | -0.834 | -0.610 | 0.509 | 3.867 | 1.119 | 1.545 |
| altruist_eater | suggested_shaping | maximin | -1.729 | -1.591 | -1.452 | -1.175 | -0.621 | -0.344 | 1.042 | 5.199 | 1.386 | 1.248 |
| altruist_eater | suggested_shaping | optimist | -1.729 | -1.883 | -2.036 | -2.343 | -2.956 | -3.263 | -4.797 | -9.400 | -1.534 |  |
| altruist_eater | suggested_shaping | utilitarian | -1.729 | -1.835 | -1.940 | -2.151 | -2.573 | -2.784 | -3.839 | -7.004 | -1.055 |  |
| altruist_eater | suggested_shaping | veto | -1.729 | -1.330 | -0.931 | -0.132 | 1.464 | 2.262 | 6.254 | 18.229 | 3.992 | 0.433 |
| altruist_eater | suggested_throw05 | average | -1.965 | -1.817 | -1.669 | -1.373 | -0.781 | -0.485 | 0.994 | 5.431 | 1.479 | 1.328 |
| altruist_eater | suggested_throw05 | egalitarian | -1.965 | -1.554 | -1.143 | -0.322 | 1.321 | 2.142 | 6.248 | 18.568 | 4.106 | 0.478 |
| altruist_eater | suggested_throw05 | majoritarian | -1.965 | -1.853 | -1.741 | -1.517 | -1.069 | -0.845 | 0.274 | 3.631 | 1.119 | 1.755 |
| altruist_eater | suggested_throw05 | maximin | -1.965 | -1.826 | -1.687 | -1.410 | -0.856 | -0.579 | 0.807 | 4.963 | 1.386 | 1.418 |
| altruist_eater | suggested_throw05 | optimist | -1.965 | -2.118 | -2.271 | -2.578 | -3.192 | -3.499 | -5.033 | -9.635 | -1.534 |  |
| altruist_eater | suggested_throw05 | utilitarian | -1.965 | -2.070 | -2.175 | -2.386 | -2.808 | -3.019 | -4.074 | -7.239 | -1.055 |  |
| altruist_eater | suggested_throw05 | veto | -1.965 | -1.565 | -1.166 | -0.368 | 1.229 | 2.027 | 6.019 | 17.993 | 3.992 | 0.492 |
| altruist_eater | suggested_throw08 | average | -1.560 | -1.412 | -1.264 | -0.969 | -0.377 | -0.081 | 1.398 | 5.835 | 1.479 | 1.055 |
| altruist_eater | suggested_throw08 | egalitarian | -1.560 | -1.150 | -0.739 | 0.082 | 1.725 | 2.546 | 6.653 | 18.972 | 4.106 | 0.380 |
| altruist_eater | suggested_throw08 | majoritarian | -1.560 | -1.448 | -1.336 | -1.113 | -0.665 | -0.441 | 0.678 | 4.036 | 1.119 | 1.394 |
| altruist_eater | suggested_throw08 | maximin | -1.560 | -1.422 | -1.283 | -1.006 | -0.452 | -0.175 | 1.211 | 5.368 | 1.386 | 1.126 |
| altruist_eater | suggested_throw08 | optimist | -1.560 | -1.714 | -1.867 | -2.174 | -2.787 | -3.094 | -4.628 | -9.231 | -1.534 |  |
| altruist_eater | suggested_throw08 | utilitarian | -1.560 | -1.666 | -1.771 | -1.982 | -2.404 | -2.615 | -3.670 | -6.835 | -1.055 |  |
| altruist_eater | suggested_throw08 | veto | -1.560 | -1.161 | -0.762 | 0.036 | 1.633 | 2.431 | 6.423 | 18.398 | 3.992 | 0.391 |
| sharer | current | average | -1.288 | -1.114 | -0.940 | -0.593 | 0.101 | 0.449 | 2.185 | 7.394 | 1.736 | 0.742 |
| sharer | current | egalitarian | -1.288 | -0.882 | -0.477 | 0.334 | 1.956 | 2.766 | 6.820 | 18.982 | 4.054 | 0.318 |
| sharer | current | majoritarian | -1.288 | -1.115 | -0.942 | -0.596 | 0.095 | 0.441 | 2.170 | 7.356 | 1.729 | 0.745 |
| sharer | current | maximin | -1.288 | -1.102 | -0.916 | -0.544 | 0.199 | 0.571 | 2.429 | 8.004 | 1.858 | 0.693 |
| sharer | current | optimist | -1.288 | -1.323 | -1.358 | -1.429 | -1.571 | -1.641 | -1.995 | -3.056 | -0.354 |  |
| sharer | current | utilitarian | -1.288 | -1.358 | -1.428 | -1.569 | -1.850 | -1.991 | -2.694 | -4.804 | -0.703 |  |
| sharer | current | veto | -1.288 | -0.886 | -0.485 | 0.317 | 1.922 | 2.725 | 6.737 | 18.775 | 4.012 | 0.321 |
| sharer | nonnegative | average | -1.461 | -1.287 | -1.114 | -0.766 | -0.072 | 0.275 | 2.012 | 7.221 | 1.736 | 0.841 |
| sharer | nonnegative | egalitarian | -1.461 | -1.274 | -1.087 | -0.713 | 0.036 | 0.410 | 2.281 | 7.894 | 1.871 | 0.781 |
| sharer | nonnegative | majoritarian | -1.461 | -1.288 | -1.115 | -0.769 | -0.078 | 0.268 | 1.997 | 7.183 | 1.729 | 0.845 |
| sharer | nonnegative | maximin | -1.461 | -1.275 | -1.089 | -0.718 | 0.026 | 0.397 | 2.256 | 7.830 | 1.858 | 0.786 |
| sharer | nonnegative | optimist | -1.461 | -1.496 | -1.532 | -1.602 | -1.744 | -1.815 | -2.168 | -3.229 | -0.354 |  |
| sharer | nonnegative | utilitarian | -1.461 | -1.531 | -1.602 | -1.742 | -2.024 | -2.164 | -2.868 | -4.978 | -0.703 |  |
| sharer | nonnegative | veto | -1.461 | -1.278 | -1.095 | -0.729 | 0.003 | 0.368 | 2.198 | 7.686 | 1.829 | 0.799 |
| sharer | suggested | average | -1.600 | -1.426 | -1.253 | -0.905 | -0.211 | 0.136 | 1.873 | 7.082 | 1.736 | 0.921 |
| sharer | suggested | egalitarian | -1.600 | -1.194 | -0.789 | 0.022 | 1.643 | 2.454 | 6.508 | 18.670 | 4.054 | 0.395 |
| sharer | suggested | majoritarian | -1.600 | -1.427 | -1.254 | -0.908 | -0.217 | 0.129 | 1.858 | 7.044 | 1.729 | 0.925 |
| sharer | suggested | maximin | -1.600 | -1.414 | -1.228 | -0.857 | -0.113 | 0.258 | 2.117 | 7.691 | 1.858 | 0.861 |
| sharer | suggested | optimist | -1.600 | -1.635 | -1.671 | -1.741 | -1.883 | -1.954 | -2.307 | -3.368 | -0.354 |  |
| sharer | suggested | utilitarian | -1.600 | -1.670 | -1.740 | -1.881 | -2.162 | -2.303 | -3.006 | -5.116 | -0.703 |  |
| sharer | suggested | veto | -1.600 | -1.199 | -0.797 | 0.005 | 1.610 | 2.413 | 6.425 | 18.463 | 4.012 | 0.399 |
| sharer | suggested_alive | average | -1.655 | -1.482 | -1.308 | -0.961 | -0.266 | 0.081 | 1.817 | 7.026 | 1.736 | 0.953 |
| sharer | suggested_alive | egalitarian | -1.655 | -1.250 | -0.845 | -0.034 | 1.588 | 2.399 | 6.453 | 18.615 | 4.054 | 0.408 |
| sharer | suggested_alive | majoritarian | -1.655 | -1.482 | -1.310 | -0.964 | -0.272 | 0.073 | 1.802 | 6.988 | 1.729 | 0.958 |
| sharer | suggested_alive | maximin | -1.655 | -1.470 | -1.284 | -0.912 | -0.169 | 0.203 | 2.061 | 7.636 | 1.858 | 0.891 |
| sharer | suggested_alive | optimist | -1.655 | -1.691 | -1.726 | -1.797 | -1.938 | -2.009 | -2.363 | -3.424 | -0.354 |  |
| sharer | suggested_alive | utilitarian | -1.655 | -1.726 | -1.796 | -1.937 | -2.218 | -2.359 | -3.062 | -5.172 | -0.703 |  |
| sharer | suggested_alive | veto | -1.655 | -1.254 | -0.853 | -0.050 | 1.555 | 2.357 | 6.370 | 18.407 | 4.012 | 0.413 |
| sharer | suggested_death5 | average | -2.155 | -1.982 | -1.808 | -1.461 | -0.766 | -0.419 | 1.317 | 6.526 | 1.736 | 1.241 |
| sharer | suggested_death5 | egalitarian | -2.155 | -1.750 | -1.345 | -0.534 | 1.088 | 1.899 | 5.953 | 18.115 | 4.054 | 0.532 |
| sharer | suggested_death5 | majoritarian | -2.155 | -1.983 | -1.810 | -1.464 | -0.772 | -0.427 | 1.302 | 6.488 | 1.729 | 1.247 |
| sharer | suggested_death5 | maximin | -2.155 | -1.970 | -1.784 | -1.412 | -0.669 | -0.297 | 1.561 | 7.136 | 1.858 | 1.160 |
| sharer | suggested_death5 | optimist | -2.155 | -2.191 | -2.226 | -2.297 | -2.438 | -2.509 | -2.863 | -3.924 | -0.354 |  |
| sharer | suggested_death5 | utilitarian | -2.155 | -2.226 | -2.296 | -2.437 | -2.718 | -2.859 | -3.562 | -5.672 | -0.703 |  |
| sharer | suggested_death5 | veto | -2.155 | -1.754 | -1.353 | -0.550 | 1.055 | 1.857 | 5.870 | 17.907 | 4.012 | 0.537 |
| sharer | suggested_ethics_as_current | average | -1.308 | -1.134 | -0.960 | -0.613 | 0.081 | 0.429 | 2.165 | 7.374 | 1.736 | 0.753 |
| sharer | suggested_ethics_as_current | egalitarian | -1.308 | -0.902 | -0.497 | 0.314 | 1.936 | 2.746 | 6.800 | 18.962 | 4.054 | 0.323 |
| sharer | suggested_ethics_as_current | majoritarian | -1.308 | -1.135 | -0.962 | -0.616 | 0.075 | 0.421 | 2.150 | 7.336 | 1.729 | 0.756 |
| sharer | suggested_ethics_as_current | maximin | -1.308 | -1.122 | -0.936 | -0.564 | 0.179 | 0.551 | 2.409 | 7.983 | 1.858 | 0.704 |
| sharer | suggested_ethics_as_current | optimist | -1.308 | -1.343 | -1.378 | -1.449 | -1.591 | -1.661 | -2.015 | -3.076 | -0.354 |  |
| sharer | suggested_ethics_as_current | utilitarian | -1.308 | -1.378 | -1.448 | -1.589 | -1.870 | -2.011 | -2.714 | -4.824 | -0.703 |  |
| sharer | suggested_ethics_as_current | veto | -1.308 | -0.906 | -0.505 | 0.297 | 1.902 | 2.705 | 6.717 | 18.755 | 4.012 | 0.326 |
| sharer | suggested_forage05 | average | -1.951 | -1.778 | -1.604 | -1.257 | -0.562 | -0.215 | 1.521 | 6.730 | 1.736 | 1.124 |
| sharer | suggested_forage05 | egalitarian | -1.951 | -1.546 | -1.141 | -0.330 | 1.292 | 2.103 | 6.157 | 18.319 | 4.054 | 0.481 |
| sharer | suggested_forage05 | majoritarian | -1.951 | -1.779 | -1.606 | -1.260 | -0.568 | -0.223 | 1.506 | 6.692 | 1.729 | 1.129 |
| sharer | suggested_forage05 | maximin | -1.951 | -1.766 | -1.580 | -1.208 | -0.465 | -0.093 | 1.765 | 7.340 | 1.858 | 1.050 |
| sharer | suggested_forage05 | optimist | -1.951 | -1.987 | -2.022 | -2.093 | -2.234 | -2.305 | -2.659 | -3.720 | -0.354 |  |
| sharer | suggested_forage05 | utilitarian | -1.951 | -2.022 | -2.092 | -2.233 | -2.514 | -2.655 | -3.358 | -5.468 | -0.703 |  |
| sharer | suggested_forage05 | veto | -1.951 | -1.550 | -1.149 | -0.346 | 1.259 | 2.061 | 6.074 | 18.111 | 4.012 | 0.486 |
| sharer | suggested_forage08 | average | -2.162 | -1.989 | -1.815 | -1.468 | -0.773 | -0.426 | 1.310 | 6.519 | 1.736 | 1.245 |
| sharer | suggested_forage08 | egalitarian | -2.162 | -1.757 | -1.352 | -0.541 | 1.081 | 1.892 | 5.946 | 18.108 | 4.054 | 0.533 |
| sharer | suggested_forage08 | majoritarian | -2.162 | -1.990 | -1.817 | -1.471 | -0.779 | -0.434 | 1.295 | 6.481 | 1.729 | 1.251 |
| sharer | suggested_forage08 | maximin | -2.162 | -1.977 | -1.791 | -1.419 | -0.676 | -0.304 | 1.554 | 7.129 | 1.858 | 1.164 |
| sharer | suggested_forage08 | optimist | -2.162 | -2.198 | -2.233 | -2.304 | -2.445 | -2.516 | -2.870 | -3.931 | -0.354 |  |
| sharer | suggested_forage08 | utilitarian | -2.162 | -2.233 | -2.303 | -2.444 | -2.725 | -2.866 | -3.569 | -5.679 | -0.703 |  |
| sharer | suggested_forage08 | veto | -2.162 | -1.761 | -1.360 | -0.557 | 1.048 | 1.850 | 5.863 | 17.900 | 4.012 | 0.539 |
| sharer | suggested_positive_sanction | average | -1.600 | -1.426 | -1.253 | -0.905 | -0.211 | 0.136 | 1.873 | 7.082 | 1.736 | 0.921 |
| sharer | suggested_positive_sanction | egalitarian | -1.600 | -1.413 | -1.226 | -0.851 | -0.103 | 0.271 | 2.142 | 7.755 | 1.871 | 0.855 |
| sharer | suggested_positive_sanction | majoritarian | -1.600 | -1.427 | -1.254 | -0.908 | -0.217 | 0.129 | 1.858 | 7.044 | 1.729 | 0.925 |
| sharer | suggested_positive_sanction | maximin | -1.600 | -1.414 | -1.228 | -0.857 | -0.113 | 0.258 | 2.117 | 7.691 | 1.858 | 0.861 |
| sharer | suggested_positive_sanction | optimist | -1.600 | -1.635 | -1.671 | -1.741 | -1.883 | -1.954 | -2.307 | -3.368 | -0.354 |  |
| sharer | suggested_positive_sanction | utilitarian | -1.600 | -1.670 | -1.740 | -1.881 | -2.162 | -2.303 | -3.006 | -5.116 | -0.703 |  |
| sharer | suggested_positive_sanction | veto | -1.600 | -1.417 | -1.234 | -0.868 | -0.136 | 0.230 | 2.059 | 7.547 | 1.829 | 0.875 |
| sharer | suggested_shaping | average | -1.325 | -1.151 | -0.977 | -0.630 | 0.064 | 0.412 | 2.148 | 7.357 | 1.736 | 0.763 |
| sharer | suggested_shaping | egalitarian | -1.325 | -0.919 | -0.514 | 0.297 | 1.919 | 2.729 | 6.783 | 18.945 | 4.054 | 0.327 |
| sharer | suggested_shaping | majoritarian | -1.325 | -1.152 | -0.979 | -0.633 | 0.058 | 0.404 | 2.133 | 7.319 | 1.729 | 0.766 |
| sharer | suggested_shaping | maximin | -1.325 | -1.139 | -0.953 | -0.581 | 0.162 | 0.534 | 2.392 | 7.967 | 1.858 | 0.713 |
| sharer | suggested_shaping | optimist | -1.325 | -1.360 | -1.395 | -1.466 | -1.608 | -1.678 | -2.032 | -3.093 | -0.354 |  |
| sharer | suggested_shaping | utilitarian | -1.325 | -1.395 | -1.465 | -1.606 | -1.887 | -2.028 | -2.731 | -4.841 | -0.703 |  |
| sharer | suggested_shaping | veto | -1.325 | -0.923 | -0.522 | 0.280 | 1.885 | 2.688 | 6.700 | 18.738 | 4.012 | 0.330 |
| sharer | suggested_throw05 | average | -1.228 | -1.054 | -0.881 | -0.534 | 0.161 | 0.508 | 2.245 | 7.453 | 1.736 | 0.707 |
| sharer | suggested_throw05 | egalitarian | -1.228 | -0.823 | -0.417 | 0.394 | 2.015 | 2.826 | 6.880 | 19.042 | 4.054 | 0.303 |
| sharer | suggested_throw05 | majoritarian | -1.228 | -1.055 | -0.882 | -0.537 | 0.155 | 0.501 | 2.229 | 7.416 | 1.729 | 0.710 |
| sharer | suggested_throw05 | maximin | -1.228 | -1.042 | -0.856 | -0.485 | 0.259 | 0.630 | 2.488 | 8.063 | 1.858 | 0.661 |
| sharer | suggested_throw05 | optimist | -1.228 | -1.263 | -1.299 | -1.370 | -1.511 | -1.582 | -1.935 | -2.997 | -0.354 |  |
| sharer | suggested_throw05 | utilitarian | -1.228 | -1.298 | -1.369 | -1.509 | -1.791 | -1.931 | -2.635 | -4.745 | -0.703 |  |
| sharer | suggested_throw05 | veto | -1.228 | -0.827 | -0.426 | 0.377 | 1.982 | 2.784 | 6.797 | 18.834 | 4.012 | 0.306 |
| sharer | suggested_throw08 | average | -1.005 | -0.831 | -0.658 | -0.311 | 0.384 | 0.731 | 2.468 | 7.676 | 1.736 | 0.579 |
| sharer | suggested_throw08 | egalitarian | -1.005 | -0.600 | -0.194 | 0.617 | 2.238 | 3.049 | 7.103 | 19.265 | 4.054 | 0.248 |
| sharer | suggested_throw08 | majoritarian | -1.005 | -0.832 | -0.659 | -0.314 | 0.378 | 0.724 | 2.453 | 7.639 | 1.729 | 0.581 |
| sharer | suggested_throw08 | maximin | -1.005 | -0.819 | -0.633 | -0.262 | 0.482 | 0.853 | 2.711 | 8.286 | 1.858 | 0.541 |
| sharer | suggested_throw08 | optimist | -1.005 | -1.040 | -1.076 | -1.146 | -1.288 | -1.359 | -1.712 | -2.773 | -0.354 |  |
| sharer | suggested_throw08 | utilitarian | -1.005 | -1.075 | -1.146 | -1.286 | -1.568 | -1.708 | -2.412 | -4.522 | -0.703 |  |
| sharer | suggested_throw08 | veto | -1.005 | -0.604 | -0.203 | 0.600 | 2.205 | 3.007 | 7.020 | 19.057 | 4.012 | 0.250 |
| sharer_hoarder | current | average | -1.927 | -1.845 | -1.763 | -1.600 | -1.272 | -1.109 | -0.290 | 2.166 | 0.819 | 2.354 |
| sharer_hoarder | current | egalitarian | -1.927 | -1.731 | -1.535 | -1.143 | -0.359 | 0.033 | 1.992 | 7.872 | 1.960 | 0.983 |
| sharer_hoarder | current | majoritarian | -1.927 | -1.842 | -1.757 | -1.587 | -1.247 | -1.077 | -0.227 | 2.323 | 0.850 | 2.267 |
| sharer_hoarder | current | maximin | -1.927 | -1.828 | -1.730 | -1.532 | -1.137 | -0.939 | 0.048 | 3.011 | 0.988 | 1.951 |
| sharer_hoarder | current | optimist | -1.927 | -1.950 | -1.973 | -2.018 | -2.109 | -2.155 | -2.382 | -3.065 | -0.228 |  |
| sharer_hoarder | current | utilitarian | -1.927 | -1.976 | -2.026 | -2.124 | -2.321 | -2.419 | -2.911 | -4.386 | -0.492 |  |
| sharer_hoarder | current | veto | -1.927 | -1.735 | -1.543 | -1.159 | -0.390 | -0.006 | 1.915 | 7.679 | 1.921 | 1.003 |
| sharer_hoarder | nonnegative | average | -2.290 | -2.208 | -2.126 | -1.962 | -1.635 | -1.471 | -0.653 | 1.803 | 0.819 | 2.797 |
| sharer_hoarder | nonnegative | egalitarian | -2.290 | -2.201 | -2.113 | -1.936 | -1.581 | -1.404 | -0.519 | 2.137 | 0.885 | 2.586 |
| sharer_hoarder | nonnegative | majoritarian | -2.290 | -2.205 | -2.120 | -1.950 | -1.610 | -1.440 | -0.590 | 1.961 | 0.850 | 2.693 |
| sharer_hoarder | nonnegative | maximin | -2.290 | -2.191 | -2.092 | -1.895 | -1.500 | -1.302 | -0.314 | 2.649 | 0.988 | 2.318 |
| sharer_hoarder | nonnegative | optimist | -2.290 | -2.313 | -2.335 | -2.381 | -2.472 | -2.517 | -2.745 | -3.428 | -0.228 |  |
| sharer_hoarder | nonnegative | utilitarian | -2.290 | -2.339 | -2.388 | -2.487 | -2.683 | -2.782 | -3.274 | -4.749 | -0.492 |  |
| sharer_hoarder | nonnegative | veto | -2.290 | -2.205 | -2.120 | -1.951 | -1.612 | -1.443 | -0.596 | 1.944 | 0.847 | 2.704 |
| sharer_hoarder | suggested | average | -2.431 | -2.349 | -2.268 | -2.104 | -1.776 | -1.613 | -0.794 | 1.661 | 0.819 | 2.970 |
| sharer_hoarder | suggested | egalitarian | -2.431 | -2.235 | -2.039 | -1.647 | -0.863 | -0.472 | 1.488 | 7.368 | 1.960 | 1.241 |
| sharer_hoarder | suggested | majoritarian | -2.431 | -2.346 | -2.261 | -2.091 | -1.751 | -1.581 | -0.731 | 1.819 | 0.850 | 2.860 |
| sharer_hoarder | suggested | maximin | -2.431 | -2.333 | -2.234 | -2.036 | -1.641 | -1.444 | -0.456 | 2.507 | 0.988 | 2.462 |
| sharer_hoarder | suggested | optimist | -2.431 | -2.454 | -2.477 | -2.522 | -2.613 | -2.659 | -2.886 | -3.569 | -0.228 |  |
| sharer_hoarder | suggested | utilitarian | -2.431 | -2.480 | -2.530 | -2.628 | -2.825 | -2.923 | -3.415 | -4.891 | -0.492 |  |
| sharer_hoarder | suggested | veto | -2.431 | -2.239 | -2.047 | -1.663 | -0.894 | -0.510 | 1.411 | 7.175 | 1.921 | 1.265 |
| sharer_hoarder | suggested_alive | average | -2.488 | -2.406 | -2.324 | -2.160 | -1.833 | -1.669 | -0.851 | 1.605 | 0.819 | 3.039 |
| sharer_hoarder | suggested_alive | egalitarian | -2.488 | -2.292 | -2.096 | -1.704 | -0.920 | -0.528 | 1.432 | 7.311 | 1.960 | 1.269 |
| sharer_hoarder | suggested_alive | majoritarian | -2.488 | -2.403 | -2.318 | -2.148 | -1.808 | -1.638 | -0.788 | 1.763 | 0.850 | 2.926 |
| sharer_hoarder | suggested_alive | maximin | -2.488 | -2.389 | -2.290 | -2.093 | -1.698 | -1.500 | -0.512 | 2.451 | 0.988 | 2.519 |
| sharer_hoarder | suggested_alive | optimist | -2.488 | -2.511 | -2.533 | -2.579 | -2.670 | -2.715 | -2.943 | -3.626 | -0.228 |  |
| sharer_hoarder | suggested_alive | utilitarian | -2.488 | -2.537 | -2.586 | -2.685 | -2.881 | -2.980 | -3.472 | -4.947 | -0.492 |  |
| sharer_hoarder | suggested_alive | veto | -2.488 | -2.296 | -2.104 | -1.719 | -0.951 | -0.567 | 1.355 | 7.118 | 1.921 | 1.295 |
| sharer_hoarder | suggested_death5 | average | -2.997 | -2.916 | -2.834 | -2.670 | -2.343 | -2.179 | -1.360 | 1.095 | 0.819 | 3.662 |
| sharer_hoarder | suggested_death5 | egalitarian | -2.997 | -2.801 | -2.605 | -2.213 | -1.430 | -1.038 | 0.922 | 6.802 | 1.960 | 1.529 |
| sharer_hoarder | suggested_death5 | majoritarian | -2.997 | -2.912 | -2.827 | -2.657 | -2.317 | -2.147 | -1.297 | 1.253 | 0.850 | 3.526 |
| sharer_hoarder | suggested_death5 | maximin | -2.997 | -2.899 | -2.800 | -2.602 | -2.207 | -2.010 | -1.022 | 1.941 | 0.988 | 3.035 |
| sharer_hoarder | suggested_death5 | optimist | -2.997 | -3.020 | -3.043 | -3.088 | -3.179 | -3.225 | -3.453 | -4.135 | -0.228 |  |
| sharer_hoarder | suggested_death5 | utilitarian | -2.997 | -3.047 | -3.096 | -3.194 | -3.391 | -3.489 | -3.981 | -5.457 | -0.492 |  |
| sharer_hoarder | suggested_death5 | veto | -2.997 | -2.805 | -2.613 | -2.229 | -1.460 | -1.076 | 0.845 | 6.609 | 1.921 | 1.560 |
| sharer_hoarder | suggested_ethics_as_current | average | -1.973 | -1.891 | -1.810 | -1.646 | -1.319 | -1.155 | -0.336 | 2.119 | 0.819 | 2.411 |
| sharer_hoarder | suggested_ethics_as_current | egalitarian | -1.973 | -1.777 | -1.581 | -1.189 | -0.406 | -0.014 | 1.946 | 7.826 | 1.960 | 1.007 |
| sharer_hoarder | suggested_ethics_as_current | majoritarian | -1.973 | -1.888 | -1.803 | -1.633 | -1.293 | -1.123 | -0.273 | 2.277 | 0.850 | 2.321 |
| sharer_hoarder | suggested_ethics_as_current | maximin | -1.973 | -1.875 | -1.776 | -1.578 | -1.183 | -0.986 | 0.002 | 2.965 | 0.988 | 1.998 |
| sharer_hoarder | suggested_ethics_as_current | optimist | -1.973 | -1.996 | -2.019 | -2.064 | -2.155 | -2.201 | -2.428 | -3.111 | -0.228 |  |
| sharer_hoarder | suggested_ethics_as_current | utilitarian | -1.973 | -2.023 | -2.072 | -2.170 | -2.367 | -2.465 | -2.957 | -4.433 | -0.492 |  |
| sharer_hoarder | suggested_ethics_as_current | veto | -1.973 | -1.781 | -1.589 | -1.205 | -0.436 | -0.052 | 1.869 | 7.633 | 1.921 | 1.027 |
| sharer_hoarder | suggested_forage05 | average | -2.677 | -2.595 | -2.514 | -2.350 | -2.022 | -1.859 | -1.040 | 1.416 | 0.819 | 3.271 |
| sharer_hoarder | suggested_forage05 | egalitarian | -2.677 | -2.481 | -2.285 | -1.893 | -1.109 | -0.717 | 1.242 | 7.122 | 1.960 | 1.366 |
| sharer_hoarder | suggested_forage05 | majoritarian | -2.677 | -2.592 | -2.507 | -2.337 | -1.997 | -1.827 | -0.977 | 1.573 | 0.850 | 3.149 |
| sharer_hoarder | suggested_forage05 | maximin | -2.677 | -2.578 | -2.480 | -2.282 | -1.887 | -1.690 | -0.702 | 2.261 | 0.988 | 2.711 |
| sharer_hoarder | suggested_forage05 | optimist | -2.677 | -2.700 | -2.723 | -2.768 | -2.859 | -2.905 | -3.132 | -3.815 | -0.228 |  |
| sharer_hoarder | suggested_forage05 | utilitarian | -2.677 | -2.726 | -2.776 | -2.874 | -3.071 | -3.169 | -3.661 | -5.137 | -0.492 |  |
| sharer_hoarder | suggested_forage05 | veto | -2.677 | -2.485 | -2.293 | -1.909 | -1.140 | -0.756 | 1.165 | 6.929 | 1.921 | 1.393 |
| sharer_hoarder | suggested_forage08 | average | -2.825 | -2.743 | -2.661 | -2.497 | -2.170 | -2.006 | -1.188 | 1.268 | 0.819 | 3.451 |
| sharer_hoarder | suggested_forage08 | egalitarian | -2.825 | -2.629 | -2.433 | -2.041 | -1.257 | -0.865 | 1.095 | 6.974 | 1.960 | 1.441 |
| sharer_hoarder | suggested_forage08 | majoritarian | -2.825 | -2.740 | -2.655 | -2.485 | -2.145 | -1.975 | -1.125 | 1.426 | 0.850 | 3.323 |
| sharer_hoarder | suggested_forage08 | maximin | -2.825 | -2.726 | -2.627 | -2.430 | -2.035 | -1.837 | -0.849 | 2.114 | 0.988 | 2.860 |
| sharer_hoarder | suggested_forage08 | optimist | -2.825 | -2.848 | -2.870 | -2.916 | -3.007 | -3.052 | -3.280 | -3.963 | -0.228 |  |
| sharer_hoarder | suggested_forage08 | utilitarian | -2.825 | -2.874 | -2.923 | -3.022 | -3.218 | -3.317 | -3.809 | -5.284 | -0.492 |  |
| sharer_hoarder | suggested_forage08 | veto | -2.825 | -2.633 | -2.441 | -2.056 | -1.288 | -0.904 | 1.018 | 6.781 | 1.921 | 1.470 |
| sharer_hoarder | suggested_positive_sanction | average | -2.431 | -2.349 | -2.268 | -2.104 | -1.776 | -1.613 | -0.794 | 1.661 | 0.819 | 2.970 |
| sharer_hoarder | suggested_positive_sanction | egalitarian | -2.431 | -2.343 | -2.254 | -2.077 | -1.723 | -1.546 | -0.661 | 1.996 | 0.885 | 2.746 |
| sharer_hoarder | suggested_positive_sanction | majoritarian | -2.431 | -2.346 | -2.261 | -2.091 | -1.751 | -1.581 | -0.731 | 1.819 | 0.850 | 2.860 |
| sharer_hoarder | suggested_positive_sanction | maximin | -2.431 | -2.333 | -2.234 | -2.036 | -1.641 | -1.444 | -0.456 | 2.507 | 0.988 | 2.462 |
| sharer_hoarder | suggested_positive_sanction | optimist | -2.431 | -2.454 | -2.477 | -2.522 | -2.613 | -2.659 | -2.886 | -3.569 | -0.228 |  |
| sharer_hoarder | suggested_positive_sanction | utilitarian | -2.431 | -2.480 | -2.530 | -2.628 | -2.825 | -2.923 | -3.415 | -4.891 | -0.492 |  |
| sharer_hoarder | suggested_positive_sanction | veto | -2.431 | -2.347 | -2.262 | -2.093 | -1.754 | -1.584 | -0.738 | 1.803 | 0.847 | 2.871 |
| sharer_hoarder | suggested_shaping | average | -1.686 | -1.605 | -1.523 | -1.359 | -1.032 | -0.868 | -0.049 | 2.406 | 0.819 | 2.060 |
| sharer_hoarder | suggested_shaping | egalitarian | -1.686 | -1.491 | -1.295 | -0.903 | -0.119 | 0.273 | 2.233 | 8.113 | 1.960 | 0.861 |
| sharer_hoarder | suggested_shaping | majoritarian | -1.686 | -1.601 | -1.516 | -1.346 | -1.006 | -0.836 | 0.014 | 2.564 | 0.850 | 1.984 |
| sharer_hoarder | suggested_shaping | maximin | -1.686 | -1.588 | -1.489 | -1.291 | -0.896 | -0.699 | 0.289 | 3.252 | 0.988 | 1.707 |
| sharer_hoarder | suggested_shaping | optimist | -1.686 | -1.709 | -1.732 | -1.778 | -1.869 | -1.914 | -2.142 | -2.824 | -0.228 |  |
| sharer_hoarder | suggested_shaping | utilitarian | -1.686 | -1.736 | -1.785 | -1.883 | -2.080 | -2.178 | -2.670 | -4.146 | -0.492 |  |
| sharer_hoarder | suggested_shaping | veto | -1.686 | -1.494 | -1.302 | -0.918 | -0.149 | 0.235 | 2.156 | 7.920 | 1.921 | 0.878 |
| sharer_hoarder | suggested_throw05 | average | -2.139 | -2.057 | -1.976 | -1.812 | -1.484 | -1.321 | -0.502 | 1.954 | 0.819 | 2.613 |
| sharer_hoarder | suggested_throw05 | egalitarian | -2.139 | -1.943 | -1.747 | -1.355 | -0.571 | -0.179 | 1.780 | 7.660 | 1.960 | 1.092 |
| sharer_hoarder | suggested_throw05 | majoritarian | -2.139 | -2.054 | -1.969 | -1.799 | -1.459 | -1.289 | -0.439 | 2.111 | 0.850 | 2.516 |
| sharer_hoarder | suggested_throw05 | maximin | -2.139 | -2.040 | -1.942 | -1.744 | -1.349 | -1.151 | -0.164 | 2.799 | 0.988 | 2.166 |
| sharer_hoarder | suggested_throw05 | optimist | -2.139 | -2.162 | -2.185 | -2.230 | -2.321 | -2.367 | -2.594 | -3.277 | -0.228 |  |
| sharer_hoarder | suggested_throw05 | utilitarian | -2.139 | -2.188 | -2.238 | -2.336 | -2.533 | -2.631 | -3.123 | -4.599 | -0.492 |  |
| sharer_hoarder | suggested_throw05 | veto | -2.139 | -1.947 | -1.755 | -1.371 | -0.602 | -0.218 | 1.703 | 7.467 | 1.921 | 1.113 |
| sharer_hoarder | suggested_throw08 | average | -1.964 | -1.882 | -1.800 | -1.637 | -1.309 | -1.145 | -0.327 | 2.129 | 0.819 | 2.399 |
| sharer_hoarder | suggested_throw08 | egalitarian | -1.964 | -1.768 | -1.572 | -1.180 | -0.396 | -0.004 | 1.956 | 7.835 | 1.960 | 1.002 |
| sharer_hoarder | suggested_throw08 | majoritarian | -1.964 | -1.879 | -1.794 | -1.624 | -1.284 | -1.114 | -0.264 | 2.287 | 0.850 | 2.310 |
| sharer_hoarder | suggested_throw08 | maximin | -1.964 | -1.865 | -1.766 | -1.569 | -1.174 | -0.976 | 0.011 | 2.975 | 0.988 | 1.988 |
| sharer_hoarder | suggested_throw08 | optimist | -1.964 | -1.987 | -2.009 | -2.055 | -2.146 | -2.192 | -2.419 | -3.102 | -0.228 |  |
| sharer_hoarder | suggested_throw08 | utilitarian | -1.964 | -2.013 | -2.062 | -2.161 | -2.357 | -2.456 | -2.948 | -4.423 | -0.492 |  |
| sharer_hoarder | suggested_throw08 | veto | -1.964 | -1.772 | -1.580 | -1.195 | -0.427 | -0.043 | 1.879 | 7.642 | 1.921 | 1.022 |

## Failed checks by sanction

Failed checks (reward functions x agent types) at each sanction:

| check | s=0 | s=0.1 | s=0.2 | s=0.4 | s=0.8 | s=1 | s=2 | s=5 |
|---|---|---|---|---|---|---|---|---|
| sanction raises altruist relative to eater | 0 | 24 | 24 | 24 | 24 | 24 | 24 | 24 |
| sanction raises altruist_eater relative to eater | 0 | 24 | 24 | 24 | 24 | 24 | 24 | 24 |
| sanction raises sharer relative to eater | 0 | 24 | 24 | 24 | 24 | 24 | 24 | 24 |
| sanction raises sharer_hoarder relative to eater | 0 | 24 | 24 | 24 | 24 | 24 | 24 | 24 |
| surviving > dying early (by 1 sd) | 8 | 8 | 8 | 8 | 14 | 18 | 24 | 22 |

By agent type:

| check | average | baseline | egalitarian | majoritarian | maximin | optimist | utilitarian | veto |
|---|---|---|---|---|---|---|---|---|
| sanction raises altruist relative to eater | 0 | 0 | 0 | 0 | 0 | 84 | 84 | 0 |
| sanction raises altruist_eater relative to eater | 0 | 0 | 0 | 0 | 0 | 84 | 84 | 0 |
| sanction raises sharer relative to eater | 0 | 0 | 0 | 0 | 0 | 84 | 84 | 0 |
| sanction raises sharer_hoarder relative to eater | 0 | 0 | 0 | 0 | 0 | 84 | 84 | 0 |
| surviving > dying early (by 1 sd) | 8 | 8 | 34 | 7 | 7 | 6 | 6 | 34 |

## Checks passed per reward function

| reward function | checks passed | failed | not checked |
|---|---|---|---|
| current_s0 | 44 | 0 | 28 |
| current_s0.1 | 64 | 8 | 0 |
| current_s0.2 | 64 | 8 | 0 |
| current_s0.4 | 64 | 8 | 0 |
| current_s0.8 | 64 | 8 | 0 |
| current_s1 | 64 | 8 | 0 |
| current_s2 | 62 | 10 | 0 |
| current_s5 | 62 | 10 | 0 |
| nonnegative_s0 | 44 | 0 | 28 |
| nonnegative_s0.1 | 64 | 8 | 0 |
| nonnegative_s0.2 | 64 | 8 | 0 |
| nonnegative_s0.4 | 64 | 8 | 0 |
| nonnegative_s0.8 | 64 | 8 | 0 |
| nonnegative_s1 | 64 | 8 | 0 |
| nonnegative_s2 | 64 | 8 | 0 |
| nonnegative_s5 | 64 | 8 | 0 |
| suggested_alive_s0 | 44 | 0 | 28 |
| suggested_alive_s0.1 | 64 | 8 | 0 |
| suggested_alive_s0.2 | 64 | 8 | 0 |
| suggested_alive_s0.4 | 64 | 8 | 0 |
| suggested_alive_s0.8 | 62 | 10 | 0 |
| suggested_alive_s1 | 62 | 10 | 0 |
| suggested_alive_s2 | 62 | 10 | 0 |
| suggested_alive_s5 | 62 | 10 | 0 |
| suggested_death5_s0 | 44 | 0 | 28 |
| suggested_death5_s0.1 | 64 | 8 | 0 |
| suggested_death5_s0.2 | 64 | 8 | 0 |
| suggested_death5_s0.4 | 64 | 8 | 0 |
| suggested_death5_s0.8 | 64 | 8 | 0 |
| suggested_death5_s1 | 64 | 8 | 0 |
| suggested_death5_s2 | 62 | 10 | 0 |
| suggested_death5_s5 | 62 | 10 | 0 |
| suggested_ethics_as_current_s0 | 44 | 0 | 28 |
| suggested_ethics_as_current_s0.1 | 64 | 8 | 0 |
| suggested_ethics_as_current_s0.2 | 64 | 8 | 0 |
| suggested_ethics_as_current_s0.4 | 64 | 8 | 0 |
| suggested_ethics_as_current_s0.8 | 62 | 10 | 0 |
| suggested_ethics_as_current_s1 | 62 | 10 | 0 |
| suggested_ethics_as_current_s2 | 62 | 10 | 0 |
| suggested_ethics_as_current_s5 | 62 | 10 | 0 |
| suggested_forage05_s0 | 44 | 0 | 28 |
| suggested_forage05_s0.1 | 64 | 8 | 0 |
| suggested_forage05_s0.2 | 64 | 8 | 0 |
| suggested_forage05_s0.4 | 64 | 8 | 0 |
| suggested_forage05_s0.8 | 64 | 8 | 0 |
| suggested_forage05_s1 | 64 | 8 | 0 |
| suggested_forage05_s2 | 62 | 10 | 0 |
| suggested_forage05_s5 | 62 | 10 | 0 |
| suggested_forage08_s0 | 44 | 0 | 28 |
| suggested_forage08_s0.1 | 64 | 8 | 0 |
| suggested_forage08_s0.2 | 64 | 8 | 0 |
| suggested_forage08_s0.4 | 64 | 8 | 0 |
| suggested_forage08_s0.8 | 64 | 8 | 0 |
| suggested_forage08_s1 | 64 | 8 | 0 |
| suggested_forage08_s2 | 62 | 10 | 0 |
| suggested_forage08_s5 | 62 | 10 | 0 |
| suggested_positive_sanction_s0 | 44 | 0 | 28 |
| suggested_positive_sanction_s0.1 | 64 | 8 | 0 |
| suggested_positive_sanction_s0.2 | 64 | 8 | 0 |
| suggested_positive_sanction_s0.4 | 64 | 8 | 0 |
| suggested_positive_sanction_s0.8 | 64 | 8 | 0 |
| suggested_positive_sanction_s1 | 64 | 8 | 0 |
| suggested_positive_sanction_s2 | 64 | 8 | 0 |
| suggested_positive_sanction_s5 | 64 | 8 | 0 |
| suggested_s0 | 44 | 0 | 28 |
| suggested_s0.1 | 64 | 8 | 0 |
| suggested_s0.2 | 64 | 8 | 0 |
| suggested_s0.4 | 64 | 8 | 0 |
| suggested_s0.8 | 62 | 10 | 0 |
| suggested_s1 | 62 | 10 | 0 |
| suggested_s2 | 62 | 10 | 0 |
| suggested_s5 | 62 | 10 | 0 |
| suggested_shaping_s0 | 36 | 8 | 28 |
| suggested_shaping_s0.1 | 56 | 16 | 0 |
| suggested_shaping_s0.2 | 56 | 16 | 0 |
| suggested_shaping_s0.4 | 56 | 16 | 0 |
| suggested_shaping_s0.8 | 56 | 16 | 0 |
| suggested_shaping_s1 | 56 | 16 | 0 |
| suggested_shaping_s2 | 58 | 14 | 0 |
| suggested_shaping_s5 | 60 | 12 | 0 |
| suggested_throw05_s0 | 44 | 0 | 28 |
| suggested_throw05_s0.1 | 64 | 8 | 0 |
| suggested_throw05_s0.2 | 64 | 8 | 0 |
| suggested_throw05_s0.4 | 64 | 8 | 0 |
| suggested_throw05_s0.8 | 64 | 8 | 0 |
| suggested_throw05_s1 | 62 | 10 | 0 |
| suggested_throw05_s2 | 62 | 10 | 0 |
| suggested_throw05_s5 | 62 | 10 | 0 |
| suggested_throw08_s0 | 44 | 0 | 28 |
| suggested_throw08_s0.1 | 64 | 8 | 0 |
| suggested_throw08_s0.2 | 64 | 8 | 0 |
| suggested_throw08_s0.4 | 64 | 8 | 0 |
| suggested_throw08_s0.8 | 64 | 8 | 0 |
| suggested_throw08_s1 | 62 | 10 | 0 |
| suggested_throw08_s2 | 62 | 10 | 0 |
| suggested_throw08_s5 | 62 | 10 | 0 |

Full results: data/reward_checks/sanctions/results.csv, checks.csv and sweep.csv.
