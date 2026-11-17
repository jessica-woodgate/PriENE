"""
The scripted-policy check of candidate reward functions (step 1 of the reward ablation), before any training:
    simulate.py -- runs scripted policies (src/agent/modules/scripted_decision_module.py) and logs every step's outcome
    rewards.py -- candidate reward functions (configs/rewards/*.yaml), and the reward each would give a logged step
    evaluate.py -- returns of every policy under every reward function and agent type, the checks, and docs/reward_checks.md
"""
