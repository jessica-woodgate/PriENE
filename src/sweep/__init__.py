"""
Parameter sweeps over training parameters (src/training_config.py), for any scenario: a sweep file (configs/sweeps/*.yaml, see spec.py)
says which parameters to vary and how; run.py runs each configuration and seed as an ordinary training run in a folder of its own
(runner.py); summarise.py collects the results and ranks the configurations by the sweep's selection rule.
    python -m src.sweep.run configs/sweeps/<sweep>.yaml [--workers N] [--dry-run]
    python -m src.sweep.summarise configs/sweeps/<sweep>.yaml
"""
