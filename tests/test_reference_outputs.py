import hashlib
import os

import pandas as pd
import pytest

from tests.conftest import NUM_AGENTS, TEST_SCENARIOS, _build, use_unequal_instant_allotments

TRAINED_CHECKPOINTS = "data/model_variables/200_days/4_agents/"
# absolute, since in_tmp_dir changes the working directory before the model is built
TRAINED_CHECKPOINTS_ABS = os.path.abspath(TRAINED_CHECKPOINTS) + "/"

# sha256 of the reports (read and rewritten by pandas) from a seeded run of each scenario, made with
# the code before unequal starting health, seasons and allotment regrowth delays were options
REFERENCE_REPORT_HASHES = {
    "agent_reports_allotment.csv": "a713b6e3f6f34ec27259436ab2f6ce437393a0247830e90454aa53b8f7a24cc1",
    "agent_reports_capabilities.csv": "4dc0236c5b50dfaf8cccff8e9c82f788bb43701247bd599de8c6e8a36a7748da",
    "agent_reports_colours.csv": "cc7e9e9f9e481b34d9f87a9fdb39cc1b1d581dfc096bcd849087e387e0bd0bd3",
    "model_episode_reports_allotment.csv": "551d284d15a8276d0cf5d592e27d2737d14e401fb41bc9a93d4af3035ec44285",
    "model_episode_reports_capabilities.csv": "66680a3af0792640d8a098d509eac606f924fe40298b199d46e5015f84030309",
    "model_episode_reports_colours.csv": "f2ba7f29c1f5595b8ff5d098e827dae756046e004e12156697c261c76bb5be6d",
}

needs_trained_checkpoints = pytest.mark.skipif(not os.path.isdir(TRAINED_CHECKPOINTS), reason="needs the trained 200_days checkpoints")


@pytest.fixture
def in_tmp_dir(tmp_path, monkeypatch):
    """HarvestModel writes to the relative path data/results/current_run -- keep that inside tmp_path."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def report_hash(path):
    # columns added since the reference outputs were made
    df = pd.read_csv(path).drop(columns=["season", "season_quality", "seasonal_berry_target", "allotment_regrowth_delay"], errors="ignore")
    return hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()


@needs_trained_checkpoints
@pytest.mark.parametrize("scenario", TEST_SCENARIOS)
def test_options_off_reproduce_reference_outputs(in_tmp_dir, scenario, monkeypatch):
    # the references predate allotments' equal split and slower allotment, so allotment is run as it was then
    use_unequal_instant_allotments(monkeypatch)
    model = _build(scenario, "homogeneous", NUM_AGENTS * 3, "utilitarian", False, TRAINED_CHECKPOINTS_ABS,
                   True, False, scenario, 2, 20, seed=42)
    while model.episode <= 2:
        model.step()
    for report in ["agent_reports", "model_episode_reports"]:
        filename = f"{report}_{scenario}.csv"
        assert report_hash(f"data/results/current_run/{filename}") == REFERENCE_REPORT_HASHES[filename]
