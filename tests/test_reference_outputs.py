import hashlib
import os

import pandas as pd
import pytest

from tests.conftest import NUM_AGENTS, TEST_SCENARIOS, _build, use_unequal_instant_allotments

TRAINED_CHECKPOINTS = "data/model_variables/200_days/4_agents/"
# absolute, since in_tmp_dir changes the working directory before the model is built
TRAINED_CHECKPOINTS_ABS = os.path.abspath(TRAINED_CHECKPOINTS) + "/"

# sha256 of the reports (read and rewritten by pandas) from a seeded run of each scenario, made with
# the code before unequal starting health, seasons and allotment regrowth delays were options. The model
# episode reports were remade when mean_berries, variance_health and gini_berries_consumed were corrected to
# be over each episode's agent rows (and mean_health changed by rounding only); every other column is unchanged.
# Both were remade again when the ethics sanction started seeing every action's effect (e.g. foraging): in test
# mode that changes only the utilitarian agents' reward and mean_reward, not what they do
REFERENCE_REPORT_HASHES = {
    "agent_reports_allotment.csv": "a5e40482821314d08e36348f6f64732ef38d69a599383a63f02739d4d775ba9d",
    "agent_reports_capabilities.csv": "b8a365e999c2afda23556bbc32ba9610028b012cd7303d9afccc80890169677b",
    "agent_reports_colours.csv": "20dfc065d514e8569df21fed98f7df9cec43e9ab1302d29ac65a97809aacf5d5",
    "model_episode_reports_allotment.csv": "e32a6075f091c038d7a6597924946c7a02cf054b504f635a20232ac92f9087f9",
    "model_episode_reports_capabilities.csv": "9c5b31768c2a480b279b938dfc07e7c0db8b88fb963e81ec658c6a26b6532900",
    "model_episode_reports_colours.csv": "d9bf3dea5d2cfff00be55ba315644c15acdf3b57337c4880fb890a0d0b76afa8",
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
