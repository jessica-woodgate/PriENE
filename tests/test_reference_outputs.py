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
# mode that changes only the utilitarian agents' reward and mean_reward, not what they do. And again when well-being was
# rounded before the sanction measures it, so float error no longer gives a false sanction: one step's reward in each
# scenario changes by 0.4 (and so mean_reward); every other column is unchanged
REFERENCE_REPORT_HASHES = {
    "agent_reports_allotment.csv": "f4e12f39f2947a72c95b7ac8fb4260bc70a93626cbc78b9663d1a910a12ebd1a",
    "agent_reports_capabilities.csv": "1c4140124f29fe230a48105deabc07c61ed53770453ad011dcafaf2523e3c14e",
    "agent_reports_colours.csv": "ffe83d49865dfd1f1dc4b18c06ce0edbeb4e3600730310690452b2ddef27eff8",
    "model_episode_reports_allotment.csv": "91d46994133c8b52a18a929e16ca1dfc3902420fb6c8d124e7d42ecf895d4cef",
    "model_episode_reports_capabilities.csv": "c1ccc1de3cff0db008a353058bf8072a9c64e62896d99c94ec138749216dd609",
    "model_episode_reports_colours.csv": "a3a7c5cd5088fc590c5d741faad4601a81080d7f517c49e216755e9f4416d990",
}

needs_trained_checkpoints = pytest.mark.skipif(not os.path.isdir(TRAINED_CHECKPOINTS), reason="needs the trained 200_days checkpoints")


@pytest.fixture
def in_tmp_dir(tmp_path, monkeypatch):
    """HarvestModel writes to the relative path data/results/current_run -- keep that inside tmp_path."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def report_hash(path):
    # columns added since the reference outputs were made
    df = pd.read_csv(path).drop(columns=["ethics_sanction", "season", "season_quality", "seasonal_berry_target", "allotment_regrowth_delay"],
                                errors="ignore")
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
