import hashlib
import os

import pandas as pd
import pytest

from tests.conftest import NUM_AGENTS, TEST_SCENARIOS, _build, use_fixed_allocations, use_unequal_instant_allotments

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
# scenario changes by 0.4 (and so mean_reward); every other column is unchanged. And again when health and days left to live were
# rounded whenever they change, so floating point error no longer stops an agent at 0.6 health throwing or keeps one at 0 health alive
REFERENCE_REPORT_HASHES = {
    "agent_reports_allotment.csv": "86c7e72a2fc9e0b7419363e56e150023667fc29459f188286429582483637ca7",
    "agent_reports_capabilities.csv": "b13ba293cf9a6e9cc9ca1dc1759573aff8d342c2b070dc1bbac4e286a59acf51",
    "agent_reports_colours.csv": "4e991f9162120c643e2a5be50e7fccdf2946d9007222514f51cf89f6f1d58eab",
    "model_episode_reports_allotment.csv": "b2f3f2c19eae33a56fdc18d6c4515960f486ef4a0db5422c6e8b657f67080405",
    "model_episode_reports_capabilities.csv": "d23d9c7fd21d3bca03961bab7116c6df35465a267f67f253db6d5b0540b16dd3",
    "model_episode_reports_colours.csv": "810be9e3bd9e81e1c984e945d9158b06862ed2575255e4edd87a1db59f274fa3",
}

needs_trained_checkpoints = pytest.mark.skipif(not os.path.isdir(TRAINED_CHECKPOINTS), reason="needs the trained 200_days checkpoints")


@pytest.fixture
def in_tmp_dir(tmp_path, monkeypatch):
    """HarvestModel writes to the relative path data/results/current_run -- keep that inside tmp_path."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def report_hash(path):
    # columns added since the reference outputs were made
    df = pd.read_csv(path).drop(columns=["thrown_to", "ethics_sanction", "season", "season_quality", "seasonal_berry_target", "allotment_regrowth_delay"],
                                errors="ignore")
    return hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()


@needs_trained_checkpoints
@pytest.mark.parametrize("scenario", TEST_SCENARIOS)
def test_options_off_reproduce_reference_outputs(in_tmp_dir, scenario, monkeypatch):
    # the references predate allotments' equal split and slower allotment, and agents being given allocations afresh each episode,
    # so they are run as they were then
    use_unequal_instant_allotments(monkeypatch)
    use_fixed_allocations(monkeypatch)
    model = _build(scenario, "homogeneous", NUM_AGENTS * 3, "utilitarian", False, TRAINED_CHECKPOINTS_ABS,
                   True, False, scenario, 2, 20, seed=42)
    while model.episode <= 2:
        model.step()
    for report in ["agent_reports", "model_episode_reports"]:
        filename = f"{report}_{scenario}.csv"
        assert report_hash(f"data/results/current_run/{filename}") == REFERENCE_REPORT_HASHES[filename]
