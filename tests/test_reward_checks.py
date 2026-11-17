"""
The scripted-policy reward check (src/reward_checks): each scripted policy chooses as specified in constructed situations, logging
outcomes doesn't change a run, every outcome is recorded with the key its reward was looked up by, and the offline rewards and returns
match the rewards agents received during a run (baseline and ethics agents, with their sanctions) for the current reward function.
"""
import os
import numpy as np
import pandas as pd
import pytest
import yaml
from src.agent.modules.ethics_module import EthicsModule
from src.agent.modules.scripted_decision_module import ScriptedDecisionModule
from src.reward_checks import rewards as rw
from src.reward_checks.evaluate import unit_sanctions, returns, agent_episodes
from src.reward_checks.simulate import build_model, simulate
from src.training_config import for_scenario
from tests.conftest import _build, _make_pretrained_checkpoint

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CURRENT = os.path.join(REPO, "configs", "rewards", "current.yaml")


@pytest.fixture
def in_tmp_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def scripted(policy):
    """A basic model whose agent 0 acts by policy and the others are eaters; agent 0, and the agents by id"""
    model = build_model("basic", {0: policy, 1: "eater", 2: "eater", 3: "eater"}, 1, 20, 4, 7)
    agents = {a.unique_id: a for a in model.get_living_agents()}
    return model, agents[0], agents


def set_state(agent, health, berries):
    agent.health, agent.berries = health, berries
    agent.days_left_to_live = agent.get_days_left_to_live()


def chosen(agent):
    return agent.actions[agent.decision_module.choose_action(agent.observe())]


# ── the policies ────────────────────────────────────────────────

def test_eater_eats_a_berry_it_carries_and_otherwise_forages(in_tmp_dir):
    _, agent, _ = scripted("eater")
    set_state(agent, 0.8, 1)
    assert chosen(agent) == "eat"
    set_state(agent, 0.8, 0)
    assert chosen(agent) == "move"


def test_passive_always_eats_so_it_never_forages(in_tmp_dir):
    _, agent, _ = scripted("passive")
    for berries in [0, 2]:
        set_state(agent, 0.8, berries)
        assert chosen(agent) == "eat"


def test_hoarder_forages_and_eats_only_when_the_next_decay_would_kill_it(in_tmp_dir):
    _, agent, _ = scripted("hoarder")
    set_state(agent, 0.5, 3)
    assert chosen(agent) == "move"
    set_state(agent, 0.1, 3)
    assert chosen(agent) == "eat"
    set_state(agent, 0.1, 0)
    assert chosen(agent) == "move"


def test_sharer_throws_to_the_neediest_reachable_living_agent_and_otherwise_acts_as_an_eater(in_tmp_dir):
    model, agent, agents = scripted("sharer")
    set_state(agent, 0.6, 1)
    set_state(agents[1], 0.5, 0)    # 5 days left
    set_state(agents[2], 0.3, 0)    # 3 days left: the neediest
    set_state(agents[3], 2.0, 0)    # 20 days left
    assert chosen(agent) == "throw_2"
    # a dead agent isn't living, and isn't thrown to
    agents[2].done = True
    assert chosen(agent) == "throw_1"
    # nor one it can't reach
    model.can_throw_to = lambda thrower, benefactor: benefactor.unique_id != 1
    assert chosen(agent) == "eat"
    model.can_throw_to = lambda thrower, benefactor: True
    # nobody below the low days-left threshold (10): as an eater
    set_state(agents[1], 1.5, 0)
    assert chosen(agent) == "eat"
    # too little health to throw, or no berry: as an eater
    set_state(agents[1], 0.5, 0)
    set_state(agent, 0.5, 1)
    assert chosen(agent) == "eat"
    set_state(agent, 0.8, 0)
    assert chosen(agent) == "move"


def test_altruist_throws_to_the_worst_off_whenever_it_can_and_otherwise_hoards(in_tmp_dir):
    _, agent, agents = scripted("altruist")
    set_state(agent, 0.6, 1)
    for agent_id, health in [(1, 3.0), (2, 2.5), (3, 4.0)]:
        set_state(agents[agent_id], health, 0)
    # everyone else is well off, but agent 2 is the worst off
    assert chosen(agent) == "throw_2"
    # equal days left: the lowest id
    set_state(agents[1], 2.5, 0)
    assert chosen(agent) == "throw_1"
    # can't throw (too little health): eats only when the next decay would kill it
    set_state(agent, 0.5, 1)
    assert chosen(agent) == "move"
    set_state(agent, 0.1, 1)
    assert chosen(agent) == "eat"


def test_random_chooses_every_action_reproducibly_from_the_agents_stream(in_tmp_dir):
    _, agent, _ = scripted("random")
    picks = [chosen(agent) for _ in range(300)]
    assert set(picks) == set(agent.actions)
    _, again, _ = scripted("random")
    assert [chosen(again) for _ in range(300)] == picks


def test_an_unknown_policy_is_refused():
    with pytest.raises(ValueError, match="unknown scripted policy"):
        ScriptedDecisionModule("lazy", None, None)


def test_an_agent_that_dies_on_the_last_day_doesnt_get_the_survive_reward(in_tmp_dir):
    model, agent, agents = scripted("eater")
    model.day = agent.max_days - 1
    table = agent.rewards
    # dies on the last day: death, not survive
    set_state(agent, 0.1, 0)
    done, reward = agent._update_attributes([0.0])
    assert done and float(np.sum(reward)) == table["death"]
    # alive at the end of the last day: survive
    set_state(agents[1], 0.5, 0)
    done, reward = agents[1]._update_attributes([0.0])
    assert not done and float(np.sum(reward)) == table["survive"]


# ── outcome logging ─────────────────────────────────────────────

def test_logging_outcomes_doesnt_change_a_run(tmp_path, monkeypatch):
    # the same run (the same trained networks, an ethics agent type, the same seed), without and with an outcome log: every agent's every
    # step the same
    monkeypatch.chdir(tmp_path)
    checkpoint = _make_pretrained_checkpoint(tmp_path, "utilitarian")
    steps, models = [], []
    for name, log in [("plain", None), ("logged", [])]:
        os.makedirs(tmp_path / name)
        monkeypatch.chdir(tmp_path / name)
        model = _build("colours", "homogeneous", 12, "utilitarian", False, checkpoint, False, False, name, 2, 15, seed=42)
        model.outcome_log = log
        trace = []
        while model.episode <= 2:
            model.step()
            trace += [(a.unique_id, a.current_action, a.current_reward, a.current_sanction, a.health, a.berries)
                      for a in sorted(model.schedule.agents, key=lambda a: a.unique_id) if a.agent_type != "berry"]
        steps.append(trace)
        models.append(model)
    assert steps[0] == steps[1] and len(steps[0]) > 50
    assert models[1].outcome_log and models[0].outcome_log is None


def test_every_outcome_is_the_key_its_reward_was_looked_up_by(in_tmp_dir):
    log = simulate("basic", {0: "random", 1: "random", 2: "sharer", 3: "passive"}, episodes=2, days=40, seed=3)
    from src.agent.harvest_agent import HarvestAgent
    assert set(log["outcome"]) <= set(HarvestAgent.OUTCOMES)
    table = rw.load(CURRENT).table("baseline")
    base = log["reward"] - np.where(log["died"], table["death"], 0) - np.where(log["last_day"] & ~log["died"], table["survive"], 0)
    assert np.allclose(base, log["outcome"].map(table))
    assert (log[log["outcome"] == "eat"]["berries_after"] == log[log["outcome"] == "eat"]["berries_before"] - 1).all()
    assert (log[log["outcome"] == "throw"]["thrown_to"].notna()).all()
    assert log["outcome"].isin(["throw", "no_berries", "forage", "eat", "neutral_reward"]).any()


# ── offline rewards ─────────────────────────────────────────────

@pytest.mark.parametrize("agent_type", ["baseline", "utilitarian", "maximin", "egalitarian", "average", "majoritarian", "optimist", "veto"])
def test_offline_rewards_match_the_rewards_received_during_a_run(in_tmp_dir, agent_type):
    # the agents get the current rewards during the run (and ethics agents their sanctions); offline, the same from the log
    log = simulate("basic", {0: "sharer", 1: "eater", 2: "altruist", 3: "random"}, episodes=2, days=60, seed=5, agent_type=agent_type)
    log = log.assign(scenario="basic", run="focal_sharer", setup="focal")
    for column in ["well_being_before", "well_being_after"]:
        log[column] = log[column].map(lambda values: "[" + ",".join(repr(float(v)) for v in values) + "]")
    gamma = for_scenario("basic").gamma
    sanctions = unit_sanctions(log, [agent_type])
    offline = rw.step_rewards(log, rw.load(CURRENT), agent_type, gamma, None if agent_type == "baseline" else sanctions[agent_type])
    assert np.allclose(offline, log["reward"])
    if agent_type != "baseline":
        assert np.allclose(0.4 * sanctions[agent_type], log["sanction"]) and (log["sanction"] != 0).any()
    # and so the returns: the undiscounted total is the sum of the rewards received, the discounted one discounts each day
    table = returns(log, [rw.load(CURRENT)], gamma, [agent_type], sanctions)
    received = log.groupby(["episode", "agent_id"])["reward"].sum().to_numpy()
    discounted = (log["reward"] * gamma ** log["day"]).groupby([log["episode"], log["agent_id"]]).sum().to_numpy()
    assert np.allclose(table["total_return"], received) and np.allclose(table["discounted_return"], discounted)


def test_a_unit_sanction_scales_to_any_sanction():
    rng = np.random.default_rng(0)
    for _ in range(200):
        before = list(rng.integers(0, 6, size=4).astype(float))
        after = [b + rng.choice([-1.0, 0.0, 6.0]) for b in before]
        for agent_type in rw.PRINCIPLES + rw.AGGREGATIONS:
            unit, scaled = EthicsModule(1, agent_type), EthicsModule(0.4, agent_type)
            unit.update_ethics_state(before)
            scaled.update_ethics_state(before)
            assert np.sum(scaled.get_sanction(after)) == pytest.approx(0.4 * np.sum(unit.get_sanction(after)))


def test_alive_and_shaping_rewards():
    log = pd.DataFrame({"outcome": ["forage", "eat", "neutral_reward"], "died": [False, False, True], "last_day": [False, False, False],
                        "days_left_before": [8.0, 13.0, 1.0], "days_left_acted": [14.0, 13.0, 1.0], "days_left_after": [13.0, 12.0, 0.0]})
    function = rw.RewardFunction("x", {**{k: 0.0 for k in ["death", "no_berries", "no_benefactor", "insufficient_health", "neutral_reward",
                                                            "throw", "forage", "eat", "survive"]}, "death": -1.0}, alive=0.02, shaping=True)
    gamma = 0.95
    expected = [0.02 + gamma * 14 - 8, 0.02 + gamma * 13 - 13, -1 + 0 - 1]
    assert np.allclose(rw.step_rewards(log, function, "baseline", gamma), expected)


def test_reward_files_load_and_current_is_the_training_configs():
    for path in sorted(os.listdir(os.path.join(REPO, "configs", "rewards"))):
        rw.load(os.path.join(REPO, "configs", "rewards", path))
    current, config = rw.load(CURRENT), for_scenario("basic")
    assert current.rewards == config.rewards and current.ethics_rewards == config.ethics_rewards and current.sanction == config.sanction


def test_a_reward_file_must_give_every_outcome(tmp_path):
    path = tmp_path / "partial.yaml"
    path.write_text(yaml.safe_dump({"rewards": {"eat": 1}}))
    with pytest.raises(rw.RewardFileError, match="doesn't give"):
        rw.load(str(path))
