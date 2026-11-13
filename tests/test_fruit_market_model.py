"""
FruitMarketModel with a stand-in for the substrate (tests/fruit_market_stand_in.py), so it runs without Melting Pot: the substrate's
reward is what agents learn from, plus the ethics sanction for ethics agents; grappling can be masked out; norm labels and reports.
"""
import os
import numpy as np
import pandas as pd
import pytest
from src.fruit_market import substrate as fm
from src.fruit_market.model import FruitMarketModel
from src.fruit_market.features import observation_size
from tests.fruit_market_stand_in import StandInBackend

ROLES = ["apple_farmer", "apple_farmer", "banana_farmer", "banana_farmer"]

@pytest.fixture
def in_tmp_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path

def model(agent_type="utilitarian", training=True, max_days=5, backend=None, checkpoint_path="checkpoints/", filepath="fm_train", **options):
    return FruitMarketModel("homogeneous", len(ROLES), agent_type, 2, max_days, training, checkpoint_path, True, False, filepath=filepath,
                            seed=7, backend=backend or StandInBackend(ROLES), **options)

def script(m, actions_by_agent):
    """
    Make each agent choose its actions in turn from actions_by_agent (then wait)
    """
    for agent in m.schedule.agents:
        queue = list(actions_by_agent.get(agent.unique_id, []))
        agent.decision_module.choose_action = lambda observation, queue=queue: queue.pop(0) if queue else 0

def tree_north_of_agent_0():
    trees = np.zeros((fm.MAP_HEIGHT, fm.MAP_WIDTH), dtype=int)
    trees[14, 10] = fm.RIPE_APPLE
    return trees

def test_agents_choose_among_the_substrates_25_actions_and_observe_every_feature(in_tmp_dir):
    m = model()
    agent = m.schedule.agents[0]
    assert agent.actions == fm.ACTION_NAMES
    assert agent.decision_module.n_actions == 25
    assert len(m.observe(agent)) == agent.n_features == observation_size(4)

def test_with_grappling_off_agents_never_hold_shove_or_pull(in_tmp_dir):
    backend = StandInBackend(ROLES)
    m = model(max_days=60, backend=backend, grappling=False)
    for _ in range(60):
        m.step()
    chosen = {action for step in backend.actions for action in step}
    assert chosen.isdisjoint(fm.GRAPPLING_ACTIONS)
    assert len(chosen) == fm.NUM_ACTIONS - len(fm.GRAPPLING_ACTIONS)
    #the network still has an output for every action, so checkpoints are the same shape with grappling on or off
    assert m.schedule.agents[0].decision_module.q_network.dqn.n_actions == fm.NUM_ACTIONS

def test_baseline_agents_learn_from_the_substrates_reward_alone(in_tmp_dir):
    backend = StandInBackend(ROLES, inventories=[(0, 1), (1, 0), (0, 0), (0, 0)])
    m = model(agent_type="baseline", backend=backend)
    script(m, {0: [8], 1: [7]})
    m.step()
    a0, a1 = m.schedule.agents[:2]
    #an apple farmer eating a banana (preferred) gets 8, eating an apple 1
    assert (a0.current_env_reward, a0.current_sanction, a0.current_reward) == (8.0, None, 8.0)
    assert (a1.current_env_reward, a1.current_reward) == (1.0, 1.0)

def test_an_attributed_sanction_rewards_only_the_agent_whose_action_helped(in_tmp_dir):
    m = model(backend=StandInBackend(ROLES, trees=tree_north_of_agent_0()))
    script(m, {0: [1]})
    m.step()
    sanctions = [agent.current_sanction for agent in m.schedule.agents]
    assert sanctions == [3.5, 0.0, 0.0, 0.0]
    assert m.schedule.agents[0].current_reward == 3.5
    assert m.schedule.agents[0].current_label == "harvest own fruit"

def test_a_joint_sanction_rewards_everyone_for_one_agents_harvest(in_tmp_dir):
    m = model(backend=StandInBackend(ROLES, trees=tree_north_of_agent_0()), sanction_mode="joint")
    script(m, {0: [1]})
    m.step()
    assert [agent.current_sanction for agent in m.schedule.agents] == [3.5, 3.5, 3.5, 3.5]

def test_a_joint_sanction_punishes_utilitarians_for_time_passing(in_tmp_dir):
    m = model(sanction_mode="joint")
    m.step()
    assert [agent.current_sanction for agent in m.schedule.agents] == [-3.5] * 4

def test_trades_are_attributed_to_both_traders(in_tmp_dir):
    #agent 1 (apple farmer) and agent 2 (banana farmer) are within trading radius; a fair 1:1 trade
    backend = StandInBackend(ROLES, inventories=[(0, 0), (1, 0), (0, 1), (0, 0)])
    m = model(agent_type="egalitarian", backend=backend)
    script(m, {1: [17], 2: [21]})
    m.step()
    assert m.state.trade_partner(1) == 2
    assert m.state.players[1].inventory == (0, 1) and m.state.players[2].inventory == (1, 0)
    #need well-being counts fruit, not which: the trade changes nobody's well-being
    assert m.schedule.agents[1].current_sanction == 0.0
    assert m.schedule.agents[1].current_label == "offer fair"

def test_a_test_run_writes_fruit_market_reports_without_touching_harvest_ones(in_tmp_dir):
    trainer = model(max_days=3)
    for _ in range(3):
        trainer.step()
    tester = model(training=False, max_days=3, filepath="fm_test")
    for _ in range(3):
        tester.step()
    agent_report = pd.read_csv("data/results/current_run/fm_agent_reports_fm_test.csv")
    episode_report = pd.read_csv("data/results/current_run/fm_model_episode_reports_fm_test.csv")
    assert list(agent_report.columns) == FruitMarketModel.AGENT_REPORT_FIELDS
    assert list(episode_report.columns) == FruitMarketModel.EPISODE_REPORT_FIELDS
    assert len(agent_report) == 3 * len(ROLES) and len(episode_report) == 1
    assert not any(name.startswith(("agent_reports_", "model_episode_reports_")) for name in os.listdir("data/results/current_run"))

def test_each_episode_gets_its_own_substrate_seed_from_the_run_seed(in_tmp_dir):
    backend = StandInBackend(ROLES)
    m = model(max_days=2, backend=backend)
    for _ in range(4):
        m.step()
    assert len(backend.seeds) == 3 and len(set(backend.seeds)) == 3
    again = StandInBackend(ROLES)
    other = model(max_days=2, backend=again, checkpoint_path="checkpoints2/", filepath="fm_train_again")
    for _ in range(4):
        other.step()
    assert again.seeds == backend.seeds
