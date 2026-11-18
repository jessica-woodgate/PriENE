from mesa import Model
import pandas as pd
import numpy as np
import json
import os
import tensorflow
from tensorflow import keras
from . import seeding
from .checkpoints import recorded_agent_types
from .harvest_exception import FileExistsException
from .harvest_exception import MixedSocietyException
from os.path import exists

class SocietyModel(Model):
    """
    What every society of learning agents shares, whatever its environment: seeding, choosing the baseline half of a mixed society,
    reports written one episode (or step) at a time, norm bookkeeping, and means over agents. HarvestModel and FruitMarketModel build on it.

    run_simulation drives a model through step(), finish_episode(collect_data), training, epsilon, min_epsilon, episode, max_episodes and
    _mean_reward(). Agents ask their model for observe(agent), observation_size(agent), agent_actions(agent), action_mask(agent) and
    get_society_well_being(observer, norms_observation, ethics_observation).
    Subclasses set seed (via _set_seed), num_agents, schedule, training, filepath, episode, day, living_agents, emerged_norms,
    previous_learned_norms and societal_norm_emergence_threshold.
    """

    def get_num_agents(self):
        return self.num_agents

    def _set_seed(self, seed):
        """
        Record the run seed every random stream is derived from (src/seeding.py), and make TensorFlow's ops deterministic (process-wide,
        so it stays on for any later models in the same process: a run is meant to have a process of its own). The global seeds
        (python's random, numpy's global RNG, tensorflow's) are set too, only so that any draw not given a stream of its own is still
        reproducible; none is relied on. Must run before any agents are created. seed=None leaves the run unseeded.
        """
        self.seed = seed
        if seed is None:
            return
        keras.utils.set_random_seed(seed)
        tensorflow.config.experimental.enable_op_determinism()
        self.reset_randomizer(seed)

    #a training run saves its checkpoint every this many episodes, and at the end of the run (set by run.py's --checkpoint-every)
    checkpoint_every = 1
    REPLAY_BUFFER_FILE = "replay_buffer.npz"

    def _decide_checkpoint(self, finishing_run):
        """
        Whether the episode being finished saves a checkpoint (read by each agent's finish_episode): every checkpoint_every episodes,
        and when the run finishes
        """
        self.saving_checkpoint = self.training and (finishing_run or self.episode % self.checkpoint_every == 0)

    def _replay_buffer_paths(self):
        #one per agent type (agents of a type share one buffer), kept in the folder of the type's lowest-id agent:
        #<checkpoint folder>/<type>/agent_<id>/replay_buffer.npz
        paths = {}
        for agent in sorted(self.schedule.agents, key=lambda a: a.unique_id):
            module = getattr(agent, "decision_module", None)
            if module is not None and hasattr(module, "q_checkpoint_path") and agent.agent_type in self.shared_replay_buffers:
                paths.setdefault(agent.agent_type, os.path.join(os.path.dirname(module.q_checkpoint_path), self.REPLAY_BUFFER_FILE))
        return paths

    def save_replay_buffers(self):
        """
        Save each agent type's replay buffer with the checkpoint, so a continued run learns from the experiences the run it continues had
        """
        for agent_type, path in self._replay_buffer_paths().items():
            buffer = self.shared_replay_buffers[agent_type]
            writing = path[:-len(".npz")] + ".writing.npz"
            np.savez(writing, **{key: np.asarray(values) for key, values in buffer.items()})
            os.replace(writing, path)

    def restore_replay_buffers(self):
        """
        A continued run's replay buffers, as its checkpoint saved them (left empty for a checkpoint saved without them)
        """
        for agent_type, path in self._replay_buffer_paths().items():
            if os.path.exists(path):
                with np.load(path) as saved:
                    buffer = self.shared_replay_buffers[agent_type]
                    for key in buffer:
                        buffer[key][:] = list(saved[key])

    def _start_episode_streams(self):
        """
        Reseed the environment streams for the episode starting (self.episode), from the seed and the episode alone; here, Mesa's
        model RNG, which decides the order agents act in. Models with other environment streams add theirs
        """
        if self.seed is not None:
            self.reset_randomizer(seeding.episode_seed(self.seed, self.episode, "activation"))

    def _choose_agent_types(self, society_mix):
        """
        Choose which agents are baseline: none in a homogeneous society; in a mixed society, half the agents, chosen from the seed (see
        _choose_baseline_ids), since agent ids decide resources in some scenarios and neither type should always get the better ones
        """
        self.baseline_ids = set() if society_mix == "homogeneous" else self._choose_baseline_ids(self._mixed_society_rng())

    def _use_recorded_agent_types(self, society_mix, checkpoint_path):
        """
        A test of a mixed society trained together has the agent types its training recorded (beside its checkpoints), not ones chosen
        from this run's seed: a test run is seeded from the test seed, which needn't choose the agents its training did
        """
        if self.training or society_mix == "homogeneous" or not checkpoint_path:
            return
        recorded = recorded_agent_types(checkpoint_path)
        if recorded:
            self.baseline_ids = {int(name.split("_")[1]) for name, agent_type in recorded.items() if agent_type == "baseline"}

    def _mixed_society_rng(self):
        #its own stream from the seed, so choosing the baseline agents doesn't shift any other random draws
        return seeding.run_rng(self.seed, "mixed_society")

    def _choose_baseline_ids(self, rng):
        """
        A random half of the agent ids (scenarios with layouts that must keep both types together or apart override this)
        """
        return {int(agent_id) for agent_id in rng.choice(self.num_agents, self._half_population(), replace=False)}

    def _agent_type_of(self, agent_id, agent_type):
        return "baseline" if agent_id in self.baseline_ids else agent_type

    def _half_population(self):
        """
        Size of each half of a mixed society (agent_type and baseline), which needs an even number of agents
        """
        if self.num_agents % 2 != 0:
            raise MixedSocietyException(f"half the agents are baseline, so the number of agents must be even, got {self.num_agents}")
        return self.num_agents // 2

    def _start_report(self, report_path, columns):
        """
        Start a report written one episode (or step) at a time with its header, never overwriting one, unless a continued training run
        is carrying on its report
        """
        if self.training and self.episode > 1 and exists(report_path):
            self._continue_report(report_path, columns)
        else:
            if exists(report_path):
                raise FileExistsException(report_path)
            pd.DataFrame(columns=columns).to_csv(report_path, mode='a', index=False)

    def _continue_report(self, report_path, columns=None):
        """
        A continued training run carries on its report, without the episodes it is about to train again, and with the report's current
        columns (empty in earlier rows for any it didn't have); written beside it and moved over it, so stopping now leaves the report whole
        """
        reports = pd.read_csv(report_path)
        reports = reports[reports["episode"] < self.episode]
        if columns is not None and list(reports.columns) != list(columns):
            reports = reports.reindex(columns=columns)
        writing_path = report_path+".writing"
        reports.to_csv(writing_path, index=False)
        os.replace(writing_path, report_path)

    def _report_row(self, fields, values, extra_values):
        """One report row: each field's value, then each column a scenario adds"""
        return pd.DataFrame({**{column: [value] for column, value in zip(fields, values, strict=True)},
                             **{column: [value] for column, value in extra_values.items()}})

    def _num_agent_norms(self, agent):
        """
        Size of an agent's norm base: behaviours in its behaviour base (rule-based), or behaviours
        it has learned (Bayesian, whose candidate set is fixed). None if norms aren't tracked.
        """
        if not self.track_norms:
            return None
        if self.norms_module_type == "rule_based":
            return len(agent.norms_module.behaviour_base)
        return agent.norms_module.num_learned_behaviours()

    def _append_norm_dict_to_file(self, norm_dictionary, filename):
        """
        Persists the current episode's emerged norms into filename, keyed by episode number
        Rewrites the whole file from a full in-memory dict each call so the file on disk is always valid,
        complete JSON, rather than relying on the run reaching its final episode to write a closing brace.
        Once this model has written the file, a new episode is appended instead (see _append_norm_episode), which gives the same text
        """
        if self._append_norm_episode(norm_dictionary, filename):
            return
        if exists(filename):
            with open(filename, "r") as file:
                try:
                    all_episodes = json.load(file)
                except json.JSONDecodeError:
                    print(f"Warning: {filename} was not valid JSON (likely left over from an interrupted run); starting a new norms file")
                    all_episodes = {}
        else:
            all_episodes = {}
        norm_list = [{key: value} for key, value in norm_dictionary.items()]
        all_episodes[str(self.episode)] = norm_list
        with open(filename, "w") as file:
            json.dump(all_episodes, file, indent=4)
        self._norms_file_written = (filename, set(all_episodes), os.stat(filename))

    def _append_norm_episode(self, norm_dictionary, filename):
        """
        Add this episode to a norms file this model wrote last (unchanged since: same size and modification time) without reading and
        rewriting it: json.dump(..., indent=4) writes each episode of the dict as the same text whatever else is in it, so the episode's
        text goes in before the closing "\\n}" and the file is exactly what the full rewrite would write. Returns False (nothing written)
        when the file isn't the one this model last wrote, or already has this episode (which the full rewrite replaces in place)
        """
        written = getattr(self, "_norms_file_written", None)
        key = str(self.episode)
        if written is None or written[0] != filename or key in written[1] or not exists(filename):
            return False
        before = os.stat(filename)
        if (before.st_size, before.st_mtime_ns) != (written[2].st_size, written[2].st_mtime_ns):
            return False
        norm_list = [{name: value} for name, value in norm_dictionary.items()]
        #'{\n    "<episode>": [...]\n}' without its braces: the episode's text as it appears in the whole dict
        episode_text = json.dumps({key: norm_list}, indent=4)[1:-2]
        with open(filename, "rb+") as file:
            file.seek(before.st_size - 2)
            if file.read(2) != b"\n}":
                return False
            file.seek(before.st_size - 2)
            file.write(("," + episode_text + "\n}").encode())
        written[1].add(key)
        self._norms_file_written = (filename, written[1], os.stat(filename))
        return True

    def _analyse_agent_norms(self):
        """
        Collects one row per agent summarising the norms it has learned by the end of this episode and
        appends them to agent_norms_<filepath>.csv. Gained/lost compare against the behaviours the same
        agent had learned at the end of the previous episode (posteriors decay between episodes, so
        behaviours can be lost as well as gained). Cooperative norms are learned commitments to throw a
        berry to another agent ("IF ... THEN throw"); uncooperative norms are learned prohibitions on
        throwing ("IF ... THEN NOT throw"). Agents' throw_X actions are collapsed to "throw" for norms.
        """
        rows = []
        for agent in self.schedule.agents:
            if agent.agent_type == "berry":
                continue
            learned = agent.norms_module.get_learned_behaviours()
            learned_ids = {b["id"] for b in learned}
            previous_ids = self.previous_learned_norms.get(agent.unique_id, set())
            rows.append({"episode": self.episode,
                         "agent_id": agent.unique_id,
                         "agent_type": agent.agent_type,
                         "end_day": self.day,
                         "num_learned_norms": len(learned),
                         "num_learned_prohibitions": sum(b["type"] == "prohibition" for b in learned),
                         "num_learned_commitments": sum(b["type"] == "commitment" for b in learned),
                         "num_gained_norms": len(learned_ids - previous_ids),
                         "num_lost_norms": len(previous_ids - learned_ids),
                         "mean_learned_posterior": float(np.mean([b["posterior"] for b in learned])) if learned else None,
                         "num_observed_behaviours": sum(agent.norms_module.behaviour_counts(*view)[0] > 0 for view in agent.norms_module.iter_behaviours()),
                         "num_learned_cooperative_norms": sum(b["type"] == "commitment" and b["action"] == "throw" for b in learned),
                         "num_learned_uncooperative_norms": sum(b["type"] == "prohibition" and b["action"] == "throw" for b in learned)})
            self.previous_learned_norms[agent.unique_id] = learned_ids
        pd.DataFrame(rows, columns=self._agent_norms_columns()).to_csv(self._agent_norms_csv_filename(), header=None, mode='a', index=False)

    def _agent_norms_csv_filename(self):
        return "data/results/current_run/agent_norms_"+self.filepath+".csv"

    def _agent_norms_columns(self):
        return ["episode", "agent_id", "agent_type", "end_day", "num_learned_norms", "num_learned_prohibitions",
                "num_learned_commitments", "num_gained_norms", "num_lost_norms", "mean_learned_posterior",
                "num_observed_behaviours", "num_learned_cooperative_norms",
                "num_learned_uncooperative_norms"]

    def _check_emerged_norms(self):
        """
        function to check the current emerged norms at this step and update the norms that have emerged across the whole episode
        for each living agent, add the behaviours in its behaviour base to society_norms_snapshot
        filter society_norms_snapshot by emergence threshold
        update norms emerged across the whole episode with the norms emerged at this step
        """
        if len(self.living_agents) < 2:
            return
        emergence_threshold = len(self.living_agents) * self.societal_norm_emergence_threshold
        society_norms_snapshot = {}
        for agent in self.schedule.agents:
            if agent.agent_type != "berry" and not agent.off_grid:
                for norm_name, norm_value in agent.norms_module.behaviour_base.items():
                    #the society's totals at this step: reward, numerosity (times used by individual agents) and fitness summed over agents,
                    #and adoption, the number of agents with the behaviour in their behaviour base
                    totals = society_norms_snapshot.setdefault(norm_name, {"reward": 0, "numerosity": 0, "fitness": 0, "adoption": 0})
                    for key in ["reward", "numerosity", "fitness"]:
                        totals[key] += norm_value[key]
                    totals["adoption"] += 1
        for norm_name, totals in society_norms_snapshot.items():
            if totals["adoption"] >= emergence_threshold:
                self._update_episode_emerged_norm(norm_name, totals)

    def _update_episode_emerged_norm(self, norm_name, norm_value):
        """
        Overwrite an emerged norm's statistics with the society's current totals (agents' behaviour
        bases already hold running totals for the episode) and track when the norm was held:
        first and last day above the emergence threshold, and number of days held
        """
        norm = self.emerged_norms.setdefault(norm_name, {"first_day_held": self.day, "days_held": 0})
        #adoption: the number of living agents with the norm in their behaviour base on the last day it was held
        for key in ["reward", "numerosity", "fitness", "adoption"]:
            norm[key] = norm_value[key]
        norm["last_day_held"] = self.day
        norm["days_held"] += 1

    def _mean_loss(self):
        return self._mean_over_agents(lambda agent: agent.get_mean_loss()) if self.training else 0

    def _mean_reward(self):
        return self._mean_over_agents(lambda agent: agent.total_episode_reward)

    def _mean_epsilon(self):
        return self._mean_over_agents(lambda agent: agent.get_epsilon())

    def _mean_over_agents(self, value):
        """
        Mean of value(agent) over every agent, dead or alive (summed in schedule order); 0 when they sum to 0
        """
        m = 0
        for agent in self.schedule.agents:
            if agent.agent_type != "berry":
                m += value(agent)
        if m == 0:
            return 0
        return m / self.num_agents

def gini(values):
    """
    Gini coefficient of values (0 = all equal); 0 when they sum to 0
    """
    x = sorted(values)
    s = sum(x)
    if s == 0:
        return 0
    n = len(x)
    b = sum(xi * (n - i) for i, xi in enumerate(x)) / (n * s)
    return 1 + (1 / n) - 2 * b
