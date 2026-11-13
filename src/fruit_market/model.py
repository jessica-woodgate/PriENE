import numpy as np
import pandas as pd
from mesa.time import BaseScheduler
from ..society_model import SocietyModel, gini
from ..agent.learning_agent import NORMS_MODULE_TYPES
from ..harvest_exception import NumAgentsException, UnrecognisedNormsModule, FileExistsException
from . import substrate as fm
from .agent import FruitMarketAgent
from .features import (observation, observation_size, compatible, within_radius, hungry_players_nearby, FEATURE_INDEX,
                       OBSERVATION_MODES, nearest, ripe_trees, MAX_DISTANCE)
from .well_being import well_being, WELL_BEING_MODES
from .sanction import sanction_vectors, SANCTION_MODES
from .labels import norm_label, move_effect_flags, COOPERATIVE_NORM_LABELS, AGGRESSIVE_NORM_LABELS
from os.path import exists

class FruitMarketModel(SocietyModel):
    """
    A society of FruitMarketAgents playing Melting Pot's Fruit Market (docs/fruit_market_design.md). Every step, each agent chooses one of
    the substrate's 25 actions from the observation before the step, the substrate steps every player at once, and each agent learns from
    the substrate's reward (plus, unless baseline, an ethics sanction) and its next observation.
    Instance variables (as well as SocietyModel's):
        max_days -- steps per episode (the substrate's 1000 by default)
        roles -- each player's role, the substrate's own assignment: the first half apple farmers, the rest banana farmers
        grappling -- whether agents may hold, shove and pull; if not, those actions are masked out of their choices
        well_being_mode -- "need" or "preference" (see well_being.py)
        sanction_mode -- "attributed" or "joint" (see sanction.py)
        observation_mode -- "global" or "view" (spatial features only within the player's view)
        backend -- the substrate (backend.MeltingPotBackend, or a stand-in in tests)
        state -- the substrate's state after the last step (a FruitMarketState)
        last_eaten -- the fruit each player last ate this episode (None before it has eaten)
    """
    AGENT_REPORT_FIELDS = ["agent_id", "agent_type", "role", "episode", "step", "action", "action_name", "norm_label", "env_reward",
                           "ethics_sanction", "reward", "hunger", "hungry_steps", "stamina", "own_fruit", "other_fruit", "apples", "bananas",
                           "well_being", "well_being_other_mode", "ate_own", "ate_other", "harvested", "trade_partner", "fruit_given",
                           "fruit_received", "offer_posted", "offer_cancelled", "offer_invalid", "offer_public", "grapple_action",
                           "grappled_by", "position_x", "position_y", "on_river", "move_effects", "num_norms"]
    EPISODE_REPORT_FIELDS = ["episode", "end_step", "epsilon", "mean_reward", "mean_loss", "mean_env_reward", "mean_sanction",
                             "total_fruit_eaten", "min_fruit_eaten", "max_fruit_eaten", "gini_fruit_eaten", "min_reward", "max_reward",
                             "gini_reward", "mean_well_being", "min_well_being", "max_well_being", "gini_well_being",
                             "mean_well_being_over_steps", "total_trades", "cross_role_trades", "mean_trade_rate", "autarky_rate",
                             "preferred_eat_rate", "invalid_offer_rate", "market_freeze_steps", "total_holds", "total_shoves",
                             "total_pulls", "mean_hunger", "steps_hungry", "mean_stamina", "num_emerged_norms", "seed"]
    #spawn key for the random stream of the substrate's seeds, one per episode
    SUBSTRATE_SEED_STREAM = 4

    def __init__(self, society_mix, num_agents, agent_type, max_episodes, max_days, training, checkpoint_path, write_data, track_norms,
                 filepath="", seed=None, norms_module_type="rule_based", start_episode=1, grappling=True, well_being_mode="need",
                 sanction_mode="attributed", observation_mode="global", backend=None):
        super().__init__()
        self._set_seed(seed)
        if num_agents <= 0 or num_agents % 2 != 0:
            raise NumAgentsException("an even number above 0 (half apple farmers, half banana farmers)", num_agents)
        if norms_module_type not in NORMS_MODULE_TYPES:
            raise UnrecognisedNormsModule(norms_module_type, NORMS_MODULE_TYPES)
        for value, options in [(well_being_mode, WELL_BEING_MODES), (sanction_mode, SANCTION_MODES), (observation_mode, OBSERVATION_MODES)]:
            if value not in options:
                raise ValueError(f"{value} is not one of {options}")
        if not 1 <= max_days <= fm.EPISODE_LENGTH:
            raise ValueError(f"steps per episode must be between 1 and {fm.EPISODE_LENGTH}, got {max_days}")
        self.num_agents = num_agents
        self.max_days = max_days
        self.max_episodes = max_episodes
        self.min_epsilon = 0.01
        self.day = 0
        self.episode = start_episode
        self.training = training
        self.write_data = write_data
        self.track_norms = track_norms
        self.norms_module_type = norms_module_type
        self.filepath = filepath
        self.grappling = grappling
        self.well_being_mode = well_being_mode
        self.sanction_mode = sanction_mode
        self.observation_mode = observation_mode
        self.societal_norm_emergence_threshold = 0.9
        self.emerged_norms = {}
        self.previous_learned_norms = {}
        self.epsilon = 1.0 if training else 0.0
        self.roles = ["apple_farmer"] * (num_agents // 2) + ["banana_farmer"] * (num_agents // 2)
        self.schedule = BaseScheduler(self)
        self.shared_replay_buffers = {}
        if backend is None:
            from .backend import MeltingPotBackend
            backend = MeltingPotBackend(self.roles)
        self.backend = backend
        self._init_reporters()
        self._choose_agent_types(society_mix)
        for i in range(num_agents):
            this_type = self._agent_type_of(i, agent_type)
            buffer = self.shared_replay_buffers.setdefault(this_type, {"s": [], "a": [], "r": [], "s_": [], "done": []})
            self.schedule.add(FruitMarketAgent(i, self, this_type, self.roles[i], training, checkpoint_path, self.epsilon, track_norms,
                                               shared_replay_buffer=buffer, norms_module_type=norms_module_type))
        self.living_agents = list(self.schedule.agents)
        self._start_episode()

    # ── what agents ask their model ─────────────────────────────

    def agent_actions(self, agent):
        return list(fm.ACTION_NAMES)

    def action_mask(self, agent):
        if self.grappling:
            return None
        mask = np.ones(fm.NUM_ACTIONS, dtype=bool)
        mask[list(fm.GRAPPLING_ACTIONS)] = False
        return mask

    def observation_size(self, agent):
        return observation_size(self.num_agents)

    def observe(self, agent, state=None, last_eaten=None):
        state = self.state if state is None else state
        others = self.get_society_well_being(agent, False, False, state, last_eaten)
        return observation(state, agent.unique_id, others, self.observation_mode == "view", self.well_being_mode)

    def get_society_well_being(self, observer, norms_observation, ethics_observation, state=None, last_eaten=None):
        """
        The current well-being of every other player (every player, observer included, for ethics), in index order; nobody dies in Fruit Market
        """
        records = self._records(self.state if state is None else state, self.last_eaten if last_eaten is None else last_eaten)
        values = [well_being(self.well_being_mode, *record) for record in records]
        if not ethics_observation:
            values = values[:observer.unique_id] + values[observer.unique_id + 1:]
        return np.array(values, dtype=float)

    def norm_antecedent_values(self, agent, state):
        p = state.players[agent.unique_id]
        found = nearest(p.position, ripe_trees(state.trees, p.own_fruit))
        compatible_nearby = sum(1 for q in state.players if q.index != p.index and within_radius(p.position, q.position) and compatible(p.offer, q.offer))
        return [p.hunger, p.stamina, p.held(p.own_fruit), p.held(p.other_fruit), 1 if any(p.offer) else 0, compatible_nearby,
                1.0 if found is None else found[0] / MAX_DISTANCE, hungry_players_nearby(state, p.index)]

    def norm_antecedent_specs(self):
        """
        How rule-based norms bucket norm_antecedent_values (docs/fruit_market_design.md §9.3)
        """
        return [{"boundaries": [0.5, 1.0], "labels": ["sated", "peckish", "hungry"]},
                {"boundaries": [0.34, 0.67], "labels": ["low stamina", "medium stamina", "high stamina"]},
                {"boundaries": [2, 5], "labels": ["few own fruit", "some own fruit", "many own fruit"], "zero_label": "no own fruit"},
                {"boundaries": [2, 5], "labels": ["few other fruit", "some other fruit", "many other fruit"], "zero_label": "no other fruit"},
                {"boundaries": [0.5], "labels": ["no offer", "offer posted"]},
                {"boundaries": [1], "labels": ["no partner nearby", "partner nearby"]},
                {"boundaries": [3 / MAX_DISTANCE, 10 / MAX_DISTANCE], "labels": ["own fruit near", "own fruit medium", "own fruit far"]},
                {"boundaries": [1], "labels": ["none hungry nearby", "hungry player nearby"]}]

    def bayesian_feature_specs(self):
        """
        Bayesian norms' predicates, on the observation's features (see features.FEATURE_NAMES)
        """
        specs = [("hunger", [0.5, 1.0]), ("stamina", [0.34, 0.67]), ("own_fruit", [0.1, 0.2, 0.5]), ("other_fruit", [0.1, 0.2, 0.5]),
                 ("offer_own", [-0.01, 0.01]), ("compatible_offers_nearby", [1 / 15]), ("own_tree_distance", [3 / MAX_DISTANCE, 10 / MAX_DISTANCE]),
                 ("hungry_players_nearby", [1 / 15])]
        return [{"name": name, "index": FEATURE_INDEX[name], "thresholds": thresholds, "repeated": False} for name, thresholds in specs]

    def get_day(self):
        return self.day

    def get_max_days(self):
        return self.max_days

    # ── stepping ────────────────────────────────────────────────

    def step(self):
        before, before_eaten = self.state, list(self.last_eaten)
        agents = self.schedule.agents
        actions = [agent.choose(self.observe(agent), self.norm_antecedent_values(agent, before)) for agent in agents]
        after = self.backend.step(actions)
        after_eaten = list(before_eaten)
        eaten, harvested = {}, {}
        for agent, action in zip(agents, actions):
            i = agent.unique_id
            fruit = fm.EAT_ACTIONS.get(action)
            ate = fruit is not None and before.players[i].held(fruit) >= 1
            eaten[i] = fruit if ate else None
            if ate:
                after_eaten[i] = fruit
            harvested[i] = self._harvested(before, after, i, eaten[i])
        before_records, after_records = self._records(before, before_eaten), self._records(after, after_eaten)
        self.state, self.last_eaten = after, after_eaten
        for agent, action in zip(agents, actions):
            i = agent.unique_id
            sanction = None
            if agent.agent_type != "baseline":
                vectors = sanction_vectors(self.sanction_mode, self.well_being_mode, before_records, after_records, i, after.trade_partner(i))
                sanction = agent.ethics_sanction(*vectors)
            label = norm_label(action, before, after, i, harvested[i])
            agent.settle(float(after.rewards[i]), sanction, self.observe(agent), label)
            self._record_step(agent, action, before, after, eaten[i], harvested[i])
        self.day += 1
        self._record_market(before, after)
        self.epsilon = self._mean_epsilon()
        if self.track_norms and self.norms_module_type == "rule_based":
            self._check_emerged_norms()
        if self.day >= self.max_days:
            self.finish_episode()

    def _harvested(self, before, after, index, eaten):
        """
        "own" or "other" if player index harvested that fruit this step, else None: its inventory change not explained by eating or trading
        """
        received = after.received(index)
        for f, fruit in enumerate(fm.FRUITS):
            change = after.players[index].inventory[f] - before.players[index].inventory[f] - received[f] + (1 if eaten == fruit else 0)
            if change > 0:
                return "own" if fruit == before.players[index].own_fruit else "other"
        return None

    def _records(self, state, last_eaten):
        return [(p.hunger, p.inventory, p.role, last) for p, last in zip(state.players, last_eaten)]

    def finish_episode(self, collect_data=True):
        if self.track_norms and collect_data:
            if self.norms_module_type == "rule_based":
                self._append_norm_dict_to_file(self.emerged_norms, "data/results/current_run/"+self.filepath+"_emerged_norms.json")
            else:
                self._analyse_agent_norms()
        for agent in self.schedule.agents:
            agent.finish_episode()
        if collect_data:
            self._collect_model_episode_data()
            if self.write_data and not self.training:
                pd.DataFrame(self.agent_rows, columns=self.AGENT_REPORT_FIELDS).to_csv(self._agent_report_path(), header=None, mode="a", index=False)
        self.episode += 1
        self._start_episode()

    def _start_episode(self):
        self.day = 0
        self.emerged_norms = {}
        self.last_eaten = [None] * self.num_agents
        self.agent_rows = []
        self.episode_totals = {"freeze_steps": 0, "offers": 0, "invalid_offers": 0, "well_being_sum": 0.0}
        self.agent_totals = {agent.unique_id: {"eaten": 0, "eaten_own": 0, "eaten_preferred": 0, "trades": 0, "cross_role_trades": 0,
                                               "given": 0, "received": 0, "holds": 0, "shoves": 0, "pulls": 0, "hunger": 0.0,
                                               "steps_hungry": 0, "stamina": 0.0, "env_reward": 0.0, "sanction": 0.0}
                             for agent in self.schedule.agents}
        for agent in self.schedule.agents:
            agent.reset()
        self.state = self.backend.reset(self._substrate_seed())

    def _substrate_seed(self):
        #from the seed and the episode, in a stream of its own: each episode's substrate is reproducible, and a continued run gets the
        #substrates the run it continues would have had
        if self.seed is None:
            return None
        return int(np.random.SeedSequence([self.seed, self.SUBSTRATE_SEED_STREAM, self.episode]).generate_state(1)[0]) + 1

    def reseed_for_continuing(self):
        if self.seed is None:
            return
        import keras
        seed = int(np.random.SeedSequence([self.seed, self.episode]).generate_state(1)[0])
        keras.utils.set_random_seed(seed)
        self.reset_randomizer(seed)

    # ── reports ─────────────────────────────────────────────────

    def _agent_report_path(self):
        return "data/results/current_run/fm_agent_reports_"+self.filepath+".csv"

    def _episode_report_path(self):
        return "data/results/current_run/fm_model_episode_reports_"+self.filepath+".csv"

    def _init_reporters(self):
        import os
        os.makedirs("data/results/current_run", exist_ok=True)
        if self.write_data and not self.training:
            if exists(self._agent_report_path()):
                raise FileExistsException(self._agent_report_path())
            pd.DataFrame(columns=self.AGENT_REPORT_FIELDS).to_csv(self._agent_report_path(), index=False)
        if self.write_data:
            self._start_report(self._episode_report_path(), self.EPISODE_REPORT_FIELDS)
        if self.track_norms and self.norms_module_type == "bayesian":
            if exists(self._agent_norms_csv_filename()):
                raise FileExistsException(self._agent_norms_csv_filename())
            pd.DataFrame(columns=self._agent_norms_columns()).to_csv(self._agent_norms_csv_filename(), index=False)

    def _record_step(self, agent, action, before, after, eaten, harvested):
        i = agent.unique_id
        p, q = before.players[i], after.players[i]
        totals = self.agent_totals[i]
        received = after.received(i)
        partner = after.trade_partner(i)
        offer = fm.offer_of(action)
        offer_public = any(q.offer)
        if eaten is not None:
            totals["eaten"] += 1
            totals["eaten_own"] += eaten == p.own_fruit
            totals["eaten_preferred"] += fm.taste_value(p.role, eaten) == fm.MOST_TASTY_REWARD
        if partner is not None:
            totals["trades"] += 1
            totals["cross_role_trades"] += after.players[partner].role != p.role
            totals["given"] += -sum(r for r in received if r < 0)
            totals["received"] += sum(r for r in received if r > 0)
        if offer is not None:
            self.episode_totals["offers"] += 1
            self.episode_totals["invalid_offers"] += not offer_public and partner is None
        grapple = fm.ACTION_NAMES[action].lower() if action in fm.GRAPPLING_ACTIONS else None
        if grapple is not None:
            totals[grapple + "s"] += 1
        totals["hunger"] += q.hunger
        totals["steps_hungry"] += q.hunger >= fm.HUNGER_LIMIT
        totals["stamina"] += q.stamina
        totals["env_reward"] += agent.current_env_reward
        totals["sanction"] += agent.current_sanction or 0
        if not (self.write_data and not self.training):
            return
        other_mode = "preference" if self.well_being_mode == "need" else "need"
        own, other = fm.FRUITS.index(q.own_fruit), fm.FRUITS.index(q.other_fruit)
        self.agent_rows.append([i, agent.agent_type, agent.role, self.episode, self.day, action, fm.ACTION_NAMES[action], agent.current_label,
                                agent.current_env_reward, agent.current_sanction, agent.current_reward, q.hunger, q.hungry_steps, q.stamina,
                                q.inventory[own], q.inventory[other], q.inventory[0], q.inventory[1],
                                well_being(self.well_being_mode, q.hunger, q.inventory, q.role, self.last_eaten[i]),
                                well_being(other_mode, q.hunger, q.inventory, q.role, self.last_eaten[i]),
                                int(eaten == q.own_fruit), int(eaten == q.other_fruit), harvested, partner,
                                -sum(r for r in received if r < 0), sum(r for r in received if r > 0), int(offer is not None),
                                int(action == fm.CANCEL_ACTION), int(offer is not None and not offer_public and partner is None), int(offer_public),
                                grapple, q.held_by, q.position[0], q.position[1], int(bool(fm.RIVER[q.position[1], q.position[0]])),
                                move_effect_flags(before, after, i), self._num_agent_norms(agent)])

    def _record_market(self, before, after):
        """
        A market freeze step: some pair of players within trading radius had compatible public offers before the step and neither traded
        """
        traders = {t.player_a for t in after.trades} | {t.player_b for t in after.trades}
        players = before.players
        frozen = any(compatible(a.offer, b.offer) and within_radius(a.position, b.position) and a.index not in traders and b.index not in traders
                     for a in players for b in players if a.index < b.index)
        self.episode_totals["freeze_steps"] += frozen
        self.episode_totals["well_being_sum"] += float(np.mean(self.get_society_well_being(None, False, True)))

    def _collect_model_episode_data(self):
        totals = list(self.agent_totals.values())
        eaten = [t["eaten"] for t in totals]
        rewards = [agent.total_episode_reward for agent in self.schedule.agents]
        final_well_being = self.get_society_well_being(None, False, True)
        eaten_total = sum(eaten)
        trades = sum(t["trades"] for t in totals) // 2
        given, received = sum(t["given"] for t in totals), sum(t["received"] for t in totals)
        values = [self.episode, self.day, self.epsilon, self._mean_reward(), self._mean_loss(),
                  float(np.mean([t["env_reward"] for t in totals])), float(np.mean([t["sanction"] for t in totals])),
                  eaten_total, min(eaten), max(eaten), gini(eaten), min(rewards), max(rewards), gini(rewards),
                  float(np.mean(final_well_being)), float(np.min(final_well_being)), float(np.max(final_well_being)), gini(final_well_being),
                  self.episode_totals["well_being_sum"] / max(self.day, 1), trades, sum(t["cross_role_trades"] for t in totals) // 2,
                  received / given if given else None,
                  sum(t["eaten_own"] for t in totals) / eaten_total if eaten_total else None,
                  sum(t["eaten_preferred"] for t in totals) / eaten_total if eaten_total else None,
                  self.episode_totals["invalid_offers"] / self.episode_totals["offers"] if self.episode_totals["offers"] else None,
                  self.episode_totals["freeze_steps"], sum(t["holds"] for t in totals), sum(t["shoves"] for t in totals),
                  sum(t["pulls"] for t in totals), sum(t["hunger"] for t in totals) / (self.num_agents * max(self.day, 1)),
                  sum(t["steps_hungry"] for t in totals), sum(t["stamina"] for t in totals) / (self.num_agents * max(self.day, 1)),
                  len(self.emerged_norms) if self.track_norms else None, self.seed]
        row = self._report_row(self.EPISODE_REPORT_FIELDS, values, {})
        if self.write_data:
            row.to_csv(self._episode_report_path(), header=None, mode="a", index=False)
        return row

    def _analyse_agent_norms(self):
        """
        As SocietyModel's, with Fruit Market's cooperative and uncooperative norms (docs/fruit_market_design.md §9.4): a cooperative norm
        is a learned commitment to a cooperative offer or prohibition of grappling, an uncooperative one the reverse
        """
        rows = []
        for agent in self.schedule.agents:
            learned = agent.norms_module.get_learned_behaviours()
            learned_ids = {b["id"] for b in learned}
            previous_ids = self.previous_learned_norms.get(agent.unique_id, set())
            def kind(b, commitments, prohibitions):
                return (b["type"] == "commitment" and b["action"] in commitments) or (b["type"] == "prohibition" and b["action"] in prohibitions)
            rows.append({"episode": self.episode, "agent_id": agent.unique_id, "agent_type": agent.agent_type, "end_day": self.day,
                         "num_learned_norms": len(learned),
                         "num_learned_prohibitions": sum(b["type"] == "prohibition" for b in learned),
                         "num_learned_commitments": sum(b["type"] == "commitment" for b in learned),
                         "num_gained_norms": len(learned_ids - previous_ids), "num_lost_norms": len(previous_ids - learned_ids),
                         "mean_learned_posterior": float(np.mean([b["posterior"] for b in learned])) if learned else None,
                         "num_observed_behaviours": sum(agent.norms_module.behaviour_counts(*view)[0] > 0 for view in agent.norms_module.iter_behaviours()),
                         "num_learned_cooperative_norms": sum(kind(b, COOPERATIVE_NORM_LABELS, AGGRESSIVE_NORM_LABELS) for b in learned),
                         "num_learned_uncooperative_norms": sum(kind(b, AGGRESSIVE_NORM_LABELS, COOPERATIVE_NORM_LABELS) for b in learned)})
            self.previous_learned_norms[agent.unique_id] = learned_ids
        pd.DataFrame(rows, columns=self._agent_norms_columns()).to_csv(self._agent_norms_csv_filename(), header=None, mode='a', index=False)
