"""
Scripted policies, to check candidate reward functions before any training (src/reward_checks): a ScriptedDecisionModule stands in for
an agent's DQNDecisionModule (the same interface), choosing an action from the agent's state by a fixed rule, and never learning.

A policy reads the agent's own state (health, berries) and the other agents' as they are (days left to live, whether they are alive,
and whether the agent can reach them, model.can_throw_to), not its observation, with the agent's own thresholds: it throws only with
at least low_health_threshold health (as a throw needs), and counts another agent as needy below low_days_left_threshold days left.
"Health would reach 0 on the next decay" is health - health_decay <= 0.

    random -- uniform over the agent's actions (move, eat and a throw to each other agent), from the agent's exploration stream
    passive -- eat every step. There is no action that does nothing (move is foraging), so with no berry its eat fails (no_berries):
        it eats a berry if it carries one, otherwise it does nothing but fail, and never forages
    eater -- eat if carrying a berry, otherwise move (forage); never throw
    hoarder -- move (forage), except eat a berry it carries when its health would reach 0 on the next decay
    sharer -- as eater, except that with at least low_health_threshold health, a berry, and a reachable living agent with fewer than
        low_days_left_threshold days left, it throws to the reachable living agent with the fewest days left (the lowest id of equals)
    altruist -- throw to the reachable living agent with the fewest days left whenever it can (a berry, at least low_health_threshold
        health, and a reachable living agent); otherwise eat a berry it carries if its health would reach 0 on the next decay, and
        move (forage) if not
    altruist_eater -- the altruist's first rule (throw to the worst-off reachable living agent whenever it can); otherwise as eater
    sharer_hoarder -- the sharer's first rule (throw to the neediest reachable living agent below low_days_left_threshold, when it can
        throw); otherwise as hoarder
"""


class ScriptedDecisionModule:
    POLICIES = ["random", "passive", "eater", "hoarder", "sharer", "altruist", "altruist_eater", "sharer_hoarder"]

    def __init__(self, policy, agent, rng):
        if policy not in self.POLICIES:
            raise ValueError(f"unknown scripted policy {policy!r}; the policies are {self.POLICIES}")
        self.policy = policy
        self.agent = agent
        self.rng = rng

    # ── the decision module's interface ─────────────────────────

    def choose_action(self, observation):
        return self.agent.actions.index(getattr(self, "_" + self.policy)())

    def learn(self, *args, **kwargs):
        pass

    def save_models(self, episode):
        pass

    def get_epsilon(self):
        return 0.0

    def start_episode(self):
        pass

    def get_mean_loss(self):
        return 0

    # ── what a policy reads ─────────────────────────────────────

    def _would_die_on_next_decay(self):
        return round(self.agent.health - self.agent.health_decay, 10) <= 0

    def _can_throw(self):
        return self.agent.berries > 0 and self.agent.health >= self.agent.low_health_threshold

    def _reachable_living_others(self):
        agent = self.agent
        return [a for a in agent.model.get_living_agents()
                if a.unique_id != agent.unique_id and not a.is_dead() and agent.model.can_throw_to(agent, a)]

    def _worst_off(self, agents):
        return min(agents, key=lambda a: (a.days_left_to_live, a.unique_id))

    @staticmethod
    def _throw_to(agent):
        #the agent's action for throwing to that agent (HarvestModel.agent_actions)
        return f"throw_{agent.unique_id}"

    # ── the policies ────────────────────────────────────────────

    def _random(self):
        return self.agent.actions[int(self.rng.integers(len(self.agent.actions)))]

    def _passive(self):
        return "eat"

    def _eater(self):
        return "eat" if self.agent.berries > 0 else "move"

    def _hoarder(self):
        return "eat" if self.agent.berries > 0 and self._would_die_on_next_decay() else "move"

    def _share_with_the_needy(self):
        """The sharer's first rule: a throw to the neediest reachable living agent below low_days_left_threshold, or None"""
        if self._can_throw():
            needy = [a for a in self._reachable_living_others() if a.days_left_to_live < self.agent.low_days_left_threshold]
            if needy:
                return self._throw_to(self._worst_off(needy))
        return None

    def _give_to_the_worst_off(self):
        """The altruist's first rule: a throw to the worst-off reachable living agent whenever it can, or None"""
        if self._can_throw():
            others = self._reachable_living_others()
            if others:
                return self._throw_to(self._worst_off(others))
        return None

    def _sharer(self):
        return self._share_with_the_needy() or self._eater()

    def _altruist(self):
        return self._give_to_the_worst_off() or self._hoarder()

    def _altruist_eater(self):
        return self._give_to_the_worst_off() or self._eater()

    def _sharer_hoarder(self):
        return self._share_with_the_needy() or self._hoarder()
