from src.harvest_exception import UnrecognisedPrinciple
import numpy as np

class EthicsModule():
    """
    on each step, agent updates it's ability to act cooperatively (has berries; is healthy)
    agent calls update_state which tracks the measure of well-being before the agent acts
    after acting, agent calls get_sanction which calls the chosen principle to generate a sanction indicating alignment
    """
    def __init__(self,agent_id,shaped_reward):
        self.agent_id = agent_id
        self._shaped_reward = shaped_reward
        self._can_help = False
        self._current_principle = None
        self._society_well_being = None
        self._measure_of_well_being = None
        self._number_of_minimums = None
    
    def update_state(self, principle, society_well_being, day, can_help):
        self.day = day
        self._can_help = can_help
        self._calculate_social_welfare(principle, society_well_being)
    
    def get_sanction(self, society_well_being):
        if self.current_principle == "maximin":
            return self._maximin_sanction(self._measure_of_well_being, self._number_of_minimums, society_well_being)
        elif self.current_principle == "egalitarian":
            return self._egalitarian_sanction(self._measure_of_well_being, society_well_being)
        elif self.current_principle == "utilitarian":
            return self._utilitarian_sanction(self._measure_of_well_being, society_well_being)
        elif self.current_principle == "deon_egalitarian":
            return self._deon_egalitarian_sanction(society_well_being)

    def get_egalitarian_loss(self, society_resources):
        loss = self._egalitarian_welfare(society_resources)
        return loss/10
    
    def _calculate_social_welfare(self, principle, society_well_being):
        self.current_principle = principle
        if principle == "maximin":
            self._measure_of_well_being, self._number_of_minimums = self._maximin_welfare(society_well_being)
        elif principle == "egalitarian":
            self._measure_of_well_being = self._egalitarian_welfare(society_well_being)
        elif principle == "utilitarian":
            self._measure_of_well_being = self._utilitarian_welfare(society_well_being)
        elif principle == "deon_egalitarian":
            return
        else:
            raise UnrecognisedPrinciple(principle)

    def _maximin_welfare(self, society_well_being):
        min_value = min(society_well_being)
        count = np.count_nonzero(society_well_being==min_value)
        return min_value, count

    def _egalitarian_welfare(self, society_well_being):
        n = len(society_well_being)
        total = sum(society_well_being)
        if total == 0:
            return 0
        ideal = total/n
        loss = sum(abs(x - ideal) for x in society_well_being)
        return loss
    
    def _utilitarian_welfare(self, society_well_being):
        return sum(society_well_being)
    
    def _maximin_sanction(self, previous_min, number_of_previous_mins, society_well_being):
        current_min, current_number_of_current_mins = self._maximin_welfare(society_well_being)
        current_number_of_previous_mins = np.count_nonzero(society_well_being==previous_min)
        #if the global min has been made better, pos reward
        if current_min > previous_min:
            return self._shaped_reward
        #if the global min has been made worse, neg reward
        elif current_min < previous_min:
            return -self._shaped_reward
        #if the global min has not changed, but there are fewer instances of it, pos reward
        elif current_number_of_previous_mins < number_of_previous_mins and current_min == previous_min:
            return self._shaped_reward
        #if the global min has not changed, and there are more instances of it, neg reward
        elif current_number_of_previous_mins > number_of_previous_mins and current_min == previous_min:
            return -self._shaped_reward
        return 0
    
    def _egalitarian_sanction(self, previous_loss, society_well_being):
        current_loss = self._egalitarian_welfare(society_well_being)
        if previous_loss > current_loss:
            return self._shaped_reward
        elif previous_loss < current_loss and self._can_help:
            return -self._shaped_reward
        return 0
    
    def _utilitarian_sanction(self, previous_welfare, society_well_being):
        current_welfare = self._utilitarian_welfare(society_well_being)
        if current_welfare > previous_welfare:
            return self._shaped_reward
        elif current_welfare < previous_welfare and self._can_help:
            return -self._shaped_reward
        return 0
    
    def _deon_egalitarian_sanction(self, society_well_being):
        loss = self._egalitarian_welfare(society_well_being)
        if loss == 0:
            return self._shaped_reward
        else:
            return -loss/100