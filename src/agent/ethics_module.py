from src.harvest_exception import UnrecognisedPrinciple

class EthicsModule():
    """
    on each step, agent updates it's ability to act cooperatively (has berries)
    agent calls update_state which tracks the measure of well-being before the agent acts
    after acting, agent calls get_sanction which calls the chosen principle to generate a sanction indicating alignment
    """
    def __init__(self,shaped_reward):
        self._shaped_reward = shaped_reward
        self._have_berries = False
        self._current_principle = None
        self._society_well_being = None
        self._measure_of_well_being = None
    
    def update_state(self, principle, society_well_being, have_berries=None):
        if have_berries != None:
            self.have_berries = have_berries
        self._calculate_social_welfare(principle, society_well_being)
    
    def _calculate_social_welfare(self, principle, society_well_being):
        self.current_principle = principle
        if principle == "maximin":
            self.measure_of_well_being = self._maximin_welfare(society_well_being)
        elif principle == "egalitarian":
            self.measure_of_well_being = self._egalitarian_welfare(society_well_being)
        elif principle == "utilitarian":
            self.measure_of_well_being = self._utilitarian_welfare(society_well_being)
        else:
            raise UnrecognisedPrinciple(principle)
    
    # def get_maximin_welfare(self, ordered_agents):
    #     agent_in_min = []
    #     self_in_min = False
    #     for a in ordered_agents:
    #         if a.days_left_to_live == ordered_agents[-1].days_left_to_live:
    #             agent_in_min.append(a)
    #             if a.unique_id == self.unique_id:
    #                 self_in_min = True
    #     #returns the minimum days left to live, the agents in that list, and whether you are in that list
    #     return ordered_agents[-1].days_left_to_live, agent_in_min, self_in_min

    def _maximin_welfare(self, society_well_being):
        return min(society_well_being)

    def _egalitarian_welfare(self, society_well_being):
        #get gini index of utility
        gini = self._calculate_gini(society_well_being)
        return gini
    
    def _utilitarian_welfare(self, society_well_being):
        return sum(society_well_being)
    
    def _calculate_gini(self, series):
        #sort series in ascending order
        x = sorted(series)
        s = sum(x)
        if s == 0:
            return 0
        N = len(series)
        #for each element xi, compute xi * (N - i); divide by num agents * sum
        B = sum(xi * (N - i) for i, xi in enumerate(x)) / (N * s)
        #
        return 1 + (1 / N) - 2 * B
    
    def get_sanction(self, society_well_being):
        if self.current_principle == "maximin":
            return self._maximin_sanction(self.measure_of_well_being, society_well_being)
        elif self.current_principle == "egalitarian":
            return self._egalitarian_sanction(self.measure_of_well_being, society_well_being)
        elif self.current_principle == "utilitarian":
            return self._utilitarian_sanction(self.measure_of_well_being, society_well_being)
        
    #after acting, look to see if you improved the minimum experience or not
    # def maximin(self, min_well_being, agents_in_min, self_in_min):
    #     for a in agents_in_min:
    #         #if the minimum experience was improved, positive shaped reward
    #         if a.days_left_to_live > min_well_being:
    #             return self.shaped_reward
    #     #else, negative
    #     if self_in_min == False and self.have_berries:
    #             return -self.shaped_reward
    #     return 0
    def _maximin_sanction(self, previous_min, society_well_being):
        current_min = self._maximin_welfare(society_well_being)
        if previous_min > current_min:
            return self._shaped_reward
        if previous_min < current_min and self.have_berries:
            return -self._shaped_reward
        return 0
    
    def _egalitarian_sanction(self, previous_gini, society_well_being):
        current_gini = self._egalitarian_welfare(society_well_being)
        if previous_gini > current_gini:
            return -self._shaped_reward
        elif previous_gini < current_gini and self.have_berries:
            return self._shaped_reward
        return 0
    
    def _utilitarian_sanction(self, previous_welfare, society_well_being):
        current_welfare = self._utilitarian_welfare(society_well_being)
        if current_welfare > previous_welfare:
            return self._shaped_reward
        elif current_welfare < previous_welfare and self.have_berries:
            return -self._shaped_reward
        return 0