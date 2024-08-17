from src.harvest_exception import UnrecognisedPrinciple
import numpy as np

class EthicsModule():
    """
    Ethics Module (Algorithm 1) evaluates societal well-being before and after acting and generates a self-directed sanction
    Instance variables:
        sanction -- amount of reward to return to agent
        current_principle -- normative ethics principle
        society_well_being -- list of well-being for each living agent
        measure_of_well_being -- metric to evaluate well-being before and after acting (minimum experience)
        number_of_minimums -- number of agents which have minimum experience
    """
    def __init__(self,agent_id,sanction):
        #self.agent_id = agent_id
        self.sanction = sanction
        self.current_principle = None
        self.society_well_being = None
        self.measure_of_well_being = None
        self.number_of_minimums = None
    
    def update_social_welfare(self, principle, society_well_being):
        """
        Updates social welfare before agent acts: measure of well-being and number of minimums (Algorithm 1 Line 1)
        """
        self._calculate_social_welfare(principle, society_well_being)
    
    def get_sanction(self, society_well_being):
        """
        Obtain sanction from principle comparing current society well-being with previous well-being (Algorithm 1 Lines 3-8)
        """
        if self.current_principle == "maximin":
            return self._maximin_sanction(self.measure_of_well_being, self.number_of_minimums, society_well_being)
        elif self.current_principle == "egalitarian":
            return self._egalitarian_sanction(self.measure_of_well_being, society_well_being)
        elif self.current_principle == "utilitarian":
            return self._utilitarian_sanction(self.measure_of_well_being, society_well_being)
    
    def _calculate_social_welfare(self, principle, society_well_being):
        self.current_principle = principle
        if principle == "maximin":
            self.measure_of_well_being, self.number_of_minimums = self._maximin_welfare(society_well_being)
        elif principle == "egalitarian":
            self.measure_of_well_being = self._egalitarian_welfare(society_well_being)
        elif principle == "utilitarian":
            self.measure_of_well_being = self._utilitarian_welfare(society_well_being)
        else:
            raise UnrecognisedPrinciple(principle)

    def _maximin_welfare(self, society_well_being):
        min_value = min(society_well_being)
        num_mins = np.count_nonzero(society_well_being==min_value)
        #print("day",self.day,"agent", self.agent_id, "maximin welfare", society_well_being, "min is", min(society_well_being), "num_mins is", num_mins)
        return min_value, num_mins

    def _egalitarian_welfare(self, society_well_being):
        n = len(society_well_being)
        total = sum(society_well_being)
        ideal = total/n
        loss = sum(abs(x - ideal) for x in society_well_being)
        #print("day",self.day,"agent", self.agent_id, "egalitarian welfare", society_well_being, "n is", n, "total is", total, "ideal is", ideal, "loss is", loss)
        return loss
    
    def _utilitarian_welfare(self, society_well_being):
        #print("day",self.day,"agent", self.agent_id, "utilitarian welfare", society_well_being, "total is", sum(society_well_being))
        return sum(society_well_being)
        
    def _maximin_sanction(self, previous_min, number_of_previous_mins, society_well_being):
        current_min, current_number_of_current_mins = self._maximin_welfare(society_well_being)
        current_number_of_previous_mins = np.count_nonzero(society_well_being==previous_min)
        #if the global min has been made better, pos reward
        if current_min > previous_min:
            #print("day",self.day,"agent", self.agent_id, "current_min", current_min, "previous_min", previous_min, "returning pos reward")
            return self.sanction
        #if the global min has been made worse, neg reward
        elif current_min < previous_min:
            #print("day",self.day,"agent", self.agent_id, "current_min", current_min, "previous_min", previous_min, "returning neg reward")
            return -self.sanction
        #if the global min has not changed, but there are fewer instances of it, pos reward
        elif current_number_of_previous_mins < number_of_previous_mins and current_min == previous_min:
            #print("day",self.day,"agent", self.agent_id, "current_min", current_min, "previous_min", previous_min, "returning pos less numbers reward")
            return self.sanction
        #if the global min has not changed, and there are more or same number of instances of it, neg reward
        elif current_number_of_previous_mins > number_of_previous_mins and current_min == previous_min:
            #print("day",self.day,"agent", self.agent_id, "current_min", current_min, "previous_min", previous_min, "returning neg more numbers reward")
            return -self.sanction
        #print("day",self.day,"agent", self.agent_id, "current_min", current_min, "previous_min", previous_min, "returning neutral reward")
        return 0
    
    def _egalitarian_sanction(self, previous_loss, society_well_being):
        current_loss = self._egalitarian_welfare(society_well_being)
        if previous_loss > current_loss:
            #print("day",self.day,"agent", self.agent_id, "current loss", current_loss, "previous loss", previous_loss, "returning pos reward")
            return self.sanction
        elif previous_loss < current_loss:
            #print("day",self.day,"agent", self.agent_id, "current loss", current_loss, "previous loss", previous_loss, "returning neg reward")
            return -self.sanction
        #print("day",self.day,"agent", self.agent_id, "current loss", current_loss, "previous loss", previous_loss, "returning neutral reward")
        return 0
    
    def _utilitarian_sanction(self, previous_welfare, society_well_being):
        current_welfare = self._utilitarian_welfare(society_well_being)
        if current_welfare > previous_welfare:
            #print("day",self.day,"agent", self.agent_id, "current_welfare", current_welfare, "previous_welfare", previous_welfare, "returning pos reward")
            return self.sanction
        elif current_welfare < previous_welfare:
            #print("day",self.day,"agent", self.agent_id, "current_welfare", current_welfare, "previous_welfare", previous_welfare, "returning neg reward")
            return -self.sanction
        #print("day",self.day,"agent", self.agent_id, "current_welfare", current_welfare, "previous_welfare", previous_welfare, "returning neutral reward")
        return 0