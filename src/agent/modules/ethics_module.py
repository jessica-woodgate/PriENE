import numpy as np
from src.harvest_exception import AgentTypeException

class EthicsModule():
    """
    Ethics Module (Algorithm 1) evaluates societal well-being before and after acting and generates a self-directed sanction
    Instance variables:
        sanction -- amount of reward to return to agent
        principle -- normative ethics principle
        society_well_being -- list of well-being for each living agent
        measure_of_well_being -- metric to evaluate well-being before and after acting (minimum experience)
        number_of_minimums -- number of agents which have minimum experience
    """
    def __init__(self,sanction,principle,aggregation):
        self.sanction = sanction
        self.principle = principle
        aggregation_methods = ["average", "majoritarian", "veto", "optimist"]
        if aggregation not in aggregation_methods:
            raise AgentTypeException(aggregation_methods, aggregation)
        self.aggregation = aggregation
        self.can_help = None
        self.society_well_being = None
        self.measure_of_well_being = None
        self.number_of_minimums = None
    
    def update_ethics_state(self, can_help, society_well_being):
        """
        Updates social welfare before agent acts: measure of well-being and number of minimums (Algorithm 1 Line 1)
        """
        self.can_help = can_help
        self._calculate_social_welfare(society_well_being)
    
    def get_sanction(self, society_well_being):
        """
        Obtain sanction from principle comparing current society well-being with previous well-being (Algorithm 1 Lines 3-8)
        """
        if self.principle == "egalitarian":
            return [self._egalitarian_sanction(self.measure_of_well_being, society_well_being)]
        elif self.principle == "maximin":
            return [self._maximin_sanction(self.measure_of_well_being, self.number_of_minimums, society_well_being)]
        elif self.principle == "utilitarian":
            return [self._utilitarian_sanction(self.measure_of_well_being, society_well_being)]
        else:
            return self._combined_sanction(society_well_being)

    def _calculate_social_welfare(self, society_well_being):
        if self.principle == "egalitarian":
            self.measure_of_well_being = self._calculate_egalitarian_welfare(society_well_being)
        elif self.principle == "maximin":
            self.measure_of_well_being, self.number_of_minimums = self._calculate_maximin_welfare(society_well_being)
        elif self.principle == "utilitarian":
            self.measure_of_well_being = self._calculate_utilitarian_welfare(society_well_being)
        else:
            self.measure_of_well_being = self._calculate_all_welfare(society_well_being)
    
    def _calculate_all_welfare(self, society_well_being):
        maximin_min, maximin_num_mins = self._calculate_maximin_welfare(society_well_being)
        return {"egalitarian": self._calculate_egalitarian_welfare(society_well_being),
                "maximin_min": maximin_min,
                "maximin_num_mins": maximin_num_mins,
                "utilitarian": self._calculate_utilitarian_welfare(society_well_being)}

    def _calculate_egalitarian_welfare(self, society_well_being):
        n = len(society_well_being)
        total = sum(society_well_being)
        ideal = total/n
        loss = sum(abs(x - ideal) for x in society_well_being)
        return loss
    
    def _calculate_maximin_welfare(self, society_well_being):
        min_value = min(society_well_being)
        num_mins = np.count_nonzero(society_well_being==min_value)
        return min_value, num_mins
    
    def _calculate_utilitarian_welfare(self, society_well_being):
        return sum(society_well_being)

    def _combined_sanction(self, society_well_being):
        egalitarian = self._egalitarian_sanction(self.measure_of_well_being["egalitarian"], society_well_being)
        maximin = self._maximin_sanction(self.measure_of_well_being["maximin_min"], self.measure_of_well_being["maximin_num_mins"], society_well_being)
        utilitarian = self._utilitarian_sanction(self.measure_of_well_being["utilitarian"], society_well_being)
        if "multiobjective" in self.principle:
            combined_sanction = [egalitarian, maximin, utilitarian]
        else:
            if self.aggregation == "veto":
                combined_sanction = self._veto_aggregation([egalitarian, maximin, utilitarian])
            elif self.aggregation == "optimist":
                combined_sanction = self._optimist_aggregation([egalitarian, maximin, utilitarian])
            elif self.aggregation == "majoritarian":
                combined_sanction = self._majoritarian_aggregation([egalitarian, maximin, utilitarian])
            elif self.aggregation == "average":
                combined_sanction = self._average_aggregation([egalitarian, maximin, utilitarian])
        return combined_sanction
    
    def _average_aggregation(self, sanction_list):
        return np.mean(sanction_list)

    def _majoritarian_aggregation(self, sanction_list):
        return [min(max(np.sum(sanction_list), -self.sanction), self.sanction)]

    def _veto_aggregation(self, sanction_list):
        if -self.sanction in sanction_list:
            combined_sanction = [-self.sanction]
        elif any(num > 0 for num in sanction_list):
            combined_sanction = [self.sanction]
        else:
            combined_sanction = [0]
        return combined_sanction
    
    def _optimist_aggregation(self, sanction_list):
        if any(num > 0 for num in sanction_list):
            combined_sanction = [self.sanction]
        elif -self.sanction in sanction_list:
            combined_sanction = [-self.sanction]
        else:
            combined_sanction = [0]
        return combined_sanction
        
    def _maximin_sanction(self, previous_min, number_of_previous_mins, society_well_being):
        current_min, current_number_of_current_mins = self._calculate_maximin_welfare(society_well_being)
        current_number_of_previous_mins = np.count_nonzero(society_well_being==previous_min)
        #if the global min has been made better, pos reward
        if current_min > previous_min:
            #print("day",self.day,"agent", self.agent_id, "current_min", current_min, "previous_min", previous_min, "returning pos reward")
            return self.sanction
        #if the global min has been made worse, neg reward
        elif current_min < previous_min and self.can_help:
            #print("day",self.day,"agent", self.agent_id, "current_min", current_min, "previous_min", previous_min, "returning neg reward")
            return -self.sanction
        #if the global min has not changed, but there are fewer instances of it, pos reward
        elif current_number_of_previous_mins < number_of_previous_mins and current_min == previous_min:
            #print("day",self.day,"agent", self.agent_id, "current_min", current_min, "previous_min", previous_min, "returning pos less numbers reward")
            return self.sanction
        #if the global min has not changed, and there are more or same number of instances of it, neg reward
        elif current_number_of_previous_mins > number_of_previous_mins and current_min == previous_min and self.can_help:
            #print("day",self.day,"agent", self.agent_id, "current_min", current_min, "previous_min", previous_min, "returning neg more numbers reward")
            return -self.sanction
        #print("day",self.day,"agent", self.agent_id, "current_min", current_min, "previous_min", previous_min, "returning neutral reward")
        return 0
    
    def _egalitarian_sanction(self, previous_loss, society_well_being):
        current_loss = self._calculate_egalitarian_welfare(society_well_being)
        if previous_loss > current_loss:
            #print("day",self.day,"agent", self.agent_id, "current loss", current_loss, "previous loss", previous_loss, "returning pos reward")
            return self.sanction
        elif previous_loss < current_loss and self.can_help:
            #print("day",self.day,"agent", self.agent_id, "current loss", current_loss, "previous loss", previous_loss, "returning neg reward")
            return -self.sanction
        #print("day",self.day,"agent", self.agent_id, "current loss", current_loss, "previous loss", previous_loss, "returning neutral reward")
        return 0
    
    def _utilitarian_sanction(self, previous_welfare, society_well_being):
        current_welfare = self._calculate_utilitarian_welfare(society_well_being)
        if current_welfare > previous_welfare:
            #print("day",self.day,"agent", self.agent_id, "current_welfare", current_welfare, "previous_welfare", previous_welfare, "returning pos reward")
            return self.sanction
        elif current_welfare < previous_welfare and self.can_help:
            #print("day",self.day,"agent", self.agent_id, "current_welfare", current_welfare, "previous_welfare", previous_welfare, "returning neg reward")
            return -self.sanction
        #print("day",self.day,"agent", self.agent_id, "current_welfare", current_welfare, "previous_welfare", previous_welfare, "returning neutral reward")
        return 0