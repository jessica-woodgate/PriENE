class NormsModule():
    """
    Norms Module (Algorithm 2) handles tracking of behaviours and norms
    Instance variables:
        agent_id -- identification of agent
        max_norms -- max size of norms and behaviour bases
        norm_clipping_frequency -- time interval to clip norms and behaviour bases
        low_health_threshold -- antecedent threshold for "low health"
        high_health_threshold -- antecedent threshold for "high health"
        low_berries_threshold -- antecedent threshold for "low berries"
        high_berries_threshold -- antecedent threshold for "high berries"
        low_days_left_threshold -- antecedent threshold for "low days"
        high_days_left_threshold -- antecedent threshold for "high days"
        norm_decay_rate -- decay of norm over time
    """
    def __init__(self,agent_id):
        self.agent_id = agent_id
        self.max_norms = 100
        self.norm_clipping_frequency = 10
        self.behaviour_base = {}
        self.low_health_threshold = 0.6
        self.high_health_threshold = 2.0
        self.low_berries_threshold = 1
        self.high_berries_threshold = 3
        self.low_days_left_threshold = 10
        self.high_days_left_threshold = 30
        self.norm_decay_rate = 0.3

    def get_antecedent(self, berries, health, well_being):
        """
        Get antecedent string from view of agent's berries and health and society well-being
        """
        if berries == 0:
            b = "no berries"
        elif berries > 0 and berries < self.low_berries_threshold:
            b = "low berries"
        elif berries >= self.low_berries_threshold and berries < self.high_berries_threshold:
            b = "medium berries"
        else:
            b = "high berries"
        if health < self.low_health_threshold:
            h = "low health"
        elif health >= self.low_health_threshold and health < self.high_health_threshold:
            h = "medium health"
        else:
            h = "high health"
        view = ["IF", b, h]
        for w in well_being:
            if w < self.low_days_left_threshold:
                view.append("low days")
            elif w >= self.low_days_left_threshold and w < self.high_days_left_threshold:
                view.append("medium days")
            else:
                view.append("high days")
        antecedent = ",".join(view)
        return antecedent

    def get_consequent(self, action):
        """
        Get consequent string from action
        """
        consequent = "THEN,"
        if action == "north" or action == "east" or action == "south" or action == "west":
            return consequent + "move"
        elif "throw" in action:
            return consequent + "throw"
        else:
            return consequent + action
    
    def update_behaviour_base(self, antecedent, action, reward, day):
        """
        Update current behaviour and then update the age of all behaviours in behaviour base
        If day == clipping frequency, clip behaviour base if it exceeds maximum capacity
        """
        self._update_behaviour(antecedent,action,reward)
        self._update_behaviours_age()
        if day % self.norm_clipping_frequency == 0:
            self._clip_behaviour_base()

    def _update_behaviour(self, antecedent, action, reward):
        consequent = self.get_consequent(action)
        current_norm = ",".join([antecedent,consequent])
        norm = self.behaviour_base.get(current_norm)
        if norm != None:
            norm["reward"] += reward
            norm["numerosity"] += 1
            self._update_norm_fitness(norm)
        else:
            self.behaviour_base[current_norm] = {"reward": reward,
                                    "numerosity": 1,
                                    "age": 0,
                                    "fitness": 0}
            
    def _update_behaviours_age(self):
        for value in self.behaviour_base.values():
            if "age" in value:
                value["age"] += 1
    
    def _update_norm_fitness(self, norm):
        if norm["age"] != 0:
            discounted_age = self.norm_decay_rate * norm["age"]
            fitness = norm["numerosity"] * norm["reward"] * discounted_age
            norm["fitness"] = round(fitness, 4)

    def _clip_behaviour_base(self):
        if len(self.behaviour_base.keys()) > self.max_norms:
            for metadata in self.behaviour_base.values():
                self._update_norm_fitness(metadata)
            assessed_base = self._assess(self.behaviour_base)
            self.behaviour_base = dict(assessed_base[:self.max_norms])

    def _assess(self, pop):
        return sorted(pop.items(), key=lambda item: item[1]["fitness"], reverse=True)