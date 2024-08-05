class NormsModule():
    def __init__(self,agent_id):
        self.agent_id = agent_id
        self.max_norms = 500
        self.norm_base = {}
        self.low_health_threshold = 0.6
        self.high_health_threshold = 2.0
        self.low_berries_threshold = 1
        self.high_berries_threshold = 3
        self.low_days_left_threshold = 10
        self.high_days_left_threshold = 30
        self.norm_decay_rate = 0.3
        self.numerosity_weight = 1.0
        self.score_weight = 1.0

    def clip_norm_base(self):
        if len(self.norm_base.keys()) > self.max_norms:
            for norm in self.norm_base:
                self.update_norm_fitness(norm)
            norms = self.assess(self.norm_base)
            self.norm_base = norms[:self.max_norms]

    def assess(self, pop):
        return sorted(pop, key = lambda i: i["fitness"], reverse=True)

    def get_antecedent(self, health, berries, well_being):
        if health < self.low_health_threshold:
            h = "low health"
        elif health >= self.low_health_threshold and health < self.high_health_threshold:
            h = "medium health"
        else:
            h = "high health"
        if berries == 0:
            b = "no berries"
        elif berries > 0 and berries < self.low_berries_threshold:
            b = "low berries"
        elif berries >= self.low_berries_threshold and berries < self.high_berries_threshold:
            b = "medium berries"
        else:
            b = "high berries"
        view = ["IF", h, b]
        #agent_days = [a.days_left_to_live for a in well_being if a.unique_id != self.agent_id and a.agent_type != "berry"]
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
        consequent = "THEN,"
        if action == "north" or action == "east" or action == "south" or action == "west":
            return consequent + "move"
        elif "throw" in action:
            return consequent + "throw"
        else:
            return consequent + action

    def update_norm(self, antecedent, action, reward):
        consequent = self.get_consequent(action)
        current_norm = ",".join([antecedent,consequent])
        norm = self.norm_base.get(current_norm)
        if norm != None:
            norm["score"] += reward
            norm["numerosity"] += 1
            self._update_norm_fitness(norm)
        else:
            self.norm_base[current_norm] = {"score": reward,
                                    "numerosity": 1,
                                    "age": 0,
                                    "fitness": 0}
    def update_norm_age(self):
        for value in self.norm_base.values():
            if "age" in value:
                value["age"] += 1
    
    def _update_norm_fitness(self, norm):
        if norm["age"] != 0:
            base_fitness = self.norm_decay_rate ** norm["age"]
            discounted_numerosity = norm["numerosity"] ** self.numerosity_weight
            discounted_score = norm["score"] ** self.score_weight
            norm["fitness"] = base_fitness * discounted_numerosity * discounted_score