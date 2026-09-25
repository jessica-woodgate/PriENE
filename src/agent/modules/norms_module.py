class NormsModule():
    """
    Norms Module (Algorithm 2) handles tracking of behaviours and norms
    NormsModule has no built-in knowledge of what the observed features or actions mean: the
    owning agent registers that at construction time (antecedent_features, consequent_rules), so
    this module can be reused by any agent/scenario wanting norm tracking over a different set of
    state features or actions without needing any changes here.
    Instance variables:
        agent_id -- identification of agent
        antecedent_features -- ordered list of feature specs used by get_antecedent to bucket
            state values into a natural language precondition string. Each spec is a dict:
                boundaries -- ascending thresholds; a value is bucketed into labels[i] for the
                    first i where value < boundaries[i], else labels[-1]
                labels -- one more label than boundaries, e.g. ["low health","medium health","high health"]
                zero_label -- optional, checked before boundaries: label used when value == 0
                repeated -- if True, the call-time value for this spec is a list, and every
                    element is bucketed and appended individually (e.g. one well-being reading
                    per other observed agent, rather than a single scalar reading)
        consequent_rules -- ordered list of rules used by get_consequent to turn a raw action
            string into a natural language postcondition. Each rule is a dict:
                prefix -- if the action string starts with this, substitute `label` for it
                label -- the natural language label to substitute
            an action matching no rule is used as-is (e.g. "move", "eat")
        max_norms -- max size of norms and behaviour bases
        norm_clipping_frequency -- time interval to clip norms and behaviour bases
        norm_decay_rate -- decay of norm over time
    """
    def __init__(self, agent_id, antecedent_features, consequent_rules):
        self.agent_id = agent_id
        self.antecedent_features = antecedent_features
        self.consequent_rules = consequent_rules
        self.max_norms = 100
        self.norm_clipping_frequency = 10
        self.behaviour_base = {}
        self.norm_decay_rate = 0.3

    def get_antecedent(self, feature_values):
        """
        Get antecedent string by bucketing an ordered list of state values, one per registered
        antecedent_features spec (a "repeated" spec's corresponding value is itself a list)
        """
        view = ["IF"]
        for spec, value in zip(self.antecedent_features, feature_values):
            if spec.get("repeated"):
                for v in value:
                    view.append(self._bucket(v, spec))
            else:
                view.append(self._bucket(value, spec))
        return ",".join(view)

    def get_consequent(self, action):
        """
        Get consequent string from action, generalised via registered consequent_rules
        """
        for rule in self.consequent_rules:
            if action.startswith(rule["prefix"]):
                return "THEN," + rule["label"]
        return "THEN," + action

    def _bucket(self, value, spec):
        """
        Buckets a single numeric value into its natural language label per a feature spec
        """
        zero_label = spec.get("zero_label")
        if zero_label is not None and value == 0:
            return zero_label
        for boundary, label in zip(spec["boundaries"], spec["labels"]):
            if value < boundary:
                return label
        return spec["labels"][-1]
    
    def update_behaviour_base(self, antecedent, action, reward, day, episode):
        """
        Update current behaviour and then update the age of all behaviours in behaviour base
        If day == clipping frequency, clip behaviour base if it exceeds maximum capacity
        """
        self._update_behaviour(antecedent,action,reward)
        self._update_behaviours_age()
        if day % self.norm_clipping_frequency == 0:
            self._clip_behaviour_base(day, episode)

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

    def _clip_behaviour_base(self, day, episode):
        if len(self.behaviour_base.keys()) > self.max_norms:
            print(f"agent {self.agent_id} clipping behaviour base, length is {len(self.behaviour_base.keys())} in episode {episode} day {day}")
            for metadata in self.behaviour_base.values():
                self._update_norm_fitness(metadata)
            assessed_base = self._assess(self.behaviour_base)
            self.behaviour_base = dict(assessed_base[:self.max_norms])

    def _assess(self, pop):
        return sorted(pop.items(), key=lambda item: item[1]["fitness"], reverse=True)