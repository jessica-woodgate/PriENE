import json
import pandas as pd
import networkx as nx

class NormProcessing():
    def __init__(self):
        self.min_instances = 1
        self.min_fitness = 0.1
        self.min_reward = 50
    
    def process_norms(self, df_labels, scenario, norms_filepath, write_filepath):
        episode_norm_dfs = []
        cooperative_dfs = []
        norms_tendencies = []
        for label in df_labels:
            input_file = norms_filepath+label+"_emerged_norms.json"
            output_file = write_filepath+scenario+"_"+label+"_norms"
            episode_norm_data, cooperative_norms = self._process_society_norms(input_file, output_file, filter_fitness=False)
            episode_norm_dfs.append(episode_norm_data)
            cooperative_dfs.append(cooperative_norms)
            norms_tendencies.append(self._calculate_norms_central_tendency(episode_norm_data, label))
        return episode_norm_dfs, cooperative_dfs, norms_tendencies

    def _process_society_norms(self, input_file, output_file, filter_fitness=False):
        f = open(input_file)
        norm_data = json.load(f)
        if filter_fitness:
            filtered_data = {}
            for society_id, rules in norm_data.items():
                filtered_rules = []
                for rule_entry in rules:
                    # rule_entry is a dict with one key (the rule string)
                    rule_str, rule_info = next(iter(rule_entry.items()))
                    if rule_info.get("fitness", 0.0) > 0.0:
                        filtered_rules.append(rule_entry)
                if filtered_rules:
                    filtered_data[society_id] = filtered_rules
            norm_data = filtered_data
        #counts total norms, cooperative norms, and proportion of cooperative norms for each episode
        episode_norm_data = self._count_norms(norm_data, output_file)
        #merges norms repeated across episodes and writes unique norms to file
        merged_norms = self._merge_norms(norm_data, output_file)
        #looks at all the merged norms and collects data for the cooperative norms
        cooperative_norms = self._get_cooperative_norms(merged_norms, output_file)
        #self._generalise_norms(merged_norms.keys(), output_file)
        #self._generalise_norms(merged_norms.keys(), output_file, cooperative_norms=True)
        return episode_norm_data, cooperative_norms
    
    def _count_norms(self, data, output_file):
        df = {"episode": [],
                        "total_norms": [],
                        "cooperative_norms": [],
                        "proportion": [],
                        "reward": [],
                        "numerosity": [],
                        "fitness": []}
        for episode, rules in data.items():
            n_episode = str(episode)
            total_norms = len(rules)
            cooperative_norms = 0
            reward = 0
            numerosity = 0
            fitness = 0
            if total_norms != 0:
                cooperative_norms = 0
                for rule in rules:
                    for norm_name, norm_value in rule.items():
                        reward += norm_value["reward"]
                        numerosity += norm_value["numerosity"]
                        fitness += norm_value["fitness"]
                        if "throw" in norm_name:
                            cooperative_norms += 1
            df["episode"].append(n_episode)
            df["total_norms"].append(total_norms)
            df["cooperative_norms"].append(cooperative_norms)
            df["proportion"].append((cooperative_norms/total_norms)*100 if cooperative_norms > 0 and total_norms > 0 else 0)
            df["reward"].append(reward)
            df["numerosity"].append(numerosity)
            df["fitness"].append(fitness)
        df = pd.DataFrame(df)
        df.to_csv(output_file+"_emerged_norms_data.csv", index=False)
        return df
    
    def _get_cooperative_norms(self, data, output_file):
        cooperative_norms = []
        for norm_name, norm_value in data.items():
            if "throw" in norm_name:
                norm_data = {"norm": norm_name, "reward": norm_value["reward"], "numerosity": norm_value["numerosity"], "fitness": norm_value["fitness"], "num_instances_across_episodes": norm_value["num_instances_across_episodes"]}
                cooperative_norms.append(norm_data)
        df = pd.DataFrame(cooperative_norms)
        df.to_csv(output_file+"_cooperative_data.csv",index=False)
        return df

    def _merge_norms(self,data,output_file):
        """
        Merges duplicates of norms into one dictionary
        Args:
            data: Norm base to remove duplicates from (dictionary).
            filename: The file to write the unique set of norms to.
            filter: Whether to filter the norms by fitness and number of instances.
            min_instances: Minimum number of instances of a norm to include in unique set.
            min_fitness: Minimum fitness of norm to include in unique set.

        Returns:
            A dictionary containing the unique set of norms.
        """
        emerged_norms = {}
        for episode_number, episode_norms in data.items():
            #key is the episode number; value is the emerged norms from that episode
            for norm in episode_norms:
                for norm_name, norm_data in norm.items():
                    if ("throw" in norm_name and "no berries" in norm_name) or ("eat" in norm_name and "no berries" in norm_name):
                        continue
                    if norm_name not in emerged_norms.keys():
                        emerged_norms[norm_name] = {"reward": norm_data["reward"],
                                                    "numerosity": norm_data["numerosity"],
                                                    "fitness": norm_data["fitness"],
                                                    "adoption": norm_data["adoption"],
                                                    "num_instances_across_episodes": 1}
                    else:
                        emerged_norms[norm_name]["reward"] += norm_data["reward"]
                        emerged_norms[norm_name]["numerosity"] += norm_data["numerosity"]
                        emerged_norms[norm_name]["fitness"] += norm_data["fitness"]
                        emerged_norms[norm_name]["adoption"] += norm_data["adoption"]
                        emerged_norms[norm_name]["num_instances_across_episodes"] += 1
        emerged_norms = dict(sorted(emerged_norms.items(), key=lambda item: item[1]["fitness"], reverse=True))

        filename = output_file+"_merged.json"
        with open(filename, "a+") as file:
            file.seek(0)
            if not file.read(1):
                file.write("\n")
            file.seek(0, 2)
            json.dump(emerged_norms, file, indent=4)
            file.write("\n")
        with open(output_file+"_merged_keys.txt", "w") as keys_file:
            keys_file.write("\n".join([key for key in emerged_norms.keys()]))
        with open(output_file+"_merged_cooperative_keys.txt", "w") as keys_file:
            for key in emerged_norms.keys():
                if "throw" in key:
                    keys_file.write(key+"\n")
        return emerged_norms
    
    def _generalise_norms(self, norms, output_file, cooperative_norms=False):
        """
        Generalises a set of norms (rules) by merging redundant or overly specific rules 
        into more general ones, and outputs the resulting simplified rule set.
        The function identifies cases where two rules have the same action, but one rule 
        has a subset of the other's conditions (i.e. it is more general). The more 
        specific rule is removed, keeping only the general rule.
        Steps:
            1. Parse each norm into a mapping from condition tuples to actions.
            2. Identify rules that can be generalised by dropping redundant conditions 
            when a simpler version exists with the same action.
            3. Merge rules accordingly and remove duplicates.
            4. Convert the merged rule set back into a list of textual rules.
            5. Generate and save a norms tree representation to the specified output file.

        Args:
            norms (list[str]): 
                A list of rule strings, where each rule follows the format 
                "IF condition1,condition2,... THEN action".
            output_file (str): 
                Path to the output file where the generalised norms or tree representation 
                will be written.
        """
        rule_dict = {}
        for rule in norms:
            conditions, action = rule.split("THEN")
            if cooperative_norms:
                action = action.strip().lower()
                if "throw" not in action:
                    continue
            conditions = conditions.split(",")[1:]
            rule_dict[tuple(conditions)] = action.strip()
        def merge_rules(rule_dict):
            merged_rules = {}
            for conditions, action in rule_dict.items():
                if conditions in merged_rules:
                    continue
                generalised_conditions = []
                for i in range(len(conditions)):
                    shorter_conditions = conditions[:i] + conditions[i+1:]
                    if tuple(shorter_conditions) in rule_dict and rule_dict[tuple(shorter_conditions)] == action:
                        generalised_conditions = shorter_conditions
                        break
                if generalised_conditions:
                    merged_rules[tuple(generalised_conditions)] = action
                else:
                    merged_rules[conditions] = action
            return merged_rules
        merged_rules = merge_rules(rule_dict)
        merged_rules = self._convert_to_rule_list(merged_rules)
        if cooperative_norms:
            output_file += "_cooperative"
        self._generate_norms_tree(merged_rules, output_file)
    
    def _convert_to_rule_list(self, data):
        rule_list = []
        for conditions, action in data.items():
            rule = ["IF"]
            rule.extend(conditions[:-1])
            rule.append("THEN")
            rule.append(action.strip(","))
            rule_string = ", ".join(rule)
            rule_list.append(rule_string)
        return rule_list
        
    def _generate_norms_tree(self, data, output_file):
        tree = {}
        for norm in data:
            conditions = norm.split("IF")[1].split(",")[:-2]
            conditions = conditions[1:]
            current_node = tree
            for condition in conditions:
                if condition not in current_node:
                    current_node[condition] = {}
                if isinstance(current_node[condition], list):
                    new_node = {}
                    for item in current_node[condition]:
                        new_node[item] = {}
                    current_node[condition] = new_node
                current_node = current_node[condition]
            consequent = norm.split("THEN")[1].strip(",")
            if isinstance(current_node, list):
                current_node.append(consequent)
            else:
                current_node[condition] = [consequent]
        with open(output_file+"_tree.txt", 'w') as f:
            f.write(self._print_tree(tree, indent="  "))

    def _print_tree(self, node, indent=""):
        output = ""
        for key, value in node.items():
            if isinstance(value, dict):
                output += f"{indent}{key}\n"
                output += self._print_tree(value, indent + "  ")
            else:
                output += f"{indent}{key}: {value}\n"
        return output
    
    def _calculate_norms_central_tendency(self, df, df_label):
        central_tendency = {"df_label": df_label,
                            "total_norms_mean": df["total_norms"].mean(),
                            "total_norms_median": df["total_norms"].median(),
                            "total_norms_stdev": df["total_norms"].std(),
                            "total_norms_summary": (df["total_norms"].mean() + df["total_norms"].median()) - df["total_norms"].std(),
                            "cooperative_norms_mean": df["cooperative_norms"].mean(),
                            "cooperative_norms_median": df["cooperative_norms"].median(),
                            "cooperative_norms_stdev": df["cooperative_norms"].std(),
                            "cooperative_norms_summary": (df["cooperative_norms"].mean() + df["cooperative_norms"].median()) - df["cooperative_norms"].std(),
                            "proportion_mean": df["proportion"].mean(),
                            "proportion_median": df["proportion"].median(),
                            "proportion_stdev": df["proportion"].std(),
                            "proportion_summary": (df["proportion"].mean() + df["proportion"].median()) - df["proportion"].std(),
                            "reward_mean": df["reward"].mean(),
                            "reward_median": df["reward"].median(),
                            "reward_stdev": df["reward"].std(),
                            "reward_summary": (df["reward"].mean() + df["reward"].median()) - df["reward"].std(),
                            "numerosity_mean": df["numerosity"].mean(),
                            "numerosity_median": df["numerosity"].median(),
                            "numerosity_stdev": df["numerosity"].std(),
                            "numerosity_summary": (df["numerosity"].mean() + df["numerosity"].median()) - df["numerosity"].std(),
                            "fitness_mean": df["fitness"].mean(),
                            "fitness_median": df["fitness"].median(),
                            "fitness_stdev": df["fitness"].std(),
                            "fitness_summary": (df["fitness"].mean() + df["fitness"].median()) - df["fitness"].std()
                            }
        return central_tendency
    
    # def _calculate_norms_tendency(self, df, df_label):
    #     if "reward" in df.columns:
    #         central_tendency = {"df_label": df_label,
    #                         "reward_mean": df["reward"].mean(),
    #                         "reward_median": df["reward"].median(),
    #                         "reward_stdev": df["reward"].std(),
    #                         "numerosity_mean": df["numerosity"].mean(),
    #                         "numerosity_median": df["numerosity"].median(),
    #                         "numerosity_stdev": df["numerosity"].std(),
    #                         "fitness_mean": df["fitness"].mean(),
    #                         "fitness_median": df["fitness"].median(),
    #                         "fitness_stdev": df["fitness"].std(),
    #                         }
    #     else:
    #         #no cooperative norms found
    #         central_tendency = {"df_label": df_label,
    #                         "reward_mean": 0,
    #                         "reward_median": 0,
    #                         "reward_stdev": 0,
    #                         "numerosity_mean": 0,
    #                         "numerosity_median": 0,
    #                         "numerosity_stdev": 0,
    #                         "fitness_mean": 0,
    #                         "fitness_median": 0,
    #                         "fitness_stdev": 0,
    #                         }
    #     return central_tendency