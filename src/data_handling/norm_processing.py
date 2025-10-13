import json
import pandas as pd

class NormProcessing():
    def __init__(self):
        self.min_instances = 1
        self.min_fitness = 0.1
        self.min_reward = 50
    
    def process_norms(self, df_labels, scenario, norms_filepath, write_filepath):
        cooperative_dfs = []
        cooperative_tendencies = []
        emerged_norms_proportions = {"society": [],
                                     "num_emerged_norms": [],
                                     "num_cooperative_norms": [],
                                     "proportion_cooperative": []}
        for label in df_labels:
            input_file = norms_filepath+label+"_emerged_norms.json"
            output_file = write_filepath+scenario+"_"+label+"_norms"
            df, n_norms, n_cooperative_norms = self._process_society_norms(input_file, output_file, label)
            cooperative_dfs.append(df)
            cooperative_tendencies.append(self._calculate_norms_tendency(df, label))
            emerged_norms_proportions["society"].append(label)
            emerged_norms_proportions["num_emerged_norms"].append(n_norms)
            emerged_norms_proportions["num_cooperative_norms"].append(n_cooperative_norms)
            emerged_norms_proportions["proportion_cooperative"].append((n_cooperative_norms/n_norms)*100 if n_norms > 0 else 0)
        return cooperative_dfs, cooperative_tendencies, emerged_norms_proportions

    def _process_society_norms(self, input_file, output_file, df_label):
        f = open(input_file)
        data = json.load(f)
        cooperative_data, n_norms, n_cooperative_norms = self._count_cooperative_norms(data, output_file)
        data = self._merge_norms(data, output_file)
        self._generalise_norms(data.keys(), output_file)
        #self._generate_norms_tree(data, output_file)
        return cooperative_data, n_norms, n_cooperative_norms
    
    def _count_cooperative_norms(self, data, output_file):
        cooperative_norms = []
        n_norms = 0
        for episode_number, episode_norms in data.items():
            for norm in episode_norms:
                n_norms += 1
                norm_name = list(norm.keys())[0]
                norm_value = list(norm.values())[0]
                consequent = norm_name.split("THEN")[1].strip(",")
                if consequent == "throw":
                    norm_data = {"reward": norm_value["reward"], "numerosity": norm_value["numerosity"], "fitness": norm_value["fitness"]}
                    cooperative_norms.append(norm_data)
        # print("Total emerged norms:", n_norms, "Total cooperative norms:", len(cooperative_norms))
        # if len(cooperative_norms) != 0 and n_norms != 0:
        #     print("Proportion of cooperative norms for "+output_file+" is "+str((len(cooperative_norms)/n_norms)*100))
        # else:
        #     print(f"Proportion of cooperative norms for {output_file} is 0. Number of norms is {n_norms}")
        n_cooperative_norms = len(cooperative_norms)
        df = pd.DataFrame(cooperative_norms)
        df.to_csv(output_file+"_cooperative_data.csv",index=False)
        return df, n_norms, n_cooperative_norms
    
    def _generalise_norms(self, norms, output_file):
        rule_dict = {}
        for rule in norms:
            conditions, action = rule.split("THEN")
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
        filename = output_file+"_merged.txt"
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
    
    def _calculate_norms_tendency(self, df, df_label):
        if "reward" in df.columns:
            central_tendency = {"df_label": df_label,
                            "reward_mean": df["reward"].mean(),
                            "reward_median": df["reward"].median(),
                            "reward_stdev": df["reward"].std(),
                            "numerosity_mean": df["numerosity"].mean(),
                            "numerosity_median": df["numerosity"].median(),
                            "numerosity_stdev": df["numerosity"].std(),
                            "fitness_mean": df["fitness"].mean(),
                            "fitness_median": df["fitness"].median(),
                            "fitness_stdev": df["fitness"].std(),
                            }
        else:
            #no cooperative norms found
            central_tendency = {"df_label": df_label,
                            "reward_mean": 0,
                            "reward_median": 0,
                            "reward_stdev": 0,
                            "numerosity_mean": 0,
                            "numerosity_median": 0,
                            "numerosity_stdev": 0,
                            "fitness_mean": 0,
                            "fitness_median": 0,
                            "fitness_stdev": 0,
                            }
        return central_tendency