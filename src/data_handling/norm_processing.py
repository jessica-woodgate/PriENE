import json
import pandas as pd

class NormProcessing():
    def __init__(self):
        pass
    
    def proccess_and_display_tree(self, input_file, output_file, filter_norms, min_instances=None, fitness_threshold=None):
        f = open(input_file) 
        data = json.load(f)
        data = self._merge_norms(data, output_file, filter_norms, min_instances, fitness_threshold)
        self._process_and_write_norms(data, output_file)

    def _process_and_write_norms(self, data, output_file):
        """
        Processes a list of dictionaries representing norms and prints the tree structure to a file.

        Args:
            data: A list of dictionaries where each dictionary represents a norm (IF,x,y,z,THEN,a) with its information.
            fitness_threshold: The minimum fitness score for a norm to be included.
            output_file: The path to the file where the tree structure will be printed.
        """
        tree = {}
        for norm in data:
            conditions = norm[0].split("IF")[1].split(",")[:-2]
            current_node = tree
            for condition in conditions:
                if condition not in current_node:
                    current_node[condition] = {}
                current_node = current_node[condition]
            current_node[condition] = norm[0].split("THEN")[1].strip(",")
        tree = tree['']
        with open(output_file+".txt", 'w') as f:
            f.write(self._print_tree(tree, indent="  "))

    def _print_tree(self, node, indent=""):
        """
        Recursively prints the tree structure to a string with indentation.

        Args:
            node: A node of the tree structure (dictionary).
            indent: The indentation string for current level.

        Returns:
            A string representation of the tree with indentation.
        """
        output = ""
        for key, value in node.items():
            if isinstance(value, dict):
                output += f"{indent}{key}\n"
                output += self._print_tree(value, indent + "  ")
            else:
                output += f"{indent}{key}: {value}\n"
        return output

    def _merge_norms(self,data,output_file,filter=False,min_instances=None,min_fitness=None):
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
        for key, value in data.items():
            for dict in value:
                for norm_name, norm_value in dict.items():
                    if norm_name not in emerged_norms:
                        emerged_norms[norm_name] = {"reward": 0,
                                                    "numerosity": 0,
                                                    "fitness": 0,
                                                    "num_instances": 0,
                                                    "num_instances_across_episodes": 0}
            emerged_norms[norm_name]["reward"] += norm_value["reward"]
            emerged_norms[norm_name]["numerosity"] += norm_value["numerosity"]
            emerged_norms[norm_name]["fitness"] += norm_value["fitness"]
            emerged_norms[norm_name]["num_instances"] += norm_value["num_instances"]
            emerged_norms[norm_name]["num_instances_across_episodes"] += 1
        if filter:
            emerged_norms = {key: value for key, value in emerged_norms.items() if value["num_instances"] > min_instances and value["fitness"] > min_fitness}
        emerged_norms = sorted(emerged_norms.items(), key=lambda item: item[1]["fitness"], reverse=True)
        with open(filename, "a+") as file:
                    file.seek(0)
                    if not file.read(1):
                        file.write("\n")
                    file.seek(0, 2)
                    json.dump(emerged_norms, file, indent=4)
                    file.write("\n")
        with open(output_file+"_merged_keys.txt", "w") as keys_file:
            keys_file.write("\n".join([key for key, value in emerged_norms]))
        return emerged_norms