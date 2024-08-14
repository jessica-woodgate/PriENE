import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from src.data_handling.norm_processing import NormProcessing

class DataAnalysis():
    def __init__(self, num_agents, filepath):
        self.num_agents = num_agents
        self.filepath = filepath
    
    def proccess_and_display_all_data(self, scenario, agent_df_list, df_labels, norm_df_list=None):
        normalised_sum_df_list, agent_end_episode_list = self._process_agent_dfs(agent_df_list, df_labels)
        self._display_graphs(normalised_sum_df_list, agent_end_episode_list, df_labels)
        self._process_norms(scenario, norm_df_list, df_labels)

    def _process_agent_dfs(self, agent_df_list, df_labels):
        normalised_sum_df_list = self._apply_function_to_list(agent_df_list, self._normalised_sum_step_across_episodes)
        self._write_df_list_to_file(normalised_sum_df_list, df_labels, self.filepath+"normalised_sum_df_")
        agent_end_episode_list = self._process_end_episode_dataframes(agent_df_list)
        self._write_df_list_to_file(agent_end_episode_list, df_labels, self.filepath+"processed_episode_df_")
        return normalised_sum_df_list, agent_end_episode_list
    
    def _display_graphs(self, normalised_sum_df_list, agent_end_episode_list, df_labels):
        self._days_left_to_live_results(normalised_sum_df_list, df_labels, self.filepath+"days_left_to_live")
        self._berries_consumed_results(normalised_sum_df_list, df_labels, self.filepath+"berries_consumed")
        self._display_violin_plot_df_list(agent_end_episode_list, df_labels, "day", self.filepath+"violin_end_day", "Violin Plot of Episode Length", "End Day")
        self._display_violin_plot_df_list(agent_end_episode_list, df_labels, "total_berries", self.filepath+"violin_total_berries", "Violin Plot of Total Berries Consumed", "Berries Consumed")

    def _process_norms(self,scenario, norm_df_list, df_labels):
        scenario_file = self.filepath+scenario
        norm_processing = NormProcessing()
        cooperative_dfs = []
        for label in df_labels:
            input_file = scenario_file+"_"+label+"_emerged_norms.json"
            output_file = scenario_file+"_"+label+"_processed_norms"
            cooperative_dfs.append(norm_processing.proccess_norms(input_file, output_file))
        self._display_swarm_plot(cooperative_dfs,df_labels, "numerosity", scenario_file+"_cooperative_numerosity")

    def _write_df_list_to_file(self, df_list, df_labels, file_string):
        i = 0
        for df in df_list:
            df.to_csv(file_string+df_labels[i]+".csv")
            i += 1

    def _normalised_sum_step_across_episodes(self, df):
        df = df.drop(["episode", "action"], axis=1)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        #Calculate counts for each (step, agent_id) combination
        count_df = df.groupby(["day", "agent_id"]).size().reset_index(name="count")
        #Sum and normalize by count
        sum_df = df.groupby(["day", "agent_id"]).sum().reset_index()
        sum_df = sum_df.reset_index(drop=True)
        count_df = count_df.reset_index(drop=True)
        to_divide_columns = list(sum_df.columns)
        to_divide_columns.remove("day")
        to_divide_columns.remove("agent_id")
        sum_df.loc[:, to_divide_columns] = sum_df.loc[:, to_divide_columns].divide(count_df["count"], axis=0)
        #sum_df = sum_df.divide(count_df["count"], axis=0)
        sum_df["count"] = count_df["count"]
        return sum_df

    def _process_end_episode_dataframes(self, dataframes):
        processed_dfs = []
        for i, df in enumerate(dataframes):
            grouped_df = df.groupby(["episode", "agent_id"])
            episode_dfs = []
            for (episode, agent_id), group_df in grouped_df:
                last_row = group_df.tail(1).loc[:, ~group_df.columns.str.contains('^Unnamed')]
                last_row["total_berries"] = last_row["berries"] + last_row["berries_consumed"]
                episode_dfs.append(last_row)
            processed_df = pd.concat(episode_dfs)
            processed_dfs.append(processed_df)
        return processed_dfs

    def _days_left_to_live_results(self, sum_df_list, df_labels, filename):
        max_days_left_to_live = self._calculate_column_across_episode(sum_df_list, df_labels, "days_left_to_live", self._calculate_max)
        self._display_dataframe(max_days_left_to_live, "Max Days Left To Live", "Days Left To Live", filename+"_max")
        min_days_left_to_live = self._calculate_column_across_episode(sum_df_list, df_labels, "days_left_to_live", self._calculate_min)
        self._display_dataframe(min_days_left_to_live, "Min Days Left To Live", "Days Left To Live", filename+"_min")
        total_days_left_to_live = self._calculate_column_across_episode(sum_df_list, df_labels, "days_left_to_live", self._calculate_total)
        self._display_dataframe(total_days_left_to_live, "Total Days Left To Live", "Days Left To Live", filename+"_total")
        gini_days_left_to_live = self._calculate_column_across_episode(sum_df_list, df_labels, "days_left_to_live", self._calculate_gini)
        self._display_dataframe(gini_days_left_to_live, "Gini Index of Days Left To Live", "Days Left To Live", filename+"_gini")
        max_days_left_to_live.to_csv(self.filepath+"max_days_left_to_live.csv")
        min_days_left_to_live.to_csv(self.filepath+"min_days_left_to_live.csv")
        gini_days_left_to_live.to_csv(self.filepath+"gini_days_left_to_live.csv")
        total_days_left_to_live.to_csv(self.filepath+"total_days_left_to_live.csv")
    
    def _berries_consumed_results(self, sum_df_list, df_labels, filename):
        max_berries_consumed = self._calculate_column_across_episode(sum_df_list, df_labels, "berries_consumed", self._calculate_max)
        self._display_dataframe(max_berries_consumed, "Max Berries Consumed", "Berries Consumed", filename+"_max")
        min_berries_consumed = self._calculate_column_across_episode(sum_df_list, df_labels, "berries_consumed", self._calculate_min)
        self._display_dataframe(min_berries_consumed, "Min Berries Consumed", "Berries Consumed", filename+"_min")
        total_berries_consumed = self._calculate_column_across_episode(sum_df_list, df_labels, "berries_consumed", self._calculate_total)
        self._display_dataframe(total_berries_consumed, "Total Berries Consumed", "Berries Consumed", filename+"_total")
        gini_berries_consumed = self._calculate_column_across_episode(sum_df_list, df_labels, "berries_consumed", self._calculate_gini)
        self._display_dataframe(gini_berries_consumed, "Gini Index of Berries Consumed", "Berries Consumed", filename+"_gini")
        max_berries_consumed.to_csv(self.filepath+"max_berries_consumed.csv")
        min_berries_consumed.to_csv(self.filepath+"min_berries_consumed.csv")
        gini_berries_consumed.to_csv(self.filepath+"gini_berries_consumed.csv")
        total_berries_consumed.to_csv(self.filepath+"total_berries_consumed.csv")

    def _calculate_column_across_episode(self, df_list, df_labels, column, calculation):
            data = []
            for df in df_list:
                series = df.groupby("day")[column].apply(calculation)
                data.append(series)
            df = pd.DataFrame(data).T
            df.columns = df_labels
            return df

    def _display_swarm_plot(self, df_list, df_labels, column, filename):
        fig, ax = plt.subplots()
        # Combine the DataFrames and add labels
        combined_df = pd.concat([df.assign(label=label) for df, label in zip(df_list, df_labels)])
        # Plot the swarm plot with reduced marker size
        sns.swarmplot(data=combined_df, x=column, y='label', ax=ax, size=3, hue='label')  # Adjust size as needed
        plt.xlabel(column)
        plt.ylabel('Society')
        plt.title('Swarm Plot of ' + column + ' by Society')
        plt.tight_layout()
        plt.show()
        plt.savefig(str(filename).split()[0])

    def _display_violin_plot_df_list(self, df_list, df_labels, column, filename, title, y_label):
        fig, ax = plt.subplots()
        combined_df = pd.concat([df.assign(label=label) for df, label in zip(df_list, df_labels)])
        colors = sns.color_palette("colorblind", n_colors=len(df_labels))
        sns.violinplot(
            data=combined_df,
            x="label",
            y=column,
            hue="label",
            palette=colors,
            ax=ax,
            legend=False,
        )

        plt.xlabel("Society")
        plt.ylabel(y_label)
        plt.title(title)
        plt.show()
        plt.savefig(str(filename).split()[0])
    
    def _display_dataframe(self, df, title, y_label, filename):
        sns.set_palette("colorblind")
        ax = sns.lineplot(data=df)
        ax.set_xlabel("day")
        ax.set_ylabel(y_label)
        ax.legend(title="Societies", loc="upper left")
        plt.title(title)
        plt.show()
        plt.savefig(str(filename).split()[0])
        plt.close()

    def _apply_function_to_list(self, list, function):
            results_list = []
            for item in list:
                result = function(item)
                results_list.append(result)
            return results_list

    def _calculate_max(self, series):
        return series.max()

    def _calculate_min(self, series):
        return series.min()

    def _calculate_variance(self, series):
        return series.var()

    def _calculate_gini(self, series):
        #sort series in ascending order
        x = sorted(series)
        s = sum(x)
        if s == 0:
            return 0
        N = self.num_agents
        #for each element xi, compute xi * (N - i); divide by num agents * sum
        B = sum(xi * (N - i) for i, xi in enumerate(x)) / (N * s)
        return 1 + (1 / N) - 2 * B
    
    def _calculate_total(self, series):
        return series.sum()

    def _create_norms_dataframe(self, filename, write_name=None):
        norms_data = []
        with open(filename, 'r') as file:
            data = json.load(file)
        # Iterate through each episode key (e.g., "100")
        for episode, rules in data.items():
            for rule in rules:
                # Extract attributes from each rule dictionary
                attributes = rule.popitem()[1]  # Get first key-value pair (rule name and its dictionary)
                attributes['episode'] = episode  # Add episode information
                norms_data.append(attributes)  # Append extracted attributes to data list
        # Create DataFrame from the data list
        df = pd.DataFrame(norms_data)
        if write_name != None:
            df.to_csv(write_name+".csv")
        return df