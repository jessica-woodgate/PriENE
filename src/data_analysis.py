import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DataAnalysis():
    def __init__(self, num_agents):
        self.num_agents = num_agents

    def process_agent_dfs(self, agent_df_list, df_labels):
        normalised_sum_df_list = self._apply_function_to_list(agent_df_list, self._normalised_sum_step_across_episodes)
        self._write_df_list_to_file(normalised_sum_df_list, df_labels, "data/current_run/normalised_sum_df_")
        agent_end_episode_list = self._process_end_episode_dataframes(agent_df_list)
        self._write_df_list_to_file(agent_end_episode_list, df_labels, "data/current_run/processed_episode_df_")
        return normalised_sum_df_list, agent_end_episode_list
    
    def display_graphs(self, normalised_sum_df_list, agent_end_episode_list, df_labels):
        self._days_left_to_live_results(normalised_sum_df_list, df_labels, "data/current_run/days_left_to_live")
        self._display_violin_plot_df_list(agent_end_episode_list, df_labels, "day", "data/current_run/violin_end_day", "Violin Plot of Episode Length", "End Day")
        self._display_violin_plot_df_list(agent_end_episode_list, df_labels, "total_berries", "data/current_run/violin_total_berries", "Violin Plot of Total Berries Consumed", "Berries Consumed")

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
            grouped_df = df.groupby("episode")
            episode_dfs = []
            for episode, group_df in grouped_df:
                last_two_rows = group_df.tail(2).loc[:, ~group_df.columns.str.contains('^Unnamed')]
                last_two_rows["total_berries"] = last_two_rows["berries"] + last_two_rows["berries_consumed"]
                episode_dfs.append(last_two_rows)
            processed_df = pd.concat(episode_dfs)
            processed_dfs.append(processed_df)
        return processed_dfs

    def _days_left_to_live_results(self, sum_df_list, df_labels, filename):
        max_days_left_to_live = self._calculate_column_across_episode(sum_df_list, df_labels, "days_left_to_live", self._calculate_max)
        self._display_dataframe(max_days_left_to_live, "Max Days Left To Live", "Days Left To Live", filename+"_max")
        min_days_left_to_live = self._calculate_column_across_episode(sum_df_list, df_labels, "days_left_to_live", self._calculate_min)
        self._display_dataframe(min_days_left_to_live, "Min Days Left To Live", "Days Left To Live", filename+"_min")
        var_days_left_to_live = self._calculate_column_across_episode(sum_df_list, df_labels, "days_left_to_live", self._calculate_variance)
        self._display_dataframe(var_days_left_to_live, "Variance Days Left To Live", "Days Left To Live", filename+"_var")
        total_days_left_to_live = self._calculate_column_across_episode(sum_df_list, df_labels, "days_left_to_live", self._calculate_total)
        self._display_dataframe(total_days_left_to_live, "Total Days Left To Live", "Days Left To Live", filename+"_total")
        gini_days_left_to_live = self._calculate_column_across_episode(sum_df_list, df_labels, "days_left_to_live", self._calculate_gini)
        self._display_dataframe(gini_days_left_to_live, "Gini Index of Days Left To Live", "Days Left To Live", filename+"_gini")
        max_days_left_to_live.to_csv("data/current_run/max_days_left_to_live.csv")
        min_days_left_to_live.to_csv("data/current_run/min_days_left_to_live.csv")
        var_days_left_to_live.to_csv("data/current_run/var_days_left_to_live.csv")
        gini_days_left_to_live.to_csv("data/current_run/gini_days_left_to_live.csv")
        total_days_left_to_live.to_csv("data/current_run/total_days_left_to_live.csv")

    def _calculate_column_across_episode(self, df_list, df_labels, column, calculation):
            data = []
            for df in df_list:
                series = df.groupby("day")[column].apply(calculation)
                data.append(series)
            df = pd.DataFrame(data).T
            df.columns = df_labels
            return df

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