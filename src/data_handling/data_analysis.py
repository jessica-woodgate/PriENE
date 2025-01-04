import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
from src.data_handling.norm_processing import NormProcessing
from scipy import stats

class DataAnalysis():
    """
    Data analysis processes data of societal metrics, data of norms, and displays graphs
    Instance variables:
        num_agents -- number of agents in the simulation
        filepath -- path of file to save data to
    """
    def __init__(self, num_agents, filepath):
        self.num_agents = num_agents
        self.filepath = filepath
    
    def proccess_and_display_data(self, agent_df_list, df_labels, get_normalised=False):
        end_episode_totals_df_list, agent_final_rows_df_list, normalised_df_list = self._process_agent_dfs(agent_df_list, df_labels, get_normalised)
        self._display_graphs(agent_final_rows_df_list, end_episode_totals_df_list, df_labels, normalised_df_list)
        #self._process_norms(df_labels, scenario, norms_filepath)

    def _process_agent_dfs(self, agent_df_list, df_labels, get_normalised):
        write = False
        end_episode_totals_df_list = []
        end_episode_central_tendencies = []
        agent_end_episode_df_list = []
        if get_normalised:
            normalised_df_list = []
        else:
            normalised_df_list = None
        for i, df in enumerate(agent_df_list):
            agent_end_episode = self._agent_end_episode_dataframes(df, df_labels[i], write)
            agent_end_episode_df_list.append(agent_end_episode)
            end_episode_totals = self._end_episode_totals(agent_end_episode, df_labels[i], write)
            end_episode_totals_df_list.append(end_episode_totals)
            end_episode_central_tendencies.append(self._calculate_central_tendency(end_episode_totals, df_labels[i]))
            if get_normalised:
                normalised_df_list.append(self._normalise_step_across_episodes(df, df_labels[i]))
        pd.DataFrame(end_episode_central_tendencies).to_csv(self.filepath+"central_tendencies_end_episode_totals.csv",index=False)
        return end_episode_totals_df_list, agent_end_episode_df_list, normalised_df_list
    
    def _display_graphs(self, agent_end_episode_list, end_episode_df_list, df_labels, normalised_sum_df_list=None):
        if normalised_sum_df_list != None:
            self._days_left_to_live_results(normalised_sum_df_list, df_labels, self.filepath+"days_left_to_live")
            self._berries_consumed_results(normalised_sum_df_list, df_labels, self.filepath+"berries_consumed")
        self._display_end_episode(end_episode_df_list, df_labels)
        self._display_violin_plot_df_list(agent_end_episode_list, df_labels, "day", self.filepath+"violin_end_day", "Violin Plot of Episode Length", "End Day")
        #self._display_violin_plot_df_list(agent_end_episode_list, df_labels, "total_berries", self.filepath+"violin_total_berries", "Violin Plot of Total Berries Consumed", "Berries Consumed")

    def _process_norms(self, df_labels, scenario, filepath):
        norm_processing = NormProcessing()
        cooperative_dfs = []
        for label in df_labels:
            input_file = filepath+"_"+label+"_emerged_norms.json"
            output_file = self.filepath+scenario+"_"+label+"_norms"
            cooperative_dfs.append(norm_processing.proccess_norms(input_file, output_file))
        self._display_swarm_plot(cooperative_dfs,df_labels, "numerosity", filepath+"_cooperative_numerosity")
        self._display_swarm_plot(cooperative_dfs,df_labels, "fitness", filepath+"_cooperative_fitness")
        self._display_swarm_plot(cooperative_dfs,df_labels, "reward", filepath+"_cooperative_reward")

    def _normalise_step_across_episodes(self, df, df_label, write):
        """
        normalises results for each step in each episode by how frequently the step occurred
        final df has one row for each step for the length of one episode
        writes each normalised sum df to file
        returns list of dfs
        """
        df = df.drop(["episode", "action", "reward"], axis=1)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        #Calculate counts for each (step, agent_id) combination
        count_df = df.groupby(["day", "agent_id"]).size().reset_index(name="count")
        count_df['count'] = count_df['count']
        #Sum and normalise by count
        sum_df = df.groupby(["day", "agent_id"]).sum().reset_index()
        sum_df = sum_df.astype(float)
        to_divide_columns = list(sum_df.columns)
        to_divide_columns.remove("day")
        to_divide_columns.remove("agent_id")
        sum_df.loc[:, to_divide_columns] = sum_df.loc[:, to_divide_columns]
        sum_df.loc[:, to_divide_columns] = sum_df.loc[:, to_divide_columns].divide(count_df["count"], axis=0)
        sum_df["count"] = count_df["count"]
        if write:
            sum_df.to_csv(self.filepath+"normalised_sum_df_"+df_label+".csv")
        return sum_df

    def _end_episode_totals(self, df, df_label, write):
        """
        gets metric totals at the end of each episode
        final df has one row for each episode
        writes each metric totals df to file
        returns list of dfs
        """
        episode_totals_df = df.groupby("episode").agg(
            min_days=('total_days', 'min'),
            max_days=('total_days', 'max'),
            average_days=('total_days', 'mean'),
            total_days=('total_days', 'sum'),
            gini_days=('total_days', lambda x: self._calculate_gini(x)),
            min_berries=('total_berries', 'min'),
            max_berries=('total_berries', 'max'),
            average_berries=('total_berries', 'mean'),
            total_berries=('total_berries', 'sum'),
            gini_berries=('total_berries', lambda x: self._calculate_gini(x)),
            min_days_survived=('day', 'min'),
            max_days_survived=('day', 'max'),
            average_days_survived=('day', 'mean'),
            total_days_survived=('day', 'sum'),
            gini_days_survived=('day', lambda x: self._calculate_gini(x))
        )
        if write:
            episode_totals_df.to_csv(self.filepath+"end_episode_totals_"+df_label+".csv")
        return episode_totals_df

    def _agent_end_episode_dataframes(self, df, df_label, write):
        """
        for each agent, sum up the berries they're holding and the berries they've eaten to get total across episode
        final df has one row for each agent for each episode
        write each end episode df to file
        returns list of dfs
        """
        grouped_df = df.groupby(["episode", "agent_id"])
        last_rows_list = []
        for (episode, agent_id), group_df in grouped_df:
            last_row = group_df.tail(1).loc[:, ~group_df.columns.str.contains('^Unnamed')]
            last_row["total_berries"] = last_row["berries"] + last_row["berries_consumed"]
            last_rows_list.append(last_row)
        agent_end_episode_df = pd.concat(last_rows_list)
        if write:
            agent_end_episode_df.to_csv(self.filepath+"agent_end_episode_df_"+df_label+".csv",index=False)
        return agent_end_episode_df

    def _days_left_to_live_results(self, sum_df_list, df_labels, filename):
        days_left_to_live_tendency = {}
        max_days_left_to_live = self._calculate_column_across_episode(sum_df_list, df_labels, "days_left_to_live", self._calculate_max)
        self._display_dataframe(max_days_left_to_live, "Max Days Left To Live", "Days Left To Live", filename+"_max")
        min_days_left_to_live = self._calculate_column_across_episode(sum_df_list, df_labels, "days_left_to_live", self._calculate_min)
        self._display_dataframe(min_days_left_to_live, "Min Days Left To Live", "Days Left To Live", filename+"_min")
        days_left_to_live_tendency["min_well_being"] = self._calculate_central_tendency(min_days_left_to_live["baseline"], min_days_left_to_live["maximin"])
        total_days_left_to_live = self._calculate_column_across_episode(sum_df_list, df_labels, "days_left_to_live", self._calculate_total)
        self._display_dataframe(total_days_left_to_live, "Total Days Left To Live", "Days Left To Live", filename+"_total")
        days_left_to_live_tendency["total_well_being"] = self._calculate_central_tendency(total_days_left_to_live["baseline"], total_days_left_to_live["maximin"])
        gini_days_left_to_live = self._calculate_column_across_episode(sum_df_list, df_labels, "days_left_to_live", self._calculate_gini)
        self._display_dataframe(gini_days_left_to_live, "Gini Index of Days Left To Live", "Days Left To Live", filename+"_gini")
        days_left_to_live_tendency["gini_well_being"] = self._calculate_central_tendency(gini_days_left_to_live["baseline"], gini_days_left_to_live["maximin"])
        max_days_left_to_live.to_csv(self.filepath+"max_days_left_to_live.csv")
        min_days_left_to_live.to_csv(self.filepath+"min_days_left_to_live.csv")
        gini_days_left_to_live.to_csv(self.filepath+"gini_days_left_to_live.csv")
        total_days_left_to_live.to_csv(self.filepath+"total_days_left_to_live.csv")
        with open(self.filepath+"tendency_well_being.json", "w") as f:
            json.dump(days_left_to_live_tendency, f, indent=4)

    def _berries_consumed_results(self, sum_df_list, df_labels, filename):
        berries_consumed_tendency = {}
        max_berries_consumed = self._calculate_column_across_episode(sum_df_list, df_labels, "berries_consumed", self._calculate_max)
        self._display_dataframe(max_berries_consumed, "Max Berries Consumed", "Berries Consumed", filename+"_max")
        min_berries_consumed = self._calculate_column_across_episode(sum_df_list, df_labels, "berries_consumed", self._calculate_min)
        self._display_dataframe(min_berries_consumed, "Min Berries Consumed", "Berries Consumed", filename+"_min")
        berries_consumed_tendency["min_berries"] = self._calculate_central_tendency(min_berries_consumed["baseline"], min_berries_consumed["maximin"])
        total_berries_consumed = self._calculate_column_across_episode(sum_df_list, df_labels, "berries_consumed", self._calculate_total)
        self._display_dataframe(total_berries_consumed, "Total Berries Consumed", "Berries Consumed", filename+"_total")
        berries_consumed_tendency["total_berries"] = self._calculate_central_tendency(total_berries_consumed["baseline"], total_berries_consumed["maximin"])
        gini_berries_consumed = self._calculate_column_across_episode(sum_df_list, df_labels, "berries_consumed", self._calculate_gini)
        self._display_dataframe(gini_berries_consumed, "Gini Index of Berries Consumed", "Berries Consumed", filename+"_gini")
        berries_consumed_tendency["gini_berries"] = self._calculate_central_tendency(gini_berries_consumed["baseline"], gini_berries_consumed["maximin"])
        max_berries_consumed.to_csv(self.filepath+"max_berries_consumed.csv")
        min_berries_consumed.to_csv(self.filepath+"min_berries_consumed.csv")
        gini_berries_consumed.to_csv(self.filepath+"gini_berries_consumed.csv")
        total_berries_consumed.to_csv(self.filepath+"total_berries_consumed.csv")
        with open(self.filepath+"tendency_berries_consumed.json", "w") as f:
            json.dump(berries_consumed_tendency, f, indent=4)

    def _display_swarm_plot(self, df_list, df_labels, column, filename):
        fig, ax = plt.subplots()
        #combine the DataFrames and add labels
        combined_df = pd.concat([df.assign(label=label) for df, label in zip(df_list, df_labels)])
        #plot the swarm plot with reduced marker size
        sns.swarmplot(data=combined_df, x=column, y='label', ax=ax, size=3, hue='label')  # Adjust size as needed
        plt.xlabel(column)
        plt.ylabel('Society')
        plt.title('Swarm Plot of ' + column + ' by Society')
        plt.tight_layout()
        plt.show()
        plt.savefig(str(filename).split()[0])
    
    def _display_end_episode(self, df_list, df_labels):
        self._display_violin_plot_df_list(df_list, df_labels, "gini_days_survived", self.filepath+"gini_days_survived", "Violin Plot of Gini Days Survived", "Days Survived")
        self._display_violin_plot_df_list(df_list, df_labels, "min_days_survived", self.filepath+"min_days_survived", "Violin Plot of Min Days Survived", "Days Survived")
        self._display_violin_plot_df_list(df_list, df_labels, "max_days_survived", self.filepath+"max_days_survived", "Violin Plot of Max Days Survived", "Days Survived")
        self._display_violin_plot_df_list(df_list, df_labels, "total_days_survived", self.filepath+"total_days_survived", "Violin Plot of Total Days Survived", "Days Survived")
        self._display_violin_plot_df_list(df_list, df_labels, "gini_berries", self.filepath+"gini_berries", "Violin Plot of Gini Berries Eaten", "Berries Eaten")
        self._display_violin_plot_df_list(df_list, df_labels, "min_berries", self.filepath+"min_berries", "Violin Plot of Min Berries Eaten", "Berries Eaten")
        self._display_violin_plot_df_list(df_list, df_labels, "max_berries", self.filepath+"max_berries", "Violin Plot of Max Berries Eaten", "Berries Eaten")
        self._display_violin_plot_df_list(df_list, df_labels, "total_berries", self.filepath+"total_berries", "Violin Plot of Total Berries Eaten", "Berries Eaten")

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

    def _display_dataframe_shaded(self, df, title, y_label, filename):
        sns.set_palette("colorblind")
        ax = sns.lineplot(data=df)
        ax.set_xlabel("day")
        ax.set_ylabel(y_label)
        ax.legend(title="Societies", loc="upper left")
        plt.title(title)
        for col in df.columns:
            mean = df[col].mean()
            sem = df[col].sem()
            x = df["day"]
            ax.fill_between(x, mean + sem, mean - sem, alpha=0.2)
        plt.show()
        plt.savefig(str(filename).split()[0])
        plt.close()
    
    def _write_df_list_to_file(self, df_list, df_labels, filepath):
        i = 0
        for df in df_list:
            df.to_csv(filepath+df_labels[i]+".csv")
            i += 1

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
    
    def _calculate_column_across_episode(self, df_list, df_labels, column, calculation):
            data = []
            for df in df_list:
                series = df.groupby("day")[column].apply(calculation)
                data.append(series)
            df = pd.DataFrame(data).T
            df.columns = df_labels
            return df

    def _calculate_central_tendency(self, df, df_label):
        central_tendency = {"df_label": df_label,
                            "mean_gini_days": df["gini_days_survived"].mean(),
                            "median_gini_days": df["gini_days_survived"].median(),
                            "stdev_gini_days": df["gini_days_survived"].std(),
                            "mean_min_days": df["min_days_survived"].mean(),
                            "median_min_days": df["min_days_survived"].median(),
                            "stdev_min_days": df["min_days_survived"].std(),
                            "mean_max_days": df["max_days_survived"].mean(),
                            "median_max_days": df["max_days_survived"].median(),
                            "stdev_max_days": df["max_days_survived"].std(),
                            "mean_total_days": df["total_days_survived"].mean(),
                            "median_total_days": df["total_days_survived"].median(),
                            "stdev_total_days": df["total_days_survived"].std(),
                            "mean_gini_berries": df["gini_berries"].mean(),
                            "median_gini_berries": df["gini_berries"].median(),
                            "stdev_gini_berries": df["gini_berries"].std(),
                            "mean_min_berries": df["min_berries"].mean(),
                            "median_min_berries": df["min_berries"].median(),
                            "stdev_min_berries": df["min_berries"].std(),
                            "mean_max_berries": df["max_berries"].mean(),
                            "median_max_berries": df["max_berries"].median(),
                            "stdev_max_berries": df["max_berries"].std(),
                            "mean_total_berries": df["total_berries"].mean(),
                            "median_total_berries": df["total_berries"].median(),
                            "stdev_total_berries": df["total_berries"].std(),
                            }
        return central_tendency

    def _cohens_d(self, x_series, y_series):
        nx = len(x_series)
        ny = len(y_series)
        if nx != ny:
            nx = len(x_series)
            ny = len(y_series)
            dof = nx + ny - 2
            return (x_series.mean() - y_series.mean()) / np.sqrt(((nx-1)*x_series.std() ** 2 + (ny-1)*y_series.std() ** 2) / dof)
        else:
            return (x_series.mean() - y_series.mean()) / np.sqrt((x_series.std() ** 2 + y_series.std() ** 2) / 2.0)