import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import csv
import numpy as np
from collections import Counter
from src.data_handling.norm_processing import NormProcessing
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

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

    def proccess_and_display_data(self, agent_df_list, principles, aggregations, scenario, norms_filepath=None, end_episode_totals_dfs=None, write=True, get_normalised=False, process_norms=True):
        df_labels = principles + aggregations
        self.principles = principles
        self.aggregations = aggregations
        #if need to analyse data
        if end_episode_totals_dfs == None:
            end_episode_totals_dfs, agent_final_rows_dfs, normalised_df_list = self._process_agent_dfs(agent_df_list, df_labels, write, get_normalised)
        else:
            #if data has already been analysed
            agent_final_rows_dfs = agent_df_list
        self._display_graphs(agent_final_rows_dfs, end_episode_totals_dfs, df_labels, normalised_df_list)
        self._test_all_variables_significance(agent_final_rows_dfs, df_labels, "agent_end")
        self._test_all_variables_significance(end_episode_totals_dfs, df_labels, "end_episode")
        if process_norms:
            self._process_norms(df_labels, scenario, norms_filepath)

    def _process_agent_dfs(self, agent_df_list, df_labels, write, get_normalised):
        end_episode_totals_df_list = []
        end_episode_central_tendencies = []
        agent_end_episode_df_list = []
        if get_normalised:
            normalised_df_list = []
            write_normalised = True
        else:
            normalised_df_list = None
        for i, df in enumerate(agent_df_list):
            agent_end_episode = self._agent_end_episode_dataframes(df, df_labels[i], write)
            agent_end_episode_df_list.append(agent_end_episode)
            end_episode_totals = self._end_episode_totals(agent_end_episode, df_labels[i], write)
            end_episode_totals_df_list.append(end_episode_totals)
            central_tendency = self._calculate_central_tendency(end_episode_totals, df_labels[i])
            central_tendency["episode_length_mean"] = agent_end_episode["day"].mean()
            central_tendency["episode_length_median"] = agent_end_episode["day"].median()
            central_tendency["episode_length_stdev"] = agent_end_episode["day"].std()
            end_episode_central_tendencies.append(central_tendency)
            if get_normalised:
                normalised_df_list.append(self._normalise_step_across_episodes(df, df_labels[i], write_normalised))
        central_tendencies = self._write_dictionary_to_file(end_episode_central_tendencies,self.filepath+"central_tendencies.csv")
        most_common = self._get_best_results(central_tendencies, "aggregations")
        self.principles += [most_common]
        self._get_best_results(central_tendencies, "principles")
        self._get_best_results(central_tendencies, "all")
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
        cooperative_tendencies = []
        for label in df_labels:
            input_file = filepath+label+"_emerged_norms.json"
            output_file = self.filepath+scenario+"_"+label+"_norms"
            df = norm_processing.proccess_norms(input_file, output_file, label)
            cooperative_dfs.append(df)
            cooperative_tendencies.append(self._calculate_norms_tendency(df, label))
        self._display_swarm_plot(cooperative_dfs,df_labels, "numerosity", filepath+"cooperative_numerosity")
        self._display_swarm_plot(cooperative_dfs,df_labels, "fitness", filepath+"cooperative_fitness")
        self._display_swarm_plot(cooperative_dfs,df_labels, "reward", filepath+"cooperative_reward")
        self._write_dictionary_to_file(cooperative_tendencies,self.filepath+"cooperative_norms_tendencies.csv")

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
            min_days=('total_days_left_to_live', 'min'),
            max_days=('total_days_left_to_live', 'max'),
            average_days=('total_days_left_to_live', 'mean'),
            total_days=('total_days_left_to_live', 'sum'),
            gini_days=('total_days_left_to_live', lambda x: self._calculate_gini(x)),
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
        days_left_to_live_tendency["min_well_being"] = self._calculate_normalised_central_tendency(min_days_left_to_live, df_labels)
        total_days_left_to_live = self._calculate_column_across_episode(sum_df_list, df_labels, "days_left_to_live", self._calculate_total)
        self._display_dataframe(total_days_left_to_live, "Total Days Left To Live", "Days Left To Live", filename+"_total")
        days_left_to_live_tendency["total_well_being"] = self._calculate_normalised_central_tendency(total_days_left_to_live, df_labels)
        gini_days_left_to_live = self._calculate_column_across_episode(sum_df_list, df_labels, "days_left_to_live", self._calculate_gini)
        self._display_dataframe(gini_days_left_to_live, "Gini Index of Days Left To Live", "Days Left To Live", filename+"_gini")
        days_left_to_live_tendency["gini_well_being"] = self._calculate_normalised_central_tendency(gini_days_left_to_live, df_labels)
        max_days_left_to_live.to_csv(self.filepath+"max_days_left_to_live.csv")
        min_days_left_to_live.to_csv(self.filepath+"min_days_left_to_live.csv")
        gini_days_left_to_live.to_csv(self.filepath+"gini_days_left_to_live.csv")
        total_days_left_to_live.to_csv(self.filepath+"total_days_left_to_live.csv")
        # with open(self.filepath+"tendency_well_being.json", "w") as f:
        #     json.dump(days_left_to_live_tendency, f, indent=4)

    def _berries_consumed_results(self, sum_df_list, df_labels, filename):
        berries_consumed_tendency = {}
        max_berries_consumed = self._calculate_column_across_episode(sum_df_list, df_labels, "berries_consumed", self._calculate_max)
        self._display_dataframe(max_berries_consumed, "Max Berries Consumed", "Berries Consumed", filename+"_max")
        min_berries_consumed = self._calculate_column_across_episode(sum_df_list, df_labels, "berries_consumed", self._calculate_min)
        self._display_dataframe(min_berries_consumed, "Min Berries Consumed", "Berries Consumed", filename+"_min")
        berries_consumed_tendency["min_berries"] = self._calculate_normalised_central_tendency(min_berries_consumed, df_labels)
        total_berries_consumed = self._calculate_column_across_episode(sum_df_list, df_labels, "berries_consumed", self._calculate_total)
        self._display_dataframe(total_berries_consumed, "Total Berries Consumed", "Berries Consumed", filename+"_total")
        berries_consumed_tendency["total_berries"] = self._calculate_normalised_central_tendency(total_berries_consumed, df_labels)
        gini_berries_consumed = self._calculate_column_across_episode(sum_df_list, df_labels, "berries_consumed", self._calculate_gini)
        self._display_dataframe(gini_berries_consumed, "Gini Index of Berries Consumed", "Berries Consumed", filename+"_gini")
        berries_consumed_tendency["gini_berries"] = self._calculate_normalised_central_tendency(gini_berries_consumed, df_labels)
        max_berries_consumed.to_csv(self.filepath+"max_berries_consumed.csv")
        min_berries_consumed.to_csv(self.filepath+"min_berries_consumed.csv")
        gini_berries_consumed.to_csv(self.filepath+"gini_berries_consumed.csv")
        total_berries_consumed.to_csv(self.filepath+"total_berries_consumed.csv")
        # with open(self.filepath+"tendency_berries_consumed.json", "w") as f:
        #     json.dump(berries_consumed_tendency, f, indent=4)

    def _display_swarm_plot(self, df_list, df_labels, column, filename):
        fig, ax = plt.subplots()
        #combine the DataFrames and add labels
        combined_df = pd.concat([df.assign(label=label) for df, label in zip(df_list, df_labels)])
        #plot the swarm plot with reduced marker size
        sns.stripplot(data=combined_df, x=column, y='label', ax=ax, size=2, hue='label', dodge=True, alpha=0.6)
        plt.xlabel(column)
        plt.ylabel('Society')
        plt.title('Swarm Plot of ' + column + ' by Society')
        plt.tight_layout()
        plt.savefig(str(filename).split()[0])
        plt.close(fig)
    
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
        plt.savefig(str(filename).split()[0])
        plt.close(fig)
    
    def _display_dataframe(self, df, title, y_label, filename):
        sns.set_palette("colorblind")
        ax = sns.lineplot(data=df)
        ax.set_xlabel("day")
        ax.set_ylabel(y_label)
        ax.legend(title="Societies", loc="upper left")
        plt.title(title)
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
        plt.savefig(str(filename).split()[0])
        plt.close()
        
    def _write_dictionary_to_file(self, dictionary, filepath):
        df = pd.DataFrame(dictionary)
        df.to_csv(filepath, index=False)
        return df
    
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
                            "gini_days_mean": df["gini_days_survived"].mean(),
                            "gini_days_median": df["gini_days_survived"].median(),
                            "gini_days_stdev": df["gini_days_survived"].std(),
                            "gini_berries_mean": df["gini_berries"].mean(),
                            "gini_berries_median": df["gini_berries"].median(),
                            "gini_berries_stdev": df["gini_berries"].std(),
                            "min_days_mean": df["min_days_survived"].mean(),
                            "min_days_median": df["min_days_survived"].median(),
                            "min_days_stdev": df["min_days_survived"].std(),
                            "min_berries_mean": df["min_berries"].mean(),
                            "min_berries_median": df["min_berries"].median(),
                            "min_berries_stdev": df["min_berries"].std(),
                            "max_days_mean": df["max_days_survived"].mean(),
                            "max_days_median": df["max_days_survived"].median(),
                            "max_days_stdev": df["max_days_survived"].std(),
                            "max_berries_mean": df["max_berries"].mean(),
                            "max_berries_median": df["max_berries"].median(),
                            "max_berries_stdev": df["max_berries"].std(),
                            "total_days_mean": df["total_days_survived"].mean(),
                            "total_days_median": df["total_days_survived"].median(),
                            "total_days_stdev": df["total_days_survived"].std(),
                            "total_berries_mean": df["total_berries"].mean(),
                            "total_berries_median": df["total_berries"].median(),
                            "total_berries_stdev": df["total_berries"].std(),
                            }
        return central_tendency
    
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
    
    def _calculate_normalised_central_tendency(self, df, df_labels):
        pass
    
    def _get_best_results(self, df, run_type):
        best_results = []
        test_names = []
        test_name_column = df.columns[0]
        if run_type == "principles":
            df = df[df[test_name_column].isin(self.principles)]
        elif run_type == "aggregations":
            df = df[df[test_name_column].isin(self.aggregations)]
        for column in df.columns[1:]:
            if "gini" in column.lower() or "stdev" in column.lower():
                index = df[column].idxmin()
                type = "Minimum"
            else:
                index = df[column].idxmax()
                type = "Maximum"
            value = df.loc[index, column]
            test_name = df.loc[index, test_name_column]
            best_results.append({
                "Metric": column,
                "Test Name": test_name,
                "Value": value,
                "Type": type
            })
            test_names.append(test_name)
        most_common, tally = Counter(test_names).most_common(1)[0] #Find the most common test and the tally
        best_results.append({
                "Metric": "most_common",
                "Test Name": most_common,
                "Value": tally,
                "Type": np.nan
            })
        self._write_dictionary_to_file(best_results,self.filepath+"best_results_"+run_type+".csv")
        return most_common

    # def _cohens_d(self, x_series, y_series):
    #     nx = len(x_series)
    #     ny = len(y_series)
    #     if nx != ny:
    #         nx = len(x_series)
    #         ny = len(y_series)
    #         dof = nx + ny - 2
    #         return (x_series.mean() - y_series.mean()) / np.sqrt(((nx-1)*x_series.std() ** 2 + (ny-1)*y_series.std() ** 2) / dof)
    #     else:
    #         return (x_series.mean() - y_series.mean()) / np.sqrt((x_series.std() ** 2 + y_series.std() ** 2) / 2.0)

    def _cohens_d(self, x_series, y_series):
        """
        Compute Cohen's d for two samples (works for unequal lengths).
        """
        nx, ny = len(x_series), len(y_series)
        mean_diff = x_series.mean() - y_series.mean()
        # pooled standard deviation
        s_pooled = np.sqrt(((nx - 1) * x_series.std(ddof=1) ** 2 + (ny - 1) * y_series.std(ddof=1) ** 2) / (nx + ny - 2))
        if s_pooled == 0:
            return 0.0  # avoid division by zero
        return mean_diff / s_pooled

    def _test_all_variables_significance(self, dfs, df_labels, df_type):
        variables = dfs[0].columns
        variables = variables[1:]
        exclude_list = ["agent_id", "episode", "action"]
        variables = [var for var in variables if var not in exclude_list]
        for dependent_variable in variables:
            if not np.issubdtype(dfs[0][dependent_variable].dtype, np.number):
                #skip non-numeric variables
                continue
            anova_table, tukey_results, anova, tukey = self._perform_anova(dfs, df_labels, dependent_variable)
            if tukey:
                with open(self.filepath+"tukey_results_"+df_type+"_"+dependent_variable+".txt", "w") as f:
                    f.write(str(tukey_results.summary()))
    
    def _perform_anova(self, dfs, df_labels, dependent_variable):
        all_data = []
        for i, df in enumerate(dfs):
            temp_df = df[[dependent_variable]].copy()
            temp_df["society"] = df_labels[i]
            all_data.append(temp_df)
        combined_df = pd.concat(all_data, ignore_index=True)
        try:
            #ordinary least squares: dependent variable is predicted by society (which is a category)
            #fit linear regression model to data
            model = ols(f'{dependent_variable} ~ C(society)', data=combined_df).fit()
            #perform anova test based on fitted linear model, use type 2 sum of squares
            anova_table = sm.stats.anova_lm(model, typ=2)
        except Exception as e:
            print(f"Exception during ANOVA test for {dependent_variable}: {e}")
            return None, None, False, False
        try:
            #perform Tukey's HSD post-hoc test used after a significant anova to determine which specific groups are significantly different
            #significance level 0.05 (95% confidence level)
            tukey_results = pairwise_tukeyhsd(combined_df[dependent_variable], combined_df["society"], alpha=0.05)
            return anova_table, tukey_results, True, True
        except Exception as e:
            print(f"Exception during post hoc test for {dependent_variable}: {e}")
            return anova_table, None, True, False