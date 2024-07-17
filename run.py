from src.scenarios.basic_harvest import BasicHarvest
from src.scenarios.capabilities_harvest import CapabilitiesHarvest
from src.scenarios.allotment_harvest import AllotmentHarvest
from src.data_analysis import DataAnalysis
import pandas as pd
import argparse
import wandb
import numpy as np

agent_types = ["baseline", "egalitarian", "maximin", "rawlsian", "utilitarian"]

def generate_graphs(scenario):
    """
    takes raw files and generates graphs displayed in the paper
    processed dfs contain data for each agent at the end of each episode
    e_epochs are run for at most t_max steps; results are normalised by frequency of step
    """
    data_analysis = DataAnalysis()
    path = "data/"+scenario+"/"
    files = [path+"baseline.csv",path+"egalitarian.csv",path+"maximin.csv",path+"rawlsian.csv",path+"utilitarian.csv"]
    labels = agent_types
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)
    normalised_sum_df_list, agent_end_episode_list = data_analysis.process_agent_dfs(dfs, labels)
    data_analysis.display_graphs(normalised_sum_df_list, agent_end_episode_list, labels)

def log_wandb_agents(model_inst, last_episode, reward_tracker):
    for i, agent in enumerate(model_inst.schedule.agents):
        if agent.type != "berry":
            base_string = agent.type+"_agent_"+str(agent.unique_id)
            if last_episode != model_inst.episode:
                string = base_string+"_total_episode_reward"
                reward = reward_tracker[i]
                wandb.log({string: reward})
            string = base_string+"_reward"
            wandb.log({string: agent.current_reward})
            mean_loss = (np.mean(agent.losses) if model_inst.training else 0)
            string = base_string+"_mean_loss"
            wandb.log({string: mean_loss})
            string = base_string+"_epsilon"
            wandb.log({string: agent.epsilon})

def run_simulation(model_inst, log_wandb):
    if log_wandb:
        wandb.init(project="PriNE")
    while (model_inst.training and model_inst.epsilon > model_inst.min_epsilon) or (not model_inst.training and model_inst.episode <= model_inst.max_episodes):
        model_inst.step()
        if log_wandb:
            reward_tracker = [a.total_episode_reward for a in model_inst.schedule.agents if a.type != "berry"]
            log_wandb_agents(model_inst, model_inst.episode, reward_tracker)
            mean_reward = model_inst.model_reporter["mean_reward"].mean()
            wandb.log({'mean_reward_test': mean_reward})
    num_episodes = model_inst.episode
    return num_episodes

def create_and_run_model(scenario, agent_type, max_episodes, training, write_data, write_norms, log_wandb):   
    num_agents = 2
    file_string = scenario+"_"+agent_type
    if scenario == "basic":
        model_inst = BasicHarvest(num_agents,agent_type,max_episodes,training,write_data,write_norms,file_string)
    elif scenario == "capabilities":
        model_inst = CapabilitiesHarvest(num_agents,agent_type,max_episodes,training,write_data,write_norms,file_string)
    elif scenario == "allotment":
        model_inst = AllotmentHarvest(num_agents,agent_type,max_episodes,training,write_data,write_norms,file_string)
    else:
        ValueError("Unknown argument: "+scenario)
    run_simulation(model_inst,log_wandb)

def run_all(scenario, max_episodes, training, write_data, write_norms, log_wandb):
    for agent_type in agent_types:
        create_and_run_model(scenario, agent_type, max_episodes, training, write_data, write_norms, log_wandb)

def get_integer_input(prompt):
    while True:
        try:
            value = int(input(prompt))
            return value
        except ValueError:
            print("Invalid input. Please enter an integer.")

#########################################################################################

parser = argparse.ArgumentParser(description="Program options")
parser.add_argument("option", choices=["test", "train", "generate_graphs"],
                    help="Choose the program operation")
parser.add_argument("-l", "--log", type=str, default=None,
                    help="Log wandb (optional)")
args = parser.parse_args()

if args.option not in ["test", "train", "generate_graphs"]:
    print("Please choose 'test', 'train', or 'generate_graphs'.")
elif args.option == "test" or args.option == "train":
    if args.option == "test":
        scenario = input("What type of scenario do you want to run (capabilities, allotment): ")
        while scenario not in ["capabilities", "allotment"]:
            scenario = input("Invalid scenario. Please choose 'capabilities', or 'allotment': ")
    else:
        scenario = "basic"
    agent_type = input("What type of agent do you want to implement (baseline, maximin, egalitarian, utilitarian, all): ")
    while agent_type not in agent_types:
        agent_type = input("Invalid agent type. Please choose 'baseline', 'maximin', 'egalitarian', or 'utilitarian', or 'all': ")
    write_data = input("Do you want to write data to file? (y, n): ")
    while write_data not in ["y", "n"]:
        write_data = input("Invalid choice. Please choose 'y' or 'n': ")
    if write_data == "y":
        write_data = True
        print("Data will be written into data/results.")
    elif write_data == "n":
        write_data = False
    write_norms = input("Do you want to write norms to file? (y, n): ")
    while write_norms not in ["y", "n"]:
        write_norms = input("Invalid choice. Please choose 'y' or 'n': ")
    if write_norms == "y":
        write_norms = True
        print("Norms will be written into data/results.")
    elif write_norms == "n":
        write_norms = False
    if args.option == "train":
        training = True
        print("Model variables will be written into model_variables/current_run")
        max_episodes = 0
    else:
        max_episodes = get_integer_input("How many episodes do you want to run: ")
        training = False
    if args.log is not None:
        log_wandb = True
    else:
        log_wandb = False
    if agent_type == "all":
        run_all(scenario, max_episodes, training, write_data, write_norms, log_wandb)
    else:
        create_and_run_model(scenario, agent_type, max_episodes, training, write_data, write_norms, log_wandb)
elif args.option == "generate_graphs":
    scenario = input("What type of scenario do you want to generate graphs for (capabilities, allotment): ")
    while scenario not in ["capabilities", "allotment"]:
        scenario = input("Invalid scenario. Please choose 'capabilities', or 'allotment': ")
    print("Graphs will be saved in data/results")
    generate_graphs(scenario)