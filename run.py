from src.scenarios.basic_harvest import BasicHarvest
from src.scenarios.capabilities_harvest import CapabilitiesHarvest
from src.scenarios.allotment_harvest import AllotmentHarvest
from src.data_analysis import DataAnalysis
from src.render_pygame import RenderPygame
import pandas as pd
import argparse
import wandb
import numpy as np

AGENT_TYPES = ["baseline", "egalitarian", "maximin", "rawlsian", "utilitarian"]
NUM_AGENTS = 4
NUM_START_BERRIES = 4
MAX_WIDTH = NUM_AGENTS * 2
MAX_HEIGHT = MAX_WIDTH

def generate_graphs(scenario):
    """
    takes raw files and generates graphs displayed in the paper
    processed dfs contain data for each agent at the end of each episode
    e_epochs are run for at most t_max steps; results are normalised by frequency of step
    """
    dataAnalysis = DataAnalysis()
    path = "data/"+scenario+"/"
    files = [path+"baseline.csv",path+"egalitarian.csv",path+"maximin.csv",path+"rawlsian.csv",path+"utilitarian.csv"]
    labels = AGENT_TYPES
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)
    normalised_sum_df_list, agent_end_episode_list = dataAnalysis.process_agent_dfs(dfs, labels)
    dataAnalysis.display_graphs(normalised_sum_df_list, agent_end_episode_list, labels)

def log_wandb_agents(modelInst, last_episode, reward_tracker):
    for i, agent in enumerate(modelInst.schedule.agents):
        if agent.type != "berry":
            base_string = agent.type+"_agent_"+str(agent.unique_id)
            if last_episode != modelInst.episode:
                string = base_string+"_total_episode_reward"
                reward = reward_tracker[i]
                wandb.log({string: reward})
            string = base_string+"_reward"
            wandb.log({string: agent.current_reward})
            mean_loss = (np.mean(agent.losses) if modelInst.training else 0)
            string = base_string+"_mean_loss"
            wandb.log({string: mean_loss})
            string = base_string+"_epsilon"
            wandb.log({string: agent.epsilon})

def run_simulation(modelInst, render, log_wandb):
    if log_wandb:
        wandb.init(project="PriENE")
    if render:
        renderInst = RenderPygame(modelInst.max_width, modelInst.max_height)
    while (modelInst.training and modelInst.epsilon > modelInst.min_epsilon) or (not modelInst.training and modelInst.episode <= modelInst.max_episodes):
        modelInst.step()
        if log_wandb:
            reward_tracker = [a.total_episode_reward for a in modelInst.schedule.agents if a.type != "berry"]
            log_wandb_agents(modelInst, modelInst.episode, reward_tracker)
            mean_reward = modelInst.model_reporter["mean_reward"].mean()
            wandb.log({'mean_reward_test': mean_reward})
        if render:
            renderInst.render_pygame(modelInst)
    num_episodes = modelInst.episode
    return num_episodes

def create_and_run_model(scenario,num_agents,num_start_berries,agent_type,max_width,max_height,max_episodes,training,write_data,write_norms,render,log_wandb):   
    file_string = scenario+"_"+agent_type
    if scenario == "basic":
        modelInst = BasicHarvest(num_agents,num_start_berries,agent_type,max_width,max_height,max_episodes,training,write_data,write_norms,file_string)
    elif scenario == "capabilities":
        modelInst = CapabilitiesHarvest(num_agents,num_start_berries,agent_type,max_width,max_height,max_episodes,training,write_data,write_norms,file_string)
    elif scenario == "allotment":
        modelInst = AllotmentHarvest(num_agents,num_start_berries,agent_type,max_width,max_height,max_episodes,training,write_data,write_norms,file_string)
    else:
        ValueError("Unknown argument: "+scenario)
    run_simulation(modelInst,render,log_wandb)

def run_all(scenario,num_start_berries,max_width,max_height,max_episodes,training,write_data,write_norms,render,log_wandb):
    for agent_type in AGENT_TYPES:
        create_and_run_model(scenario,num_start_berries,agent_type,max_width,max_height,max_episodes,training,write_data,write_norms,render,log_wandb)

def get_integer_input(prompt):
    while True:
        try:
            value = int(input(prompt))
            return value
        except ValueError:
            print("Invalid input. Please enter an integer.")

def write_data_input(data_type):
    write_data = input(f"Do you want to write {data_type} to file? (y, n): ")
    while write_data not in ["y", "n"]:
        write_data = input("Invalid choice. Please choose 'y' or 'n': ")
    if write_data == "y":
        write_data = True
        print(f"{data_type} will be written into data/results.")
    elif write_data == "n":
        write_data = False
    return write_data

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
    #########################################################################################
    agent_type = input("What type of agent do you want to implement (baseline, maximin, egalitarian, utilitarian, all): ")
    while agent_type not in AGENT_TYPES:
        agent_type = input("Invalid agent type. Please choose 'baseline', 'maximin', 'egalitarian', or 'utilitarian', or 'all': ")
    #########################################################################################
    write_data = write_data_input("data")
    #########################################################################################
    write_norms = write_data_input("norms")
    #########################################################################################
    render = input("Do you want to render the simulation? (y, n): ")
    while render not in ["y", "n"]:
        render = input("Invalid choice. Please choose 'y' or 'n': ")
    if render == "y":
        render = True
    elif render == "n":
        render = False
    #########################################################################################
    if args.option == "train":
        training = True
        print("Model variables will be written into model_variables/current_run")
        max_episodes = 0
    else:
        max_episodes = get_integer_input("How many episodes do you want to run: ")
        training = False
    #########################################################################################
    if args.log is not None:
        log_wandb = True
    else:
        log_wandb = False
    if agent_type == "all":
        run_all(scenario,NUM_AGENTS,NUM_START_BERRIES,MAX_WIDTH,MAX_HEIGHT,max_episodes,training,write_data,write_norms,render,log_wandb)
    else:
        create_and_run_model(scenario,NUM_AGENTS,NUM_START_BERRIES,agent_type,MAX_WIDTH,MAX_HEIGHT,max_episodes,training,write_data,write_norms,render,log_wandb)
#########################################################################################
elif args.option == "generate_graphs":
    scenario = input("What type of scenario do you want to generate graphs for (capabilities, allotment): ")
    while scenario not in ["capabilities", "allotment"]:
        scenario = input("Invalid scenario. Please choose 'capabilities', or 'allotment': ")
    print("Graphs will be saved in data/results")
    generate_graphs(scenario)