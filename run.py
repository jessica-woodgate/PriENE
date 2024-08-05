from src.scenarios.basic_harvest import BasicHarvest
from src.scenarios.capabilities_harvest import CapabilitiesHarvest
from src.scenarios.allotment_harvest import AllotmentHarvest
from src.data_analysis import DataAnalysis
from src.render_pygame import RenderPygame
import pandas as pd
import argparse
import wandb
import numpy as np

#AGENT_TYPES = ["baseline", "egalitarian", "maximin", "utilitarian", "deon_egalitarian", "deon_egalitarian_2", "deon_egalitarian_loss"]
AGENT_TYPES = ["baseline", "maximin"]
NUM_AGENTS = 2
NUM_START_BERRIES = NUM_AGENTS * 2
#MAX_WIDTH = NUM_AGENTS * 2
MAX_WIDTH = NUM_AGENTS * 4 #allotment
MAX_HEIGHT = MAX_WIDTH
MAX_TEST_EPISODES = 2000

def generate_graphs(scenario, num_agents):
    """
    takes raw files and generates graphs displayed in the paper
    processed dfs contain data for each agent at the end of each episode
    e_epochs are run for at most t_max steps; results are normalised by frequency of step
    """
    data_analysis = DataAnalysis(num_agents)
    path_1 = "data/results/current_run/agent_reports_"+scenario+"_"
    path_2 = "data/results/run_3_baseline_maximin/2_agents/capabilities/agent_reports_"+scenario+"_"
    path_3 = "data/results/run_2/run_2_capabilities/with_deon_egal/agent_reports_"+scenario+"_"
    #files = [path_2+"baseline.csv",path_2+"egalitarian.csv",path_2+"maximin.csv",path_2+"utilitarian.csv",path_3+"deon_egalitarian.csv",path_1+"deon_egalitarian_2.csv",path_1+"deon_egalitarian_loss.csv"]
    files = [path_1+"baseline.csv",path_1+"maximin.csv"]
    labels = AGENT_TYPES
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)
    normalised_sum_df_list, agent_end_episode_list = data_analysis.process_agent_dfs(dfs, labels)
    data_analysis.display_graphs(normalised_sum_df_list, agent_end_episode_list, labels)

def log_wandb_agents(model_inst, last_episode, reward_tracker):
    for i, agent in enumerate(model_inst.schedule.agents):
        if agent.agent_type != "berry":
            base_string = agent.agent_type+"_agent_"+str(agent.unique_id)
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

def run_simulation(model_inst, render, log_wandb):
    if log_wandb:
        wandb.init(project="PriENE")
    if render:
        render_inst = RenderPygame(model_inst.max_width, model_inst.max_height)
    while (model_inst.training and model_inst.epsilon > model_inst.min_epsilon) or (not model_inst.training and model_inst.episode <= model_inst.max_episodes):
        model_inst.step()
        if log_wandb:
            reward_tracker = [a.total_episode_reward for a in model_inst.schedule.agents if a.agent_type != "berry"]
            log_wandb_agents(model_inst, model_inst.episode, reward_tracker)
            mean_reward = model_inst.model_episode_reporter["mean_reward"].mean()
            wandb.log({'mean_episode_reward': mean_reward})
        if render:
            render_inst.render_pygame(model_inst)
    num_episodes = model_inst.episode
    return num_episodes

def create_and_run_model(scenario,num_agents,num_start_berries,agent_type,max_width,max_height,max_episodes,training,write_data,write_norms,render,log_wandb):   
    file_string = scenario+"_"+agent_type
    if scenario == "basic":
        model_inst = BasicHarvest(num_agents,num_start_berries,agent_type,max_width,max_height,max_episodes,training,write_data,write_norms,file_string)
    elif scenario == "capabilities":
        model_inst = CapabilitiesHarvest(num_agents,num_start_berries,agent_type,max_width,max_height,max_episodes,training,write_data,write_norms,file_string)
    elif scenario == "allotment":
        model_inst = AllotmentHarvest(num_agents,num_start_berries,agent_type,max_width,max_height,max_episodes,training,write_data,write_norms,file_string)
    else:
        ValueError("Unknown argument: "+scenario)
    run_simulation(model_inst,render,log_wandb)

def run_all(scenario,num_agents,num_start_berries,max_width,max_height,max_episodes,training,write_data,write_norms,render,log_wandb):
    for agent_type in AGENT_TYPES:
        create_and_run_model(scenario,num_agents,num_start_berries,agent_type,max_width,max_height,max_episodes,training,write_data,write_norms,render,log_wandb)

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
        print(f"{data_type} will be written into data/results/current_run.")
    elif write_data == "n":
        write_data = False
    return write_data

#########################################################################################

parser = argparse.ArgumentParser(description="Program options")
parser.add_argument("option", choices=["test", "train", "graphs"],
                    help="Choose the program operation")
parser.add_argument("-l", "--log", type=str, default=None,
                    help="Log wandb (optional)")
args = parser.parse_args()

if args.option not in ["test", "train", "graphs"]:
    print("Please choose 'test', 'train', or 'graphs'.")
elif args.option == "test" or args.option == "train":
    if args.option == "test":
        scenario = input("What type of scenario do you want to run (capabilities, allotment): ")
        while scenario not in ["capabilities", "allotment"]:
            scenario = input("Invalid scenario. Please choose 'capabilities', or 'allotment': ")
    else:
        scenario = "basic"
    #########################################################################################
    agent_type = input(f"What type of agent do you want to implement {AGENT_TYPES}, all): ")
    while agent_type != "all" and agent_type not in AGENT_TYPES:
        agent_type = input(f"Invalid agent type. Please choose {AGENT_TYPES}, or 'all': ")
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
        #max_episodes = get_integer_input("How many episodes do you want to run: ")
        max_episodes = MAX_TEST_EPISODES
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
elif args.option == "graphs":
    scenario = input("What type of scenario do you want to generate graphs for (capabilities, allotment): ")
    while scenario not in ["capabilities", "allotment"]:
        scenario = input("Invalid scenario. Please choose 'capabilities', or 'allotment': ")
    print("Graphs will be saved in data/results/current_run")
    generate_graphs(scenario, NUM_AGENTS)