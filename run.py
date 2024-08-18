from src.scenarios.basic_harvest import BasicHarvest
from src.scenarios.capabilities_harvest import CapabilitiesHarvest
from src.scenarios.allotment_harvest import AllotmentHarvest
from src.data_handling.data_analysis import DataAnalysis
from src.data_handling.render_pygame import RenderPygame
import pandas as pd
import argparse
import wandb
import numpy as np

#AGENT_TYPES = ["baseline", "egalitarian", "maximin", "utilitarian"]
AGENT_TYPES = ["baseline", "maximin"]
#AGENT_TYPES = ["egalitarian", "utilitarian"]
SCENARIO_TYPES = ["capabilities", "allotment"]
NUM_AGENTS_OPTIONS = ["2", "4", "6"]
MAX_EPISODES = 2000
MAX_DAYS = 50
RESULTS_FILEPATH = "data/results/current_run/"
RUN_OPTIONS = ["current_run", "run_1", "run_2", "run_3"]

def generate_graphs(scenario, num_agents):
    """
    takes raw files and generates graphs displayed in the paper
    processed dfs contain data for each agent at the end of each episode
    e_epochs are run for at most t_max steps; results are normalised by frequency of step
    """
    data_analysis = DataAnalysis(num_agents, RESULTS_FILEPATH)
    path = RESULTS_FILEPATH+"agent_reports_"+scenario+"_"
    #files = [path+"baseline.csv",path+"egalitarian.csv",path+"maximin.csv",path+"utilitarian.csv"]
    # files = [path+"baseline.csv",path+"maximin.csv",path+"utilitarian.csv"]
    # dfs = []
    # for file in files:
    #     df = pd.read_csv(file)
    #     dfs.append(df)
    # data_analysis.proccess_and_display_all_data(scenario, dfs, AGENT_TYPES)
    data_analysis._process_norms(scenario, AGENT_TYPES)

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

def create_and_run_model(scenario,run_name,num_agents,num_start_berries,agent_type,max_width,max_height,max_episodes,max_days,training,write_data,write_norms,render,log_wandb):   
    file_string = scenario+"_"+agent_type
    checkpoint_path = "data/model_variables/"+run_name+"/"+str(num_agents)+"_agents/"
    if scenario == "basic":
        model_inst = BasicHarvest(num_agents,num_start_berries,agent_type,max_width,max_height,max_episodes,max_days,training,checkpoint_path,write_data,write_norms,file_string)
    elif scenario == "capabilities":
        model_inst = CapabilitiesHarvest(num_agents,num_start_berries,agent_type,max_width,max_height,max_episodes,max_days,training,checkpoint_path,write_data,write_norms,file_string)
    elif scenario == "allotment":
        model_inst = AllotmentHarvest(num_agents,num_start_berries,agent_type,max_width,max_height,max_episodes,max_days,training,checkpoint_path,write_data,write_norms,file_string)
    else:
        ValueError("Unknown argument: "+scenario)
    run_simulation(model_inst,render,log_wandb)

def run_all(scenario,run_name,num_agents,num_start_berries,max_width,max_height,max_episodes,max_days,training,write_data,write_norms,render,log_wandb):
    for agent_type in AGENT_TYPES:
        create_and_run_model(scenario,run_name,num_agents,num_start_berries,agent_type,max_width,max_height,max_episodes,max_days,training,write_data,write_norms,render,log_wandb)

def get_integer_input(prompt):
    while True:
        try:
            value = int(input(prompt))
            return value
        except ValueError:
            print("Invalid input. Please enter an integer.")

def get_input(input_string, error_string, valid_options):
    variable = input(input_string)
    while variable not in valid_options:
        variable = input(error_string)
    return variable

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
        scenario = get_input(f"What type of scenario do you want to run {SCENARIO_TYPES}: ", f"Invalid scenario. Please choose {SCENARIO_TYPES}: ", SCENARIO_TYPES)#########################################################################################
        run_name = get_input(f"What run do you want to test {RUN_OPTIONS}: ", f"Invalid name of run. Please choose {RUN_OPTIONS}: ", RUN_OPTIONS)
        max_episodes = MAX_EPISODES #get_integer_input("How many episodes do you want to run: ")
        training = False
    else:
        training = True
        scenario = "basic"
        run_name = "current_run"
        max_episodes = 0
    #########################################################################################
    types = AGENT_TYPES + ["all"]
    agent_type = get_input(f"What type of agent do you want to implement {types}: ", f"Invalid agent type. Please choose {types}: ", types)
    #########################################################################################
    num_agents = int(get_input(f"How many agents do you want to implement {NUM_AGENTS_OPTIONS}: ", f"Invalid number of agents. Please choose {NUM_AGENTS_OPTIONS}: ", NUM_AGENTS_OPTIONS))
    #########################################################################################
    write_data = write_data_input("data")
    #########################################################################################
    if args.option == "train":
        print("Model variables will be written into",run_name)
        write_norms = False
        render = False
    else:
        write_norms = write_data_input("norms")
        render = get_input("Do you want to render the simulation? (y, n): ", "Invalid choice. Please choose 'y' or 'n': ", ["y", "n"])
        if render == "y":
            render = True
        elif render == "n":
            render = False
    #########################################################################################
    if args.log is not None:
        log_wandb = True
    else:
        log_wandb = False
    if scenario != "allotment":
        MAX_WIDTH = num_agents * 2
    else:
        MAX_WIDTH = num_agents * 4
    MAX_HEIGHT = num_agents * 2
    NUM_BERRIES = num_agents * 3
    if agent_type == "all":
        run_all(scenario,run_name,num_agents,NUM_BERRIES,MAX_WIDTH,MAX_HEIGHT,max_episodes,MAX_DAYS,training,write_data,write_norms,render,log_wandb)
    else:
        create_and_run_model(scenario,run_name,num_agents,NUM_BERRIES,agent_type,MAX_WIDTH,MAX_HEIGHT,max_episodes,MAX_DAYS,training,write_data,write_norms,render,log_wandb)
#########################################################################################
elif args.option == "graphs":
    scenario = input("What type of scenario do you want to generate graphs for (capabilities, allotment): ")
    while scenario not in ["capabilities", "allotment"]:
        scenario = input("Invalid scenario. Please choose 'capabilities', or 'allotment': ")
    num_agents = int(get_input(f"How many agents do you want to implement {NUM_AGENTS_OPTIONS}: ", f"Invalid number of agents. Please choose {NUM_AGENTS_OPTIONS}: ", NUM_AGENTS_OPTIONS))
    print("Graphs will be saved in data/results/current_run")
    generate_graphs(scenario,num_agents)