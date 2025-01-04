from src.scenarios.basic_harvest import BasicHarvest
from src.scenarios.colours_harvest import ColoursHarvest
from src.scenarios.allotment_harvest import AllotmentHarvest
from src.data_handling.data_analysis import DataAnalysis
from src.data_handling.render_pygame import RenderPygame
import pandas as pd
import argparse
import wandb
import numpy as np
import re

#all_principles = average
#all_principles_3 = majoritarian
#all_principles_5 = veto (do_no_harm in model variables)
#all_principles_6 = optimist

#AGENT_TYPES = ["baseline", "egalitarian", "maximin", "utilitarian", "average", "majoritarian", "veto", "optimist"]
AGENT_TYPES = ["average", "majoritarian", "veto", "optimist"]
SCENARIO_TYPES = ["colours", "allotment"]
NUM_AGENTS_OPTIONS = ["2", "4", "6"]
MAX_EPISODES = 1000
RUN_OPTIONS = ["current_run", "50_days", "200_days"]

def generate_graphs(scenario, run_name, num_agents):
    """
    takes raw files and generates graphs displayed in the paper
    processed dfs contain data for each agent at the end of each episode
    e_epochs are run for at most t_max steps; results are normalised by frequency of step
    """
    writing_filepath = "data/results/current_run/"
    data_analysis = DataAnalysis(num_agents, writing_filepath)
    if run_name == "current_run":
        reading_filepath = "data/results/"+run_name+"/agent_reports_"+scenario+"_"
    else:
        reading_filepath = "data/results/"+run_name+"/"+scenario+"/agent_reports_"+scenario+"_"
    #files = [reading_filepath+"baseline.csv",reading_filepath+"egalitarian.csv",reading_filepath+"maximin.csv",reading_filepath+"utilitarian.csv",reading_filepath+"average.csv"]
    files = [reading_filepath+"average.csv",reading_filepath+"majoritarian.csv",reading_filepath+"optimist.csv",reading_filepath+"veto.csv"]
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)
    data_analysis.proccess_and_display_data(dfs, AGENT_TYPES)

def log_wandb_agents(model_inst, last_episode, reward_tracker):
    for i, agent in enumerate(model_inst.schedule.agents):
        if agent.agent_type != "berry":
            base_string = agent.agent_type+"_agent_"+str(agent.unique_id)
            if last_episode != model_inst.episode:
                string = base_string+"_total_episode_reward"
                reward = reward_tracker[i]
                wandb.log({string: reward})
            string = base_string+"_reward"
            if "multiobjective" in agent.agent_type:
                wandb.log({string: sum(agent.current_reward)})
            else:
                wandb.log({string: agent.current_reward})
            mean_loss = (np.mean(agent.losses) if model_inst.training else 0)
            string = base_string+"_mean_loss"
            wandb.log({string: mean_loss})
            string = base_string+"_epsilon"
            wandb.log({string: agent.epsilon})

def run_simulation(model_inst, render, log_wandb, wandb_project):
    if log_wandb:
        wandb.init(project=wandb_project)
    if render:
        render_inst = RenderPygame(model_inst.max_width, model_inst.max_height)
    while (model_inst.training and model_inst.epsilon > model_inst.min_epsilon) or (not model_inst.training and model_inst.episode <= model_inst.max_episodes):
        last_episode = model_inst.episode
        model_inst.step()
        if log_wandb:
            reward_tracker = [a.total_episode_reward for a in model_inst.schedule.agents if a.agent_type != "berry"]
            log_wandb_agents(model_inst, last_episode, reward_tracker)
            mean_reward = model_inst.model_episode_reporter["mean_reward"].mean()
            wandb.log({'mean_episode_reward': mean_reward})
        if render:
            render_inst.render_pygame(model_inst)
    num_episodes = model_inst.episode
    return num_episodes

def create_and_run_model(scenario,run_name,num_agents,num_start_berries,num_allotments,agent_type,max_width,max_height,max_episodes,max_days,training,write_data,write_norms,render,log_wandb,wandb_project=None):   
    file_string = scenario+"_"+agent_type
    checkpoint_path = "data/model_variables/"+run_name+"/"+str(num_agents)+"_agents/"
    if scenario == "basic":
        model_inst = BasicHarvest(num_agents,num_start_berries,agent_type,max_width,max_height,max_episodes,max_days,training,checkpoint_path,write_data,write_norms,file_string)
    elif scenario == "colours":
        model_inst = ColoursHarvest(num_agents,num_start_berries,agent_type,max_width,max_height,max_episodes,max_days,training,checkpoint_path,write_data,write_norms,file_string)
    elif scenario == "allotment":
        model_inst = AllotmentHarvest(num_agents,num_start_berries,num_allotments,agent_type,max_width,max_height,max_episodes,max_days,training,checkpoint_path,write_data,write_norms,file_string)
    else:
        ValueError("Unknown argument: "+scenario)
    run_simulation(model_inst,render,log_wandb,wandb_project)

def run_all(scenario,run_name,num_agents,num_start_berries,num_allotments,max_width,max_height,max_episodes,max_days,training,write_data,write_norms,render,log_wandb,wandb_project=None):
    for agent_type in AGENT_TYPES:
        create_and_run_model(scenario,run_name,num_agents,num_start_berries,num_allotments,agent_type,max_width,max_height,max_episodes,max_days,training,write_data,write_norms,render,log_wandb,wandb_project)

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

def get_bool(input):
    if input == "y":
        return True
    elif input == "n":
        return False

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

def extract_number(string):
    match = re.match(r"(\d+)_days", string)
    if match:
        return int(match.group(1))
    return None

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
        scenario = get_input(f"What type of scenario do you want to run {SCENARIO_TYPES}: ", f"Invalid scenario. Please choose {SCENARIO_TYPES}: ", SCENARIO_TYPES)
        run_name = get_input(f"What run do you want to test {RUN_OPTIONS}: ", f"Invalid name of run. Please choose {RUN_OPTIONS}: ", RUN_OPTIONS)
        max_episodes = MAX_EPISODES
        training = False
        if run_name != "current_run":
            max_days = extract_number(run_name)
        else:
            max_days = get_integer_input("How many days in each episode: ")
    else:
        training = True
        scenario = "basic"
        run_name = "current_run"
        max_episodes = 0
        max_days = get_integer_input("How many days in each episode: ")
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
        render = get_bool(render)
     #########################################################################################
    if args.log is not None:
        log_wandb = True
        wandb_project = args.log
    else:
        log_wandb = False
        wandb_project = None
    #########################################################################################
    if scenario != "allotment":
        MAX_WIDTH = num_agents * 2
        num_allotments = 1
    else:
        MAX_WIDTH = num_agents * 4
        num_allotments = 2
    MAX_HEIGHT = num_agents * 2
    NUM_BERRIES = num_agents * 3
    if agent_type == "all":
        run_all(scenario,run_name,num_agents,NUM_BERRIES,num_allotments,MAX_WIDTH,MAX_HEIGHT,max_episodes,max_days,training,write_data,write_norms,render,log_wandb,wandb_project)
    else:
        create_and_run_model(scenario,run_name,num_agents,NUM_BERRIES,num_allotments,agent_type,MAX_WIDTH,MAX_HEIGHT,max_episodes,max_days,training,write_data,write_norms,render,log_wandb,wandb_project)
#########################################################################################
elif args.option == "graphs":
    graph_runs = ["current_run", "50_days", "200_days"]
    run_name = get_input(f"What run do you want to generate graphs for (select 200_days to reproduce graphs in the paper) {graph_runs}: ", f"Invalid name of run. Please choose {graph_runs}: ", graph_runs)
    scenario = get_input("What type of scenario do you want to generate graphs for (colours, allotment): ", "Invalid scenario. Please choose 'colours', or 'allotment': ", ["colours", "allotment"])
    num_agents = 4
    print("Graphs will be saved in data/results/current_run")
    generate_graphs(scenario,run_name,num_agents)