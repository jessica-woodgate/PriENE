from src.scenarios.basic_harvest import BasicHarvest
from src.scenarios.capabilities_harvest import CapabilitiesHarvest
from src.scenarios.allotment_harvest import AllotmentHarvest
from src.data_handling.data_analysis import DataAnalysis
from src.data_handling.render_pygame import RenderPygame
import pandas as pd
import argparse
import numpy as np

AGENT_TYPES = ["baseline", "maximin"]
SCENARIO_TYPES = ["capabilities", "allotment"]
NUM_AGENTS_OPTIONS = ["2", "4"]
MAX_EPISODES = 2000
MAX_DAYS = 50
RUN_OPTIONS = ["current_run", "pretrained"]

def generate_graphs(scenario, run_name, num_agents):
    """
    takes raw files and generates graphs displayed in the paper
    processed dfs contain data for each agent at the end of each episode
    e_epochs are run for at most t_max steps; results are normalised by frequency of step
    """
    writing_filepath = "data/results/current_run/"
    data_analysis = DataAnalysis(num_agents, writing_filepath)
    reading_filepath = "data/results/"+run_name+"/"+str(num_agents)+"_agents/"+scenario+"/agent_reports_"+scenario+"_"
    norms_filepath = "data/results/"+run_name+"/"+str(num_agents)+"_agents/"+scenario+"/"+scenario
    files = [reading_filepath+"baseline.csv",reading_filepath+"maximin.csv"]
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)
    data_analysis.proccess_and_display_all_data(dfs, AGENT_TYPES, scenario, norms_filepath)

def run_simulation(model_inst, render):
    if render:
        render_inst = RenderPygame(model_inst.max_width, model_inst.max_height)
    while (model_inst.training and model_inst.epsilon > model_inst.min_epsilon) or (not model_inst.training and model_inst.episode <= model_inst.max_episodes):
        model_inst.step()
        if render:
            render_inst.render_pygame(model_inst)
    num_episodes = model_inst.episode
    return num_episodes

def create_and_run_model(scenario,run_name,num_agents,num_start_berries,agent_type,max_width,max_height,max_episodes,max_days,training,write_data,write_norms,render):   
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
    run_simulation(model_inst,render)

def run_all(scenario,run_name,num_agents,num_start_berries,max_width,max_height,max_episodes,max_days,training,write_data,write_norms,render):
    for agent_type in AGENT_TYPES:
        create_and_run_model(scenario,run_name,num_agents,num_start_berries,agent_type,max_width,max_height,max_episodes,max_days,training,write_data,write_norms,render)

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
args = parser.parse_args()

parser = argparse.ArgumentParser(description="Program options")
parser.add_argument("option", choices=["test", "train", "graphs"],
                    help="Choose the program operation")
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
    if scenario != "allotment":
        MAX_WIDTH = num_agents * 2
    else:
        MAX_WIDTH = num_agents * 4
    MAX_HEIGHT = num_agents * 2
    NUM_BERRIES = num_agents * 3
    if agent_type == "all":
        run_all(scenario,run_name,num_agents,NUM_BERRIES,MAX_WIDTH,MAX_HEIGHT,max_episodes,MAX_DAYS,training,write_data,write_norms,render)
    else:
        create_and_run_model(scenario,run_name,num_agents,NUM_BERRIES,agent_type,MAX_WIDTH,MAX_HEIGHT,max_episodes,MAX_DAYS,training,write_data,write_norms,render)
#########################################################################################
elif args.option == "graphs":
    run_name = get_input(f"What run do you want to generate graphs for {RUN_OPTIONS}: ", f"Invalid name of run. Please choose {RUN_OPTIONS}: ", RUN_OPTIONS)
    scenario = get_input("What type of scenario do you want to generate graphs for (capabilities, allotment): ", "Invalid scenario. Please choose 'capabilities', or 'allotment': ", ["capabilities", "allotment"])
    num_agents = int(get_input(f"How many agents do you want to implement {NUM_AGENTS_OPTIONS}: ", f"Invalid number of agents. Please choose {NUM_AGENTS_OPTIONS}: ", NUM_AGENTS_OPTIONS))
    print("Graphs will be saved in data/results/current_run")
    generate_graphs(scenario,run_name,num_agents)