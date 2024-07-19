from src.scenarios.basic_harvest import BasicHarvest
from src.scenarios.capabilities_harvest import CapabilitiesHarvest
from src.scenarios.allotment_harvest import AllotmentHarvest
from src.data_analysis import DataAnalysis
import pandas as pd
import argparse
import wandb
import pygame
import numpy as np

AGENT_TYPES = ["baseline", "egalitarian", "maximin", "rawlsian", "utilitarian"]
NUM_AGENTS = 4
#render globals
MAX_WIDTH = NUM_AGENTS * 2
MAX_HEIGHT = MAX_WIDTH
BLOCK_SIZE = 640/MAX_WIDTH
COLOUR_MAP = {
    "red": (255, 0, 0),
    "blue": (0, 0, 255),
    "green": (0, 255, 0),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "yellow": (255, 255, 0),
    "purple": (255, 0, 255),
}

OBJ_COLOURS = {
    "None": "red",
    "0": "purple",
    "1": "blue"
}

AGENT_COLOURS = {
    "baseline": "red",
    "egalitarian": "yellow",
    "maximin": "blue",
    "utilitarian": "green",
    "berry": "purple"
}


def draw_agent(screen, colour, x, y):
    pygame.draw.rect(screen, colour, (x+BLOCK_SIZE/4, y, BLOCK_SIZE/2, BLOCK_SIZE/4))
    pygame.draw.circle(screen, "blue", (x+BLOCK_SIZE/2-BLOCK_SIZE/8,y+BLOCK_SIZE/8), (BLOCK_SIZE/8))
    pygame.draw.circle(screen, "blue", (x+BLOCK_SIZE/2+BLOCK_SIZE/8,y+BLOCK_SIZE/8), (BLOCK_SIZE/8))
    pygame.draw.rect(screen, colour, (x, y+BLOCK_SIZE/4, BLOCK_SIZE, BLOCK_SIZE/4))
    pygame.draw.rect(screen, colour, (x, y+BLOCK_SIZE/2, BLOCK_SIZE/4, BLOCK_SIZE/2))
    pygame.draw.rect(screen, colour, (x+BLOCK_SIZE*3/4, y+BLOCK_SIZE/2, BLOCK_SIZE/4, BLOCK_SIZE/2))

def draw_berry(screen, colour, x, y):
    pygame.draw.circle(screen, colour, (x+BLOCK_SIZE/2,y+BLOCK_SIZE/2), (BLOCK_SIZE/3))
    pygame.draw.arc(screen, "green", (x,y, BLOCK_SIZE/2, BLOCK_SIZE/2), 0/57, 90/57)

def generate_graphs(scenario):
    """
    takes raw files and generates graphs displayed in the paper
    processed dfs contain data for each agent at the end of each episode
    e_epochs are run for at most t_max steps; results are normalised by frequency of step
    """
    data_analysis = DataAnalysis()
    path = "data/"+scenario+"/"
    files = [path+"baseline.csv",path+"egalitarian.csv",path+"maximin.csv",path+"rawlsian.csv",path+"utilitarian.csv"]
    labels = AGENT_TYPES
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

def run_simulation(model_inst, render, log_wandb):
    if log_wandb:
        wandb.init(project="PriENE")
    if render:
        screen = init_pygame()
    while (model_inst.training and model_inst.epsilon > model_inst.min_epsilon) or (not model_inst.training and model_inst.episode <= model_inst.max_episodes):
        model_inst.step()
        if log_wandb:
            reward_tracker = [a.total_episode_reward for a in model_inst.schedule.agents if a.type != "berry"]
            log_wandb_agents(model_inst, model_inst.episode, reward_tracker)
            mean_reward = model_inst.model_reporter["mean_reward"].mean()
            wandb.log({'mean_reward_test': mean_reward})
        if render:
            screen = render_pygame(screen, model_inst)
    num_episodes = model_inst.episode
    return num_episodes

def init_pygame():
    # create window
    pygame.init()
    # setup screen
    width, height = MAX_WIDTH * BLOCK_SIZE, MAX_HEIGHT * BLOCK_SIZE
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('Harvest Model')
    # setup timer
    pygame.time.Clock()
    return screen

def render_pygame(screen, model_inst):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            break
    screen.fill(COLOUR_MAP["black"])
    for a in model_inst.schedule.agents:
        # if a has attribute deceased
        if hasattr(a, "off_grid") and a.off_grid:
            continue
        x = a.pos[0] * BLOCK_SIZE
        y = a.pos[1] * BLOCK_SIZE
        if a.agent_type == "berry":
            colour = COLOUR_MAP[AGENT_COLOURS[str(a.agent_type)]]
            draw_berry(screen, colour, x, y)
        else:
            colour = COLOUR_MAP[AGENT_COLOURS[str(a.agent_type)]]
            draw_agent(screen, colour, x, y)
    pygame.display.flip()
    return screen

def create_and_run_model(scenario,agent_type,max_width,max_height,max_episodes,training,write_data,write_norms,render,log_wandb):   
    file_string = scenario+"_"+agent_type
    if scenario == "basic":
        model_inst = BasicHarvest(NUM_AGENTS,agent_type,max_width,max_height,max_episodes,training,write_data,write_norms,file_string)
    elif scenario == "capabilities":
        model_inst = CapabilitiesHarvest(NUM_AGENTS,agent_type,max_width,max_height,max_episodes,training,write_data,write_norms,file_string)
    elif scenario == "allotment":
        model_inst = AllotmentHarvest(NUM_AGENTS,agent_type,max_width,max_height,max_episodes,training,write_data,write_norms,file_string)
    else:
        ValueError("Unknown argument: "+scenario)
    run_simulation(model_inst,render,log_wandb)

def run_all(scenario,max_width,max_height,max_episodes,training,write_data,write_norms,render,log_wandb):
    for agent_type in AGENT_TYPES:
        create_and_run_model(scenario,agent_type,max_width,max_height,max_episodes,training,write_data,write_norms,render,log_wandb)

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

def get_write_data_input(data_type):
    write_data = input(f"Do you want to write {data_type} to file? (y, n): ")
    while write_data not in ["y", "n"]:
        write_data = input("Invalid choice. Please choose 'y' or 'n': ")
    if write_data == "y":
        write_data = True
        print(f"{data_type} will be written into data/results.")
    elif write_data == "n":
        write_data = False
    return write_data

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
    write_data = get_write_data_input("data")
    #########################################################################################
    write_norms = get_write_data_input("norms")
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
        run_all(scenario,MAX_WIDTH,MAX_HEIGHT,max_episodes,training,write_data,write_norms,render,log_wandb)
    else:
        create_and_run_model(scenario,MAX_WIDTH,MAX_HEIGHT,agent_type,max_episodes,training,write_data,write_norms,render,log_wandb)
#########################################################################################
elif args.option == "generate_graphs":
    scenario = input("What type of scenario do you want to generate graphs for (capabilities, allotment): ")
    while scenario not in ["capabilities", "allotment"]:
        scenario = input("Invalid scenario. Please choose 'capabilities', or 'allotment': ")
    print("Graphs will be saved in data/results")
    generate_graphs(scenario)