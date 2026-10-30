from src.scenarios.basic_harvest import BasicHarvest
from src.scenarios.colours_harvest import ColoursHarvest
from src.scenarios.allotment_harvest import AllotmentHarvest
from src.scenarios.capabilities_harvest import CapabilitiesHarvest
from src.data_handling.data_analysis import DataAnalysis
from src.data_handling.render_pygame import RenderPygame
from src.agent.harvest_agent import NORMS_MODULE_TYPES
from src.checkpoints import RUN_OPTIONS, DEFAULT_CHECKPOINT
from src.checkpoints import training_checkpoint_name, training_checkpoint_path, prepare_training_checkpoints
from src.checkpoints import society_checkpoint_path, loading_checkpoint_path, record_agent_types
from src.checkpoints import resolve_checkpoint, trained_checkpoints, checkpoint_max_days, days_in_name
import pandas as pd
import argparse
import wandb
import time

PRINCIPLES = ["baseline", "utilitarian", "maximin", "egalitarian"]
AGGREGATIONS = ["average", "majoritarian", "optimist", "veto"]
AGENT_TYPES = PRINCIPLES + AGGREGATIONS
SCENARIO_TYPES = ["basic", "colours", "capabilities", "allotment"]
NUM_AGENTS_OPTIONS = ["2", "4", "6", "20", "50", "100"]
MAX_EPISODES = 1000
SOCIETY_MIXES = ["homogeneous", "heterogeneous"]
DEFAULT_SEED = 42
DEFAULT_SEASON_LENGTH = 25
#options that aren't flags, and why
REMOVED_OPTIONS = {
    "--initial-healths": "unequal starting health was removed; every agent starts each episode with the default health",
    "--season-quality": "it is now part of seasons, chosen at the prompt, where each season is randomly normal, good or bad",
    "--season-swaps": "random swaps of berry allocations between groups were removed; seasons (chosen at the prompt) give each season a random quality instead",
    "--seasons": "seasons are chosen at the prompt",
    "--season-length": "the season length is chosen at the prompt",
}

def generate_paper_graphs(scenario):
    """
    takes data files and generates graphs displayed in the paper
    agent_end_episode dfs contain data for each agent at the end of each episode
    end_episode_totals contain the totals across the society at the end of each episode
    """
    run_name = "200_days"
    num_agents = 4
    writing_filepath = "data/results/current_run/"
    data_analysis = DataAnalysis(num_agents, writing_filepath)
    agents_filepath = "data/results/"+run_name+"/"+str(num_agents)+"_agents/"+scenario+"/agent_end_episode_df_"
    episodes_filepath = "data/results/"+run_name+"/"+str(num_agents)+"_agents/"+scenario+"/end_episode_totals_"
    agent_end_episode_files = [agents_filepath+"baseline.csv",agents_filepath+"utilitarian.csv",agents_filepath+"maximin.csv",agents_filepath+"egalitarian.csv",agents_filepath+"average.csv",agents_filepath+"majoritarian.csv",agents_filepath+"optimist.csv",agents_filepath+"veto.csv"]
    end_episode_totals_files = [episodes_filepath+"baseline.csv",episodes_filepath+"utilitarian.csv",episodes_filepath+"maximin.csv",episodes_filepath+"egalitarian.csv",episodes_filepath+"average.csv",episodes_filepath+"majoritarian.csv",episodes_filepath+"optimist.csv",episodes_filepath+"veto.csv"]
    agent_end_episode_dfs = []
    end_episode_totals_dfs = []
    for i, file in enumerate(agent_end_episode_files):
        df = pd.read_csv(file)
        agent_end_episode_dfs.append(df)
        df = pd.read_csv(end_episode_totals_files[i])
        end_episode_totals_dfs.append(df)
    data_analysis.proccess_and_display_data(agent_end_episode_dfs, PRINCIPLES, AGGREGATIONS, scenario, end_episode_totals_dfs=end_episode_totals_dfs, write=True, get_normalised=False, process_norms=False)

def generate_graphs(scenario, run_name, num_agents, seasons=False, allotment_regrowth_delays=None, mixed_allotments="split", society_mix="homogeneous", process_norms=True):
    """
    takes raw files and generates graphs
    processed dfs contain data for each agent at the end of each episode
    seasons, allotment regrowth delay and mixed society runs are read from the same scenario folder, with "_seasons", "_regrow" and the
    delays (e.g. "_regrow0-1"), and/or "_mixed" ("_mixed_separate" for separate allotments) after the scenario in each filename;
    a mixed society has no baseline-only run, so its graphs compare the other agent types
    process_norms -- whether the run wrote rule-based norms (<file>_emerged_norms.json) to analyse
    """
    writing_filepath = "data/results/current_run/"
    file_scenario = scenario_file_string(scenario, seasons, allotment_regrowth_delays, mixed_allotments, society_mix)
    if run_name == "current_run":
        norms_filepath = "data/results/"+run_name+"/"+file_scenario+"_"
    else:
        norms_filepath = "data/results/"+run_name+"/"+str(num_agents)+"_agents/"+scenario+"/norms/"+file_scenario+"_"
    data_analysis = DataAnalysis(num_agents, writing_filepath)
    if run_name == "current_run":
        reading_filepath = "data/results/"+run_name+"/agent_reports_"+file_scenario+"_"
    else:
        reading_filepath = "data/results/"+run_name+"/"+str(num_agents)+"_agents/"+scenario+"/agent_reports_"+file_scenario+"_"
    #the agent types run, in the same order as the labels (principles + aggregations) the analysis gives them
    agent_types = agent_types_to_run(society_mix)
    principles = [agent_type for agent_type in PRINCIPLES if agent_type in agent_types]
    files = [reading_filepath+agent_type+".csv" for agent_type in agent_types]
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)
    data_analysis.proccess_and_display_data(dfs, principles, AGGREGATIONS, scenario, norms_filepath, write=True, get_normalised=False, process_norms=process_norms)

def run_simulation(model_inst, render, log_wandb, wandb_project):
    if log_wandb:
        wandb.init(project=wandb_project)
    if render:
        render_inst = RenderPygame(model_inst.max_width, model_inst.max_height)
    #training runs until exploration has decayed to its floor; testing runs max_episodes
    while (model_inst.training and round(model_inst.epsilon, 5) > model_inst.min_epsilon) or (not model_inst.training and model_inst.episode <= model_inst.max_episodes):
        model_inst.step()
        if log_wandb:
            mean_reward = model_inst._mean_reward()
            wandb.log({'mean_episode_reward': mean_reward})
        if render:
            if not render_inst.render_pygame(model_inst):
                break
    model_inst.finish_episode(collect_data=False)
    num_episodes = model_inst.episode
    print(f"simulation finished time: {time.time()}")
    return num_episodes

def scenario_file_string(scenario, seasons, allotment_regrowth_delays=None, mixed_allotments="split", society_mix="homogeneous"):
    """
    scenario part of output filenames; options that are on are added after the scenario so runs don't overwrite each other
    """
    file_scenario = scenario
    if seasons:
        file_scenario += "_seasons"
    if allotment_regrowth_delays is not None:
        file_scenario += "_regrow" + "-".join(str(delay) for delay in allotment_regrowth_delays)
    if society_mix != "homogeneous":
        file_scenario += "_mixed"
        if scenario == "allotment" and mixed_allotments == "separate":
            file_scenario += "_separate"
    return file_scenario

def get_checkpoint_input():
    """
    Ask which checkpoint to test: one of RUN_OPTIONS, a trained checkpoint's name, or a path to a folder holding <n>_agents folders;
    pressing enter uses DEFAULT_CHECKPOINT, the agents trained in basic for the paper. Returns the folder and the answer given
    """
    trained = trained_checkpoints()
    trained_string = f", a trained checkpoint {trained}" if trained else ""
    while True:
        value = input(f"Which checkpoint do you want to test? Choose a run {RUN_OPTIONS}{trained_string} or a path; press enter for {DEFAULT_CHECKPOINT} (the agents trained in basic for the paper): ").strip()
        checkpoint_root = resolve_checkpoint(value)
        if checkpoint_root is not None:
            return checkpoint_root, value or DEFAULT_CHECKPOINT
        print(f"Invalid checkpoint. {value} is not a run, a trained checkpoint or a folder.")

def get_max_days_input(checkpoint_name, checkpoint_path):
    """
    Days in each episode for testing a checkpoint: as it was trained if its name or settings say, otherwise ask
    """
    max_days = checkpoint_max_days(checkpoint_name, checkpoint_path)
    if max_days is None:
        return get_integer_input("How many days in each episode: ")
    if days_in_name(checkpoint_name) is None:
        print(f"Testing with {max_days} days in each episode, as the checkpoint was trained.")
    return max_days

def create_and_run_model(scenario,society_mix,checkpoint_path,num_agents,num_start_berries,num_allocations,agent_type,max_width,max_height,max_episodes,max_days,training,write_data,track_norms,render,log_wandb,wandb_project=None,seed=None,norms_module_type="rule_based",seasons=False,season_length=DEFAULT_SEASON_LENGTH,allotment_regrowth_delays=None,mixed_allotments="split"):
    """
    checkpoint_path is the folder holding the checkpoints for this number of agents: training saves there, testing loads from it
    (in a mixed society, from the folder of agent_type trained with baseline agents; see society_checkpoint_path)
    """
    #training outputs are named apart from testing outputs, so testing the same scenario doesn't clash with them
    file_string = ("train_" if training else "")+scenario_file_string(scenario, seasons, allotment_regrowth_delays, mixed_allotments, society_mix)+"_"+agent_type
    if training:
        checkpoint_path = society_checkpoint_path(checkpoint_path, society_mix, agent_type)
    else:
        checkpoint_path = loading_checkpoint_path(checkpoint_path, society_mix, agent_type)
    if scenario == "basic":
        model_inst = BasicHarvest(society_mix,num_agents,num_start_berries,agent_type,max_width,max_height,max_episodes,max_days,training,checkpoint_path,write_data,track_norms,file_string,seed,norms_module_type,seasons,season_length,allotment_regrowth_delays)
    elif scenario == "colours":
        model_inst = ColoursHarvest(society_mix,num_agents,num_start_berries,agent_type,max_width,max_height,max_episodes,max_days,training,checkpoint_path,write_data,track_norms,file_string,seed,norms_module_type,seasons,season_length,allotment_regrowth_delays)
    elif scenario == "allotment":
        model_inst = AllotmentHarvest(society_mix,num_agents,num_start_berries,num_allocations,agent_type,max_width,max_height,max_episodes,max_days,training,checkpoint_path,write_data,track_norms,file_string,seed,norms_module_type,seasons,season_length,allotment_regrowth_delays,mixed_allotments)
    elif scenario == "capabilities":
        model_inst = CapabilitiesHarvest(society_mix,num_agents,num_start_berries,num_allocations,agent_type,max_width,max_height,max_episodes,max_days,training,checkpoint_path,write_data,track_norms,file_string,seed,norms_module_type,seasons,season_length,allotment_regrowth_delays)
    else:
        raise ValueError("Unknown argument: "+scenario)
    if training and society_mix != "homogeneous":
        record_agent_types(checkpoint_path, model_inst.get_living_agents())
    run_simulation(model_inst,render,log_wandb,wandb_project)

def agent_types_to_run(society_mix):
    #in a mixed society half the agents are baseline, so baseline alone isn't run
    return AGENT_TYPES[1:] if society_mix == "heterogeneous" else AGENT_TYPES

def run_all(scenario,society_mix,checkpoint_path,num_agents,num_start_berries,num_allocations,max_width,max_height,max_episodes,max_days,training,write_data,track_norms,render,log_wandb,wandb_project=None,seed=None,norms_module_type="rule_based",seasons=False,season_length=DEFAULT_SEASON_LENGTH,allotment_regrowth_delays=None,mixed_allotments="split"):
    for agent_type in agent_types_to_run(society_mix):
        create_and_run_model(scenario,society_mix,checkpoint_path,num_agents,num_start_berries,num_allocations,agent_type,max_width,max_height,max_episodes,max_days,training,write_data,track_norms,render,log_wandb,wandb_project,seed,norms_module_type,seasons,season_length,allotment_regrowth_delays,mixed_allotments)

def get_mixed_allotments_input(num_allotments):
    """
    Ask how the two agent types of a mixed society share the allotments; separate needs an even number of allotments
    """
    if num_allotments % 2 != 0:
        print(f"With {num_allotments} allotments the agent types are split within the allotments (separate allotments need an even number).")
        return "split"
    return get_input("How should the two agent types share the allotments? split: each allotment has agents of both types; separate: each type has its own allotments ['split', 'separate']: ",
                     "Invalid choice. Please choose ['split', 'separate']: ", ["split", "separate"])

def get_season_length_input():
    """
    Ask for the number of days in each season; pressing enter uses DEFAULT_SEASON_LENGTH
    """
    while True:
        value = input(f"How many days in each season? Press enter for the default {DEFAULT_SEASON_LENGTH}: ").strip()
        if value == "":
            return DEFAULT_SEASON_LENGTH
        try:
            season_length = int(value)
        except ValueError:
            print("Invalid input. Please enter an integer.")
            continue
        if season_length < 1:
            print("Invalid choice. Must be at least 1.")
            continue
        return season_length

class RemovedOption(argparse.Action):
    """
    Fails with a message saying the option isn't a flag, whatever value it is given
    """
    def __call__(self, parser, namespace, values, option_string=None):
        parser.error(f"{option_string} is not an option: {REMOVED_OPTIONS[option_string]}")

def get_integer_input(prompt, error_string=None, max_value=None, min_value=None):
    while True:
        try:
            value = int(input(prompt))
            if max_value != None and value > max_value:
                print(error_string)
                continue
            if min_value != None and value < min_value:
                print(error_string)
                continue
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

#########################################################################################

parser = argparse.ArgumentParser(description="Program options")
parser.add_argument("option", choices=["test", "train", "graphs"],
                    help="Choose the program operation")
parser.add_argument("-s", "--seed", type=int, default=DEFAULT_SEED,
                    help="Random seed for reproducible runs (optional)")
parser.add_argument("-l", "--log", type=str, default=None,
                    help="Log wandb (optional)")
for removed_option in REMOVED_OPTIONS:
    parser.add_argument(removed_option, action=RemovedOption, nargs="*", help=argparse.SUPPRESS)
args = parser.parse_args()

if args.option not in ["test", "train", "graphs"]:
    print("Please choose 'test', 'train', or 'graphs'.")
elif args.option == "test" or args.option == "train":
    #the mode decides learning, exploration, saving and outputs; any scenario, with or without seasons, runs in either mode
    training = args.option == "train"
    mode = "train" if training else "test"
    scenario = get_input(f"What type of scenario do you want to {mode} in {SCENARIO_TYPES}: ", f"Invalid scenario. Please choose {SCENARIO_TYPES}: ", SCENARIO_TYPES)
    seasons = get_bool(get_input("Do you want seasons, where each season is randomly normal, good (one more berry per agent in each group) or bad (one fewer)? (y, n): ", "Invalid choice. Please choose 'y' or 'n': ", ["y", "n"]))
    season_length = get_season_length_input() if seasons else DEFAULT_SEASON_LENGTH
    if training:
        max_episodes = 0
        max_days = get_integer_input("How many days in each episode: ")
    else:
        checkpoint_root, checkpoint_name = get_checkpoint_input()
        max_episodes = MAX_EPISODES
    #in a mixed society half the agents are baseline, and train or test alongside the other half
    society_mix = get_input(f"What mix of society do you want to {mode} {SOCIETY_MIXES}: ", f"Invalid society mix. Please choose {SOCIETY_MIXES}: ", SOCIETY_MIXES)
    #########################################################################################
    types = AGENT_TYPES + ["all"]
    agent_type = get_input(f"What type of agent do you want to implement {types}: ", f"Invalid agent type. Please choose {types}: ", types)
    #########################################################################################
    num_agents = int(get_input(f"How many agents do you want to implement (select 4 to replicate results in the paper) {NUM_AGENTS_OPTIONS}: ", f"Invalid number of agents. Please choose {NUM_AGENTS_OPTIONS}: ", NUM_AGENTS_OPTIONS))
    #########################################################################################
    #allotment uses its default regrowth delays (allotment 1 regrows slower than the others)
    allotment_regrowth_delays = None
    mixed_allotments = "split"
    if scenario == "allotment":
        MAX_WIDTH = num_agents * 4
        num_allocations = get_integer_input(f"How many allotments do you want? (select 2 to replicate results in the paper) (must be more than 0 and less than {num_agents+1}): ", f"Invalid choice. Must be a number between 1 and {num_agents}", num_agents, 1)
        if society_mix == "heterogeneous":
            mixed_allotments = get_mixed_allotments_input(num_allocations)
    elif scenario == "capabilities":
        MAX_WIDTH = num_agents * 2
        num_allocations = get_integer_input(f"How many capabilities do you want? (must be more than 0 and less than {num_agents+1}) (select 2 to replicate results in the paper): ", f"Invalid choice. Must be a number between 1 and {num_agents}", num_agents, 1)
    else:
        MAX_WIDTH = num_agents * 2
        num_allocations = 1
    MAX_HEIGHT = num_agents * 2
    NUM_BERRIES = num_agents * 3
    #########################################################################################
    write_data = write_data_input("data")
    #########################################################################################
    if training:
        #the delays allotments actually regrow with, so the name says which were used
        delays = allotment_regrowth_delays
        if scenario == "allotment" and delays is None:
            delays = AllotmentHarvest.default_regrowth_delays(num_allocations)
        checkpoint_path = training_checkpoint_path(training_checkpoint_name(scenario, seasons, season_length, num_allocations, delays, max_days, args.seed, society_mix, mixed_allotments), num_agents)
    else:
        checkpoint_path = checkpoint_root+str(num_agents)+"_agents/"
        max_days = get_max_days_input(checkpoint_name, checkpoint_path)
    #########################################################################################
    norms_module_type = NORMS_MODULE_TYPES[0]
    if training:
        prepare_training_checkpoints(checkpoint_path, agent_types_to_run(society_mix) if agent_type == "all" else [agent_type],
                                     {"scenario": scenario, "seasons": seasons, "season_length": season_length if seasons else None,
                                      "num_allocations": num_allocations, "allotment_regrowth_delays": allotment_regrowth_delays,
                                      "max_days": max_days, "seed": args.seed, "num_agents": num_agents, "society_mix": society_mix,
                                      "mixed_allotments": mixed_allotments if scenario == "allotment" and society_mix != "homogeneous" else None},
                                     society_mix)
        print("Model variables will be written into",checkpoint_path)
        track_norms = False
        render = False
    else:
        track_norms = write_data_input("norms")
        if track_norms:
            norms_module_type = get_input(f"Which norms module do you want to use {NORMS_MODULE_TYPES}: ", f"Invalid norms module. Please choose {NORMS_MODULE_TYPES}: ", NORMS_MODULE_TYPES)
        render = get_input("Do you want to render the simulation? (y, n): ", "Invalid choice. Please choose 'y' or 'n': ", ["y", "n"])
        render = get_bool(render)
     #########################################################################################
    if args.log is not None:
        log_wandb = True
        wandb_project = args.log
    else:
        log_wandb = False
        wandb_project = None
    if agent_type == "all":
        run_all(scenario,society_mix,checkpoint_path,num_agents,NUM_BERRIES,num_allocations,MAX_WIDTH,MAX_HEIGHT,max_episodes,max_days,training,write_data,track_norms,render,log_wandb,wandb_project,seed=args.seed,norms_module_type=norms_module_type,seasons=seasons,season_length=season_length,allotment_regrowth_delays=allotment_regrowth_delays,mixed_allotments=mixed_allotments)
    else:
        create_and_run_model(scenario,society_mix,checkpoint_path,num_agents,NUM_BERRIES,num_allocations,agent_type,MAX_WIDTH,MAX_HEIGHT,max_episodes,max_days,training,write_data,track_norms,render,log_wandb,wandb_project,seed=args.seed,norms_module_type=norms_module_type,seasons=seasons,season_length=season_length,allotment_regrowth_delays=allotment_regrowth_delays,mixed_allotments=mixed_allotments)
#########################################################################################
elif args.option == "graphs":
    graph_runs = ["current_run", "50_days", "200_days", "paper"]
    run_name = get_input(f"What run do you want to generate graphs for (select paper to reproduce graphs in the paper) {graph_runs}: ", f"Invalid name of run. Please choose {graph_runs}: ", graph_runs)
    scenario = get_input(f"What type of scenario do you want to generate graphs for {SCENARIO_TYPES}: ", f"Invalid scenario. Please choose {SCENARIO_TYPES} ", SCENARIO_TYPES)
    print("Graphs will be saved in data/results/current_run")
    if run_name == "paper":
        generate_paper_graphs(scenario)
    else:
        num_agents = int(get_input(f"How many agents {NUM_AGENTS_OPTIONS}: ", f"Invalid number of agents. Please choose {NUM_AGENTS_OPTIONS}: ", NUM_AGENTS_OPTIONS))
        seasons = get_bool(get_input("Did the run have seasons? (y, n): ", "Invalid choice. Please choose 'y' or 'n': ", ["y", "n"]))
        society_mix = get_input(f"What mix of society was it {SOCIETY_MIXES}: ", f"Invalid society mix. Please choose {SOCIETY_MIXES}: ", SOCIETY_MIXES)
        mixed_allotments = "split"
        if scenario == "allotment" and society_mix == "heterogeneous":
            separate = get_bool(get_input("Did each agent type have separate allotments? (y, n): ", "Invalid choice. Please choose 'y' or 'n': ", ["y", "n"]))
            mixed_allotments = "separate" if separate else "split"
        process_norms = get_bool(get_input("Did the run write rule-based norms to file? (y, n): ", "Invalid choice. Please choose 'y' or 'n': ", ["y", "n"]))
        generate_graphs(scenario,run_name,num_agents,seasons,mixed_allotments=mixed_allotments,society_mix=society_mix,process_norms=process_norms)