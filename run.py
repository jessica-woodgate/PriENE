from src.scenarios.basic_harvest import BasicHarvest
from src.scenarios.colours_harvest import ColoursHarvest
from src.scenarios.allotment_harvest import AllotmentHarvest
from src.scenarios.capabilities_harvest import CapabilitiesHarvest
from src.scenarios.commons_harvest import CommonsHarvest
from src.data_handling.data_analysis import DataAnalysis
from src.data_handling.render_pygame import RenderPygame
from src.agent.harvest_agent import NORMS_MODULE_TYPES
from src.checkpoints import RUN_OPTIONS, DEFAULT_CHECKPOINT
from src.checkpoints import training_checkpoint_name, training_checkpoint_path, prepare_training_checkpoints, fruit_market_checkpoint_name
from src.checkpoints import society_checkpoint_path, loading_checkpoint_path, record_agent_types, training_folder, finished_episode
from src.checkpoints import resolve_checkpoint, trained_checkpoints, checkpoint_max_days, days_in_name
from src.harvest_exception import CheckpointExistsException
from src.training_config import load as load_training_config, for_scenario as training_config_for_scenario, DEFAULT as DEFAULT_TRAINING_CONFIG
import pandas as pd
import argparse
import os
import wandb
import time

PRINCIPLES = ["baseline", "utilitarian", "maximin", "egalitarian"]
AGGREGATIONS = ["average", "majoritarian", "optimist", "veto"]
AGENT_TYPES = PRINCIPLES + AGGREGATIONS
HARVEST_SCENARIO_TYPES = ["basic", "colours", "capabilities", "allotment", "commons"]
#fruit_market is Melting Pot's Fruit Market (src/fruit_market), an optional extra: it needs Melting Pot installed (requirements-fruit-market.txt)
SCENARIO_TYPES = HARVEST_SCENARIO_TYPES + ["fruit_market"]
FRUIT_MARKET_PLAYER_OPTIONS = ["2", "4", "8", "16"]
#the substrate's own episode length and number of players
FRUIT_MARKET_STEPS = 1000
FRUIT_MARKET_PLAYERS = 16
FRUIT_MARKET_TEST_EPISODES = 100
NUM_AGENTS_OPTIONS = ["2", "4", "6", "20", "50", "100"]
MAX_EPISODES = 1000
SOCIETY_MIXES = ["homogeneous", "heterogeneous"]
DEFAULT_SEED = 42
DEFAULT_SEASON_LENGTH = 25
#the paper's settings, used when enter is pressed at a prompt: agents trained in basic (and tested in colours, capabilities and
#allotment), 200 days, 4 agents, 2 allotments or capabilities, no seasons, societies of one type, every agent type, rule-based norms
PAPER_DEFAULTS = {"train_scenario": "basic", "test_scenario": "colours", "seasons": "n", "max_days": 200, "society_mix": "homogeneous",
                  "agent_type": "all", "num_agents": "4", "num_allocations": 2, "mixed_allotments": "split", "write_data": "y",
                  "track_norms": "y", "norms_module_type": "rule_based", "render": "n", "graph_run": "paper"}
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
    #every agent type, in the order of the labels (principles + aggregations) the analysis gives them
    agent_end_episode_dfs = [pd.read_csv(agents_filepath+agent_type+".csv") for agent_type in AGENT_TYPES]
    end_episode_totals_dfs = [pd.read_csv(episodes_filepath+agent_type+".csv") for agent_type in AGENT_TYPES]
    data_analysis.proccess_and_display_data(agent_end_episode_dfs, PRINCIPLES, AGGREGATIONS, scenario, end_episode_totals_dfs=end_episode_totals_dfs, write=True, process_norms=False)

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
    data_analysis.proccess_and_display_data(dfs, principles, AGGREGATIONS, scenario, norms_filepath, write=True, process_norms=process_norms)

def format_duration(seconds):
    """
    A duration as hours, minutes and seconds, e.g. 3h 05m 12s (or 5m 12s, or 12s)
    """
    hours, remainder = divmod(int(round(seconds)), 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"

def within_training_budget(model_inst):
    """
    Whether a training run may start or carry on its current episode: always, unless its training config limits it to
    max_training_episodes and that many have finished
    """
    budget = model_inst.training_config.max_training_episodes
    return budget is None or model_inst.episode <= budget

def run_simulation(model_inst, render, log_wandb, wandb_project):
    start_time = time.time()
    if log_wandb:
        wandb.init(project=wandb_project)
    if render:
        render_inst = RenderPygame(model_inst.max_width, model_inst.max_height)
    #training runs until exploration has decayed to its floor; testing runs max_episodes
    while (model_inst.training and round(model_inst.epsilon, 5) > model_inst.min_epsilon and within_training_budget(model_inst)) or (not model_inst.training and model_inst.episode <= model_inst.max_episodes):
        model_inst.step()
        if log_wandb:
            mean_reward = model_inst._mean_reward()
            wandb.log({'mean_episode_reward': mean_reward})
        if render:
            if not render_inst.render_pygame(model_inst):
                break
    model_inst.finish_episode(collect_data=False)
    num_episodes = model_inst.episode
    end_time = time.time()
    print(f"Simulation finished at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}, after {format_duration(end_time - start_time)}")
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
        return get_integer_input("How many days in each episode: ", default=PAPER_DEFAULTS["max_days"])
    if days_in_name(checkpoint_name) is None:
        print(f"Testing with {max_days} days in each episode, as the checkpoint was trained.")
    return max_days

def output_file_string(training, scenario, seasons, allotment_regrowth_delays, mixed_allotments, society_mix, agent_type):
    #training outputs are named apart from testing outputs, so testing the same scenario doesn't clash with them
    return ("train_" if training else "")+scenario_file_string(scenario, seasons, allotment_regrowth_delays, mixed_allotments, society_mix)+"_"+agent_type

def get_continue_training_input(checkpoint_path, agent_types, society_mix, file_strings, report_prefix="model_episode_reports_"):
    """
    For each agent type whose training run has already saved into checkpoint_path, ask whether to continue it from the episode after the
    last one it finished, since training never overwrites a run. Returns the episode each continued agent type starts from
    file_strings -- each agent type's training output name, so its report (report_prefix + name) is found
    """
    start_episodes = {}
    for agent_type in agent_types:
        folder = training_folder(checkpoint_path, society_mix, agent_type)
        if not os.path.exists(folder):
            continue
        finished = finished_episode(folder, "data/results/current_run/"+report_prefix+file_strings[agent_type]+".csv")
        if finished is None:
            print(f"No episode of {agent_type}'s training in {folder} finished, so there is nothing to continue; move it to train again.")
            raise CheckpointExistsException(folder)
        continue_run = get_bool(get_input(f"{agent_type}'s training in {folder} stopped after episode {finished}. Do you want to continue it from episode {finished+1}? (y, n): ",
                                          "Invalid choice. Please choose 'y' or 'n': ", ["y", "n"]))
        if not continue_run:
            raise CheckpointExistsException(folder)
        start_episodes[agent_type] = finished+1
    return start_episodes

def create_and_run_model(scenario,society_mix,checkpoint_path,num_agents,num_start_berries,num_allocations,agent_type,max_width,max_height,max_episodes,max_days,training,write_data,track_norms,render,log_wandb,wandb_project=None,seed=None,norms_module_type="rule_based",seasons=False,season_length=DEFAULT_SEASON_LENGTH,allotment_regrowth_delays=None,mixed_allotments="split",start_episode=1,commons_parameters=None,training_config=None):
    """
    checkpoint_path is the folder holding the checkpoints for this number of agents: training saves there, testing loads from it
    (in a mixed society, from the folder of agent_type trained with baseline agents; see society_checkpoint_path)
    start_episode is 1, or the episode a stopped training run continues from
    commons_parameters is the commons' carrying_capacity, growth_rate and initial_stock (commons only)
    training_config is the training parameters (src/training_config.py); None uses configs/training/harvest.yaml's
    """
    file_string = output_file_string(training, scenario, seasons, allotment_regrowth_delays, mixed_allotments, society_mix, agent_type)
    if training:
        checkpoint_path = society_checkpoint_path(checkpoint_path, society_mix, agent_type)
    else:
        checkpoint_path = loading_checkpoint_path(checkpoint_path, society_mix, agent_type)
    if scenario == "basic":
        model_inst = BasicHarvest(society_mix,num_agents,num_start_berries,agent_type,max_width,max_height,max_episodes,max_days,training,checkpoint_path,write_data,track_norms,file_string,seed,norms_module_type,seasons,season_length,allotment_regrowth_delays,start_episode=start_episode,training_config=training_config)
    elif scenario == "colours":
        model_inst = ColoursHarvest(society_mix,num_agents,num_start_berries,agent_type,max_width,max_height,max_episodes,max_days,training,checkpoint_path,write_data,track_norms,file_string,seed,norms_module_type,seasons,season_length,allotment_regrowth_delays,start_episode=start_episode,training_config=training_config)
    elif scenario == "allotment":
        model_inst = AllotmentHarvest(society_mix,num_agents,num_start_berries,num_allocations,agent_type,max_width,max_height,max_episodes,max_days,training,checkpoint_path,write_data,track_norms,file_string,seed,norms_module_type,seasons,season_length,allotment_regrowth_delays,mixed_allotments,start_episode=start_episode,training_config=training_config)
    elif scenario == "capabilities":
        model_inst = CapabilitiesHarvest(society_mix,num_agents,num_start_berries,num_allocations,agent_type,max_width,max_height,max_episodes,max_days,training,checkpoint_path,write_data,track_norms,file_string,seed,norms_module_type,seasons,season_length,allotment_regrowth_delays,start_episode=start_episode,training_config=training_config)
    elif scenario == "commons":
        model_inst = CommonsHarvest(society_mix,num_agents,commons_parameters["carrying_capacity"],commons_parameters["growth_rate"],agent_type,max_width,max_height,max_episodes,max_days,training,checkpoint_path,write_data,track_norms,file_string,seed,norms_module_type,seasons,season_length,allotment_regrowth_delays,start_episode=start_episode,initial_stock=commons_parameters["initial_stock"],training_config=training_config)
    else:
        raise ValueError("Unknown argument: "+scenario)
    if start_episode > 1:
        model_inst.reseed_for_continuing()
        print(f"Continuing {agent_type}'s training from episode {start_episode}")
    if training and society_mix != "homogeneous":
        record_agent_types(checkpoint_path, model_inst.get_living_agents())
    run_simulation(model_inst,render,log_wandb,wandb_project)

def agent_types_to_run(society_mix):
    #in a mixed society half the agents are baseline, so baseline alone isn't run
    return AGENT_TYPES[1:] if society_mix == "heterogeneous" else AGENT_TYPES

def run_all(scenario,society_mix,checkpoint_path,num_agents,num_start_berries,num_allocations,max_width,max_height,max_episodes,max_days,training,write_data,track_norms,render,log_wandb,wandb_project=None,seed=None,norms_module_type="rule_based",seasons=False,season_length=DEFAULT_SEASON_LENGTH,allotment_regrowth_delays=None,mixed_allotments="split",start_episodes=None,commons_parameters=None,training_config=None):
    for agent_type in agent_types_to_run(society_mix):
        start_episode = start_episodes.get(agent_type, 1) if start_episodes else 1
        create_and_run_model(scenario,society_mix,checkpoint_path,num_agents,num_start_berries,num_allocations,agent_type,max_width,max_height,max_episodes,max_days,training,write_data,track_norms,render,log_wandb,wandb_project,seed,norms_module_type,seasons,season_length,allotment_regrowth_delays,mixed_allotments,start_episode,commons_parameters,training_config)

def get_commons_parameters_input(num_agents, num_cells):
    """
    Ask for the commons' carrying capacity K, growth rate r and initial stock R0; pressing enter uses each default
    (CommonsHarvest.CARRYING_CAPACITY_PER_AGENT per agent, CommonsHarvest.DEFAULT_GROWTH_RATE, and K // 2)
    """
    max_capacity = num_cells - num_agents
    default_capacity = min(CommonsHarvest.CARRYING_CAPACITY_PER_AGENT * num_agents, max_capacity)
    carrying_capacity = get_number_input(f"What carrying capacity (most berries the commons holds) do you want? (at least 1 and at most {max_capacity}, so every berry and agent has a cell) Press enter for {default_capacity}: ",
                                         int, default_capacity, 1, max_capacity)
    growth_rate = get_number_input(f"What growth rate do you want? (more than 0) Press enter for {CommonsHarvest.DEFAULT_GROWTH_RATE}: ",
                                   float, CommonsHarvest.DEFAULT_GROWTH_RATE, 0, None, exclusive_min=True)
    initial_stock = get_number_input(f"How many berries should the commons start each episode with? (at least 1 and at most {carrying_capacity}) Press enter for {carrying_capacity // 2}: ",
                                     int, max(1, carrying_capacity // 2), 1, carrying_capacity)
    return {"carrying_capacity": carrying_capacity, "growth_rate": growth_rate, "initial_stock": initial_stock}

def get_number_input(prompt, number_type, default, min_value, max_value, exclusive_min=False, out_of_range="Invalid choice. Out of range."):
    """
    Ask until a number_type at least min_value (more than it, if exclusive_min) and at most max_value is given (None: no bound),
    printing out_of_range for one outside them; pressing enter chooses default, if there is one
    """
    while True:
        value = input(prompt).strip()
        if value == "" and default is not None:
            return default
        try:
            number = number_type(value)
        except ValueError:
            print(f"Invalid input. Please enter {'an integer' if number_type is int else 'a number'}.")
            continue
        if (min_value is not None and (number < min_value or (exclusive_min and number == min_value))) or (max_value is not None and number > max_value):
            print(out_of_range)
            continue
        return number

def get_mixed_allotments_input(num_allotments):
    """
    Ask how the two agent types of a mixed society share the allotments; separate needs an even number of allotments
    """
    if num_allotments % 2 != 0:
        print(f"With {num_allotments} allotments the agent types are split within the allotments (separate allotments need an even number).")
        return "split"
    return get_input("How should the two agent types share the allotments? split: each allotment has agents of both types; separate: each type has its own allotments ['split', 'separate']: ",
                     "Invalid choice. Please choose ['split', 'separate']: ", ["split", "separate"], PAPER_DEFAULTS["mixed_allotments"])

def fruit_market_file_string(training, settings, society_mix, agent_type):
    """
    Output name of a Fruit Market run: fruit_market, then any setting that isn't the default (docs/fruit_market_design.md §10), the mix
    and the agent type, e.g. train_fruit_market_grappling-off_utilitarian
    """
    parts = ["fruit_market"]
    if settings["max_days"] != FRUIT_MARKET_STEPS:
        parts.append(f"{settings['max_days']}steps")
    if settings["num_agents"] != FRUIT_MARKET_PLAYERS:
        parts.append(f"{settings['num_agents']}players")
    if not settings["grappling"]:
        parts.append("grappling-off")
    for key, default in [("well_being", "need"), ("sanction", "attributed")]:
        if settings[key] != default:
            parts.append(settings[key])
    if settings["observation"] != "global":
        parts.append("view")
    if society_mix != "homogeneous":
        parts.append("mixed")
    return ("train_" if training else "")+"_".join(parts)+"_"+agent_type

def get_fruit_market_settings_input():
    """
    Ask for a Fruit Market training run's settings; pressing enter at each gives the original substrate and the design's defaults
    """
    from src.fruit_market.well_being import WELL_BEING_MODES
    from src.fruit_market.sanction import SANCTION_MODES
    from src.fruit_market.features import OBSERVATION_MODES
    max_days = get_number_input(f"How many steps in each episode? (1 to {FRUIT_MARKET_STEPS}; the substrate's own is {FRUIT_MARKET_STEPS}) Press enter for {FRUIT_MARKET_STEPS}: ",
                                 int, FRUIT_MARKET_STEPS, 1, FRUIT_MARKET_STEPS, out_of_range=f"Invalid choice. Must be between 1 and {FRUIT_MARKET_STEPS}.")
    num_agents = int(get_input(f"How many players {FRUIT_MARKET_PLAYER_OPTIONS} (half apple farmers, half banana farmers; the substrate's own is {FRUIT_MARKET_PLAYERS}): ",
                               f"Invalid number of players. Please choose {FRUIT_MARKET_PLAYER_OPTIONS}: ", FRUIT_MARKET_PLAYER_OPTIONS, str(FRUIT_MARKET_PLAYERS)))
    grappling = get_bool(get_input("Do you want grappling (hold, shove, pull), as in the substrate? (y, n): ", "Invalid choice. Please choose 'y' or 'n': ", ["y", "n"], "y"))
    well_being = get_input(f"Which well-being do the ethics principles use {WELL_BEING_MODES} (need: steps until hungry; preference: fruit weighted by its value to the player): ",
                           f"Invalid well-being. Please choose {WELL_BEING_MODES}: ", WELL_BEING_MODES, "need")
    sanction = get_input(f"What is the ethics sanction from {SANCTION_MODES} (attributed: only the agent's own action; joint: everyone's, and time passing): ",
                         f"Invalid choice. Please choose {SANCTION_MODES}: ", SANCTION_MODES, "attributed")
    observation = get_input(f"Which spatial features do agents observe {OBSERVATION_MODES} (view: only what is in the player's view): ",
                            f"Invalid choice. Please choose {OBSERVATION_MODES}: ", OBSERVATION_MODES, "global")
    return {"scenario": "fruit_market", "max_days": max_days, "num_agents": num_agents, "grappling": grappling, "well_being": well_being,
            "sanction": sanction, "observation": observation}

def fruit_market_test_settings(checkpoint_root):
    """
    The settings a Fruit Market checkpoint was trained with (its settings.json), and the folder of its players: asks which number of
    players if it has more than one
    """
    import json
    from src.checkpoints import SETTINGS_FILE
    folders = sorted(name for name in os.listdir(checkpoint_root) if name.endswith("_agents") and os.path.exists(checkpoint_root+name+"/"+SETTINGS_FILE))
    if not folders:
        raise ValueError(f"{checkpoint_root} has no Fruit Market checkpoints (no <n>_agents/{SETTINGS_FILE})")
    options = [name.split("_")[0] for name in folders]
    players = options[0] if len(options) == 1 else get_input(f"How many players {options}: ", f"Invalid number of players. Please choose {options}: ", options)
    checkpoint_path = checkpoint_root+players+"_agents/"
    with open(checkpoint_path+SETTINGS_FILE) as file:
        settings = json.load(file)
    if settings.get("scenario") != "fruit_market":
        raise ValueError(f"{checkpoint_path} wasn't trained in fruit_market")
    print(f"Testing with the settings the checkpoint was trained with: {settings}")
    return checkpoint_path, settings

def run_fruit_market(training, seed, log=None, training_config=None):
    """
    Train or test agents in Melting Pot's Fruit Market (docs/fruit_market_design.md), asking for its options
    training_config -- the training parameters to use instead of configs/training/fruit_market.yaml's (None: that file's; when testing,
        the ones the checkpoint was trained with)
    """
    from src.training_config import TrainingConfig
    scenario_config = training_config_for_scenario("fruit_market")
    from src.fruit_market.backend import meltingpot_available, INSTALL_HINT
    if not meltingpot_available():
        print(INSTALL_HINT)
        return
    from src.fruit_market.model import FruitMarketModel
    mode = "train" if training else "test"
    if training:
        settings = get_fruit_market_settings_input()
        max_episodes = 0
    else:
        checkpoint_root, _ = get_checkpoint_input()
        checkpoint_path, settings = fruit_market_test_settings(checkpoint_root)
        max_episodes = get_number_input(f"How many episodes do you want to test? Press enter for {FRUIT_MARKET_TEST_EPISODES}: ", int, FRUIT_MARKET_TEST_EPISODES, 1, None,
                                        out_of_range="Invalid choice. Must be at least 1.")
    society_mix = get_input(f"What mix of society do you want to {mode} {SOCIETY_MIXES}: ", f"Invalid society mix. Please choose {SOCIETY_MIXES}: ", SOCIETY_MIXES, PAPER_DEFAULTS["society_mix"])
    types = AGENT_TYPES + ["all"]
    agent_type = get_input(f"What type of agent do you want to implement {types}: ", f"Invalid agent type. Please choose {types}: ", types, PAPER_DEFAULTS["agent_type"])
    write_data = write_data_input("data", PAPER_DEFAULTS["write_data"])
    agent_types = agent_types_to_run(society_mix) if agent_type == "all" else [agent_type]
    track_norms = False
    norms_module_type = NORMS_MODULE_TYPES[0]
    start_episodes = {}
    if training:
        training_config = training_config or scenario_config
        checkpoint_path = training_checkpoint_path(fruit_market_checkpoint_name(settings["max_days"], settings["grappling"], settings["well_being"], settings["sanction"],
                                                                                settings["observation"], seed, society_mix), settings["num_agents"])
        start_episodes = get_continue_training_input(checkpoint_path, agent_types, society_mix,
                                                     {t: fruit_market_file_string(True, settings, society_mix, t) for t in agent_types}, report_prefix="fm_model_episode_reports_")
        #the training config is recorded only when it isn't the scenario's own, so runs trained with other parameters can't share a checkpoint
        recorded_config = {} if training_config == scenario_config else {"training_config": training_config.to_dict()}
        prepare_training_checkpoints(checkpoint_path, agent_types, {**settings, "seed": seed, "society_mix": society_mix, **recorded_config}, society_mix, continuing=start_episodes)
        print("Model variables will be written into", checkpoint_path)
    else:
        if training_config is None:
            training_config = TrainingConfig(**settings["training_config"]) if "training_config" in settings else scenario_config
        track_norms = write_data_input("norms", PAPER_DEFAULTS["track_norms"])
        if track_norms:
            norms_module_type = get_input(f"Which norms module do you want to use {NORMS_MODULE_TYPES}: ", f"Invalid norms module. Please choose {NORMS_MODULE_TYPES}: ", NORMS_MODULE_TYPES,
                                          PAPER_DEFAULTS["norms_module_type"])
    for this_type in agent_types:
        start_episode = start_episodes.get(this_type, 1)
        file_string = fruit_market_file_string(training, settings, society_mix, this_type)
        type_checkpoint_path = society_checkpoint_path(checkpoint_path, society_mix, this_type) if training else loading_checkpoint_path(checkpoint_path, society_mix, this_type)
        model_inst = FruitMarketModel(society_mix, settings["num_agents"], this_type, max_episodes, settings["max_days"], training, type_checkpoint_path, write_data,
                                      track_norms, file_string, seed, norms_module_type, start_episode, settings["grappling"], settings["well_being"],
                                      settings["sanction"], settings["observation"], training_config=training_config)
        if start_episode > 1:
            model_inst.reseed_for_continuing()
            print(f"Continuing {this_type}'s training from episode {start_episode}")
        if training and society_mix != "homogeneous":
            record_agent_types(type_checkpoint_path, model_inst.get_living_agents())
        run_simulation(model_inst, False, log is not None, log)
        model_inst.backend.close()

class RemovedOption(argparse.Action):
    """
    Fails with a message saying the option isn't a flag, whatever value it is given
    """
    def __call__(self, parser, namespace, values, option_string=None):
        parser.error(f"{option_string} is not an option: {REMOVED_OPTIONS[option_string]}")

def with_default(prompt, default):
    """
    The prompt, saying what pressing enter chooses (when there is a default)
    """
    if default is None:
        return prompt
    return prompt.rstrip().rstrip(":") + f" Press enter for {default}: "

def get_integer_input(prompt, error_string=None, max_value=None, min_value=None, default=None):
    return get_number_input(with_default(prompt, default), int, default, min_value, max_value, out_of_range=error_string)

def get_input(input_string, error_string, valid_options, default=None):
    """
    Ask until one of valid_options is given; pressing enter chooses default, if there is one
    """
    variable = input(with_default(input_string, default)).strip()
    while variable not in valid_options:
        if variable == "" and default is not None:
            return default
        variable = input(with_default(error_string, default)).strip()
    return variable

def get_bool(input):
    if input == "y":
        return True
    elif input == "n":
        return False

def write_data_input(data_type, default=None):
    write_data = get_input(f"Do you want to write {data_type} to file? (y, n): ", "Invalid choice. Please choose 'y' or 'n': ", ["y", "n"], default)
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
parser.add_argument("--training-config", type=str, default=None,
                    help="Training parameters file to use instead of the scenario's (configs/training/), e.g. one written by the parameter sweep")
parser.add_argument("-l", "--log", type=str, default=None,
                    help="Log wandb (optional)")
for removed_option in REMOVED_OPTIONS:
    parser.add_argument(removed_option, action=RemovedOption, nargs="*", help=argparse.SUPPRESS)
args = parser.parse_args()

#argparse only accepts test, train or graphs
if args.option == "test" or args.option == "train":
    #the mode decides learning, exploration, saving and outputs; any scenario, with or without seasons, runs in either mode
    training = args.option == "train"
    mode = "train" if training else "test"
    scenario = get_input(f"What type of scenario do you want to {mode} in {SCENARIO_TYPES}: ", f"Invalid scenario. Please choose {SCENARIO_TYPES}: ", SCENARIO_TYPES,
                         PAPER_DEFAULTS["train_scenario" if training else "test_scenario"])
    #the training parameters: the scenario's own file in configs/training/, unless --training-config gives another
    training_config = load_training_config(args.training_config) if args.training_config else training_config_for_scenario(scenario)
    if scenario == "fruit_market":
        #Fruit Market has its own options (see run_fruit_market); none of the harvest scenarios' are asked
        run_fruit_market(training, args.seed, args.log, None if args.training_config is None else training_config)
        raise SystemExit(0)
    season_effect ="the commons regrows 1.5 times as fast" if scenario == "commons" else "one more berry per agent in each group"
    season_bad = "half as fast" if scenario == "commons" else "one fewer"
    seasons = get_bool(get_input(f"Do you want seasons, where each season is randomly normal, good ({season_effect}) or bad ({season_bad})? (y, n): ", "Invalid choice. Please choose 'y' or 'n': ", ["y", "n"], PAPER_DEFAULTS["seasons"]))
    season_length = DEFAULT_SEASON_LENGTH
    if seasons:
        season_length = get_number_input(f"How many days in each season? Press enter for the default {DEFAULT_SEASON_LENGTH}: ", int, DEFAULT_SEASON_LENGTH,
                                         1, None, out_of_range="Invalid choice. Must be at least 1.")
    if training:
        max_episodes = 0
        max_days = get_integer_input("How many days in each episode: ", "Invalid choice. Must be at least 1", min_value=1, default=PAPER_DEFAULTS["max_days"])
    else:
        checkpoint_root, checkpoint_name = get_checkpoint_input()
        max_episodes = MAX_EPISODES
    #in a mixed society half the agents are baseline, and train or test alongside the other half
    society_mix = get_input(f"What mix of society do you want to {mode} {SOCIETY_MIXES}: ", f"Invalid society mix. Please choose {SOCIETY_MIXES}: ", SOCIETY_MIXES, PAPER_DEFAULTS["society_mix"])
    #########################################################################################
    types = AGENT_TYPES + ["all"]
    agent_type = get_input(f"What type of agent do you want to implement {types}: ", f"Invalid agent type. Please choose {types}: ", types, PAPER_DEFAULTS["agent_type"])
    #########################################################################################
    num_agents = int(get_input(f"How many agents do you want to implement {NUM_AGENTS_OPTIONS}: ", f"Invalid number of agents. Please choose {NUM_AGENTS_OPTIONS}: ", NUM_AGENTS_OPTIONS, PAPER_DEFAULTS["num_agents"]))
    #########################################################################################
    #allotment uses its default regrowth delays (allotment 1 regrows slower than the others)
    allotment_regrowth_delays = None
    mixed_allotments = "split"
    if scenario == "allotment":
        MAX_WIDTH = num_agents * 4
        num_allocations = get_integer_input(f"How many allotments do you want? (must be more than 0 and less than {num_agents+1}): ", f"Invalid choice. Must be a number between 1 and {num_agents}", num_agents, 1,
                                            min(PAPER_DEFAULTS["num_allocations"], num_agents))
        if society_mix == "heterogeneous":
            mixed_allotments = get_mixed_allotments_input(num_allocations)
    elif scenario == "capabilities":
        MAX_WIDTH = num_agents * 2
        num_allocations = get_integer_input(f"How many capabilities do you want? (must be more than 0 and less than {num_agents+1}): ", f"Invalid choice. Must be a number between 1 and {num_agents}", num_agents, 1,
                                            min(PAPER_DEFAULTS["num_allocations"], num_agents))
    else:
        MAX_WIDTH = num_agents * 2
        num_allocations = 1
    MAX_HEIGHT = num_agents * 2
    NUM_BERRIES = num_agents * 3
    #the commons' stock is its own, so NUM_BERRIES isn't used
    commons_parameters = get_commons_parameters_input(num_agents, MAX_WIDTH * MAX_HEIGHT) if scenario == "commons" else None
    #########################################################################################
    write_data = write_data_input("data", PAPER_DEFAULTS["write_data"])
    #########################################################################################
    if training:
        #the delays allotments actually regrow with, so the name says which were used
        delays = allotment_regrowth_delays
        if scenario == "allotment" and delays is None:
            delays = AllotmentHarvest.default_regrowth_delays(num_allocations)
        checkpoint_path = training_checkpoint_path(training_checkpoint_name(scenario, seasons, season_length, num_allocations, delays, max_days, args.seed, society_mix, mixed_allotments, commons_parameters), num_agents)
    else:
        checkpoint_path = checkpoint_root+str(num_agents)+"_agents/"
        max_days = get_max_days_input(checkpoint_name, checkpoint_path)
    #########################################################################################
    norms_module_type = NORMS_MODULE_TYPES[0]
    start_episodes = {}
    if training:
        training_types = agent_types_to_run(society_mix) if agent_type == "all" else [agent_type]
        #a run that has already saved here can only be continued, never overwritten
        start_episodes = get_continue_training_input(checkpoint_path, training_types, society_mix,
                                                     {t: output_file_string(training, scenario, seasons, allotment_regrowth_delays, mixed_allotments, society_mix, t) for t in training_types})
        prepare_training_checkpoints(checkpoint_path, training_types,
                                     {"scenario": scenario, "seasons": seasons, "season_length": season_length if seasons else None,
                                      "num_allocations": num_allocations, "allotment_regrowth_delays": allotment_regrowth_delays,
                                      "max_days": max_days, "seed": args.seed, "num_agents": num_agents, "society_mix": society_mix,
                                      "mixed_allotments": mixed_allotments if scenario == "allotment" and society_mix != "homogeneous" else None,
                                      **({"commons_parameters": commons_parameters} if commons_parameters else {}),
                                      #recorded only when it isn't the harvest scenarios' own, so runs trained with other parameters can't share a checkpoint
                                      **({"training_config": training_config.to_dict()} if training_config != DEFAULT_TRAINING_CONFIG else {})},
                                     society_mix, continuing=start_episodes)
        print("Model variables will be written into",checkpoint_path)
        track_norms = False
        render = False
    else:
        track_norms = write_data_input("norms", PAPER_DEFAULTS["track_norms"])
        if track_norms:
            norms_module_type = get_input(f"Which norms module do you want to use {NORMS_MODULE_TYPES}: ", f"Invalid norms module. Please choose {NORMS_MODULE_TYPES}: ", NORMS_MODULE_TYPES,
                                          PAPER_DEFAULTS["norms_module_type"])
        render = get_input("Do you want to render the simulation? (y, n): ", "Invalid choice. Please choose 'y' or 'n': ", ["y", "n"], PAPER_DEFAULTS["render"])
        render = get_bool(render)
     #########################################################################################
    if args.log is not None:
        log_wandb = True
        wandb_project = args.log
    else:
        log_wandb = False
        wandb_project = None
    if agent_type == "all":
        run_all(scenario,society_mix,checkpoint_path,num_agents,NUM_BERRIES,num_allocations,MAX_WIDTH,MAX_HEIGHT,max_episodes,max_days,training,write_data,track_norms,render,log_wandb,wandb_project,seed=args.seed,norms_module_type=norms_module_type,seasons=seasons,season_length=season_length,allotment_regrowth_delays=allotment_regrowth_delays,mixed_allotments=mixed_allotments,start_episodes=start_episodes,commons_parameters=commons_parameters,training_config=training_config)
    else:
        create_and_run_model(scenario,society_mix,checkpoint_path,num_agents,NUM_BERRIES,num_allocations,agent_type,MAX_WIDTH,MAX_HEIGHT,max_episodes,max_days,training,write_data,track_norms,render,log_wandb,wandb_project,seed=args.seed,norms_module_type=norms_module_type,seasons=seasons,season_length=season_length,allotment_regrowth_delays=allotment_regrowth_delays,mixed_allotments=mixed_allotments,start_episode=start_episodes.get(agent_type, 1),commons_parameters=commons_parameters,training_config=training_config)
#########################################################################################
elif args.option == "graphs":
    graph_runs = ["current_run", "50_days", "200_days", "paper"]
    run_name = get_input(f"What run do you want to generate graphs for (paper reproduces the graphs in the paper) {graph_runs}: ", f"Invalid name of run. Please choose {graph_runs}: ", graph_runs,
                         PAPER_DEFAULTS["graph_run"])
    scenario = get_input(f"What type of scenario do you want to generate graphs for {HARVEST_SCENARIO_TYPES}: ", f"Invalid scenario. Please choose {HARVEST_SCENARIO_TYPES} ", HARVEST_SCENARIO_TYPES,
                         PAPER_DEFAULTS["test_scenario"])
    print("Graphs will be saved in data/results/current_run")
    if run_name == "paper":
        generate_paper_graphs(scenario)
    else:
        num_agents = int(get_input(f"How many agents {NUM_AGENTS_OPTIONS}: ", f"Invalid number of agents. Please choose {NUM_AGENTS_OPTIONS}: ", NUM_AGENTS_OPTIONS, PAPER_DEFAULTS["num_agents"]))
        seasons = get_bool(get_input("Did the run have seasons? (y, n): ", "Invalid choice. Please choose 'y' or 'n': ", ["y", "n"], PAPER_DEFAULTS["seasons"]))
        society_mix = get_input(f"What mix of society was it {SOCIETY_MIXES}: ", f"Invalid society mix. Please choose {SOCIETY_MIXES}: ", SOCIETY_MIXES, PAPER_DEFAULTS["society_mix"])
        mixed_allotments = "split"
        if scenario == "allotment" and society_mix == "heterogeneous":
            separate = get_bool(get_input("Did each agent type have separate allotments? (y, n): ", "Invalid choice. Please choose 'y' or 'n': ", ["y", "n"], "n"))
            mixed_allotments = "separate" if separate else "split"
        process_norms = get_bool(get_input("Did the run write rule-based norms to file? (y, n): ", "Invalid choice. Please choose 'y' or 'n': ", ["y", "n"], PAPER_DEFAULTS["track_norms"]))
        generate_graphs(scenario,run_name,num_agents,seasons,mixed_allotments=mixed_allotments,society_mix=society_mix,process_norms=process_norms)