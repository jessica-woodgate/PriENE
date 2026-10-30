from src.harvest_exception import CheckpointExistsException
import json
import os
import re

#checkpoints trained before checkpoints were named by their settings, in data/model_variables/<run>/; 200_days is the paper's
RUN_OPTIONS = ["current_run", "50_days", "200_days"]
DEFAULT_CHECKPOINT = "200_days"
CHECKPOINTS_DIR = "data/model_variables/"
#training saves into data/model_variables/trained/<name from its settings>/<n>_agents/<agent type>/, never overwriting a checkpoint;
#a mixed society saves into .../<n>_agents/<agent type>_with_baseline/<agent type or baseline>/, with AGENT_TYPES_FILE saying which agent is which type
TRAINED_DIR = CHECKPOINTS_DIR+"trained/"
SETTINGS_FILE = "settings.json"
AGENT_TYPES_FILE = "agent_types.json"
MIXED_SUFFIX = "_with_baseline"

def training_checkpoint_name(scenario, seasons, season_length, num_allocations, allotment_regrowth_delays, max_days, seed,
                             society_mix="homogeneous", mixed_allotments="split"):
    """
    Name of a training run's checkpoint folder, from every setting that changes what is learned (the number of agents is the
    folder inside it), e.g. allotment_seasons25_2allotments_regrow0-2_200days_seed42, or with a mixed society
    allotment_2allotments_regrow0-2_mixed-separate_200days_seed42
    allotment_regrowth_delays -- the delays the allotments regrow with (AllotmentHarvest.default_regrowth_delays when none are chosen);
        only used, and needed, for the allotment scenario
    """
    parts = [scenario]
    if seasons:
        parts.append(f"seasons{season_length}")
    if scenario == "capabilities":
        parts.append(f"{num_allocations}capabilities")
    elif scenario == "allotment":
        if allotment_regrowth_delays is None:
            raise ValueError("an allotment checkpoint name needs the regrowth delays its allotments use")
        parts += [f"{num_allocations}allotments", "regrow" + "-".join(str(delay) for delay in allotment_regrowth_delays)]
    if society_mix != "homogeneous":
        parts.append("mixed-separate" if scenario == "allotment" and mixed_allotments == "separate" else "mixed")
    parts += [f"{max_days}days", f"seed{seed}"]
    return "_".join(parts)

def training_checkpoint_path(name, num_agents, trained_dir=TRAINED_DIR):
    return trained_dir+name+"/"+str(num_agents)+"_agents/"

def society_checkpoint_path(checkpoint_path, society_mix, agent_type):
    """
    Folder holding a training run's <type>/agent_<id> checkpoints: checkpoint_path for a society of one type, or its own folder for
    agent_type trained with baseline agents, since baseline agents trained alongside each agent type learn differently
    """
    if society_mix == "homogeneous":
        return checkpoint_path
    return checkpoint_path+agent_type+MIXED_SUFFIX+"/"

def loading_checkpoint_path(checkpoint_path, society_mix, agent_type):
    """
    Folder a test run loads <type>/agent_<id> checkpoints from: for a mixed society, the agents trained together if checkpoint_path has
    them, otherwise each type's own agents (e.g. the paper's agents, each type trained in a society of its own type)
    """
    mixed_path = society_checkpoint_path(checkpoint_path, society_mix, agent_type)
    return mixed_path if os.path.isdir(mixed_path) else checkpoint_path

def record_agent_types(society_path, agents):
    """
    Write which type each agent is, as agent_<id>: type, beside a mixed society's checkpoints
    """
    with open(society_path+AGENT_TYPES_FILE, "w") as file:
        json.dump({f"agent_{agent.unique_id}": agent.agent_type for agent in sorted(agents, key=lambda a: a.unique_id)}, file, indent=4)

def recorded_agent_types(society_path):
    path = society_path+AGENT_TYPES_FILE
    if not os.path.exists(path):
        return None
    with open(path) as file:
        return json.load(file)

def prepare_training_checkpoints(checkpoint_path, agent_types, settings, society_mix="homogeneous"):
    """
    Check no agent type would overwrite a checkpoint before any training starts, and record the settings next to the checkpoints
    (an existing record must match, so agent types trained separately share one set of settings)
    """
    if society_mix == "homogeneous":
        folders = [checkpoint_path+agent_type for agent_type in agent_types]
    else:
        folders = [society_checkpoint_path(checkpoint_path, society_mix, agent_type) for agent_type in agent_types]
    existing = [folder for folder in folders if os.path.exists(folder)]
    if existing:
        raise CheckpointExistsException(", ".join(existing))
    settings_path = checkpoint_path+SETTINGS_FILE
    if os.path.exists(settings_path):
        with open(settings_path) as file:
            recorded = json.load(file)
        if recorded != settings:
            raise ValueError(f"{settings_path} records different settings {recorded} from this run's {settings}")
    else:
        os.makedirs(checkpoint_path, exist_ok=True)
        #written beside it and moved into place, so a run starting at the same time (e.g. another agent type's job) never reads it half-written
        writing_path = f"{settings_path}.{os.getpid()}.writing"
        with open(writing_path, "w") as file:
            json.dump(settings, file, indent=4)
        os.replace(writing_path, settings_path)

def resolve_checkpoint(value, trained_dir=TRAINED_DIR, checkpoints_dir=CHECKPOINTS_DIR):
    """
    Folder (holding <n>_agents folders) for a checkpoint chosen to test: "" for DEFAULT_CHECKPOINT, one of RUN_OPTIONS, a trained
    checkpoint's name, or a path to a folder; None if it is none of these
    """
    if value == "":
        value = DEFAULT_CHECKPOINT
    if value in RUN_OPTIONS:
        return checkpoints_dir+value+"/"
    if value in trained_checkpoints(trained_dir):
        return trained_dir+value+"/"
    if os.path.isdir(value):
        return value.rstrip("/")+"/"
    return None

def trained_checkpoints(trained_dir=TRAINED_DIR):
    return sorted(os.listdir(trained_dir)) if os.path.isdir(trained_dir) else []

def days_in_name(name):
    match = re.match(r"(\d+)_days", name)
    if match:
        return int(match.group(1))
    return None

def checkpoint_max_days(checkpoint_name, checkpoint_path):
    """
    Days in each episode a checkpoint was trained with: from the run's name or the last folder of its path (e.g. 200_days), or from the
    settings recorded when it was trained (None if neither says)
    """
    max_days = days_in_name(os.path.basename(checkpoint_name.rstrip("/")))
    if max_days is not None:
        return max_days
    settings_path = checkpoint_path+SETTINGS_FILE
    if os.path.exists(settings_path):
        with open(settings_path) as file:
            return json.load(file)["max_days"]
    return None
