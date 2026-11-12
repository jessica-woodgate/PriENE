class HarvestException(Exception):
    """Every exception here builds its message once, from its arguments, and passes it to Exception"""

class FileExistsException(HarvestException):
    def __init__(self, file_name):
        super().__init__(f"File {file_name} already exists")

class NoBerriesException(HarvestException):
    def __init__(self, agent_id=None, coordinates=None):
        if coordinates == None:
            message = f"Agent {agent_id} has found no berries in the grid"
        elif agent_id == None:
            message = f"Coordinates {coordinates} has no berries"
        else:
            message = f"Agent {agent_id} trying to forage at {coordinates} which has no berry"
        super().__init__(message)

class NumAgentsException(HarvestException):
    def __init__(self, num_expected_agents, num_agents):
        super().__init__(f"Expected {num_expected_agents} agents and got {num_agents}")

class NumAllotmentsException(HarvestException):
    def __init__(self, num_agents, num_allotments):
        super().__init__(f"The number of allotments must be between 1 and the number of agents: currently asking for {num_agents} agents and {num_allotments} allotments")

class AllotmentWidthException(HarvestException):
    def __init__(self, max_width, num_allotments):
        super().__init__(f"Every allotment must be at least 1 column wide: a grid {max_width} columns wide can't be split into {num_allotments} allotments")

class AllotmentCapacityException(HarvestException):
    def __init__(self, allotment_id, allotment, num_cells, num_agents, num_berries):
        super().__init__(f"Allotment {allotment_id} (columns {allotment[0]} to {allotment[1]-1}, rows {allotment[2]} to {allotment[3]-1}) has {num_cells} cells, "
                         f"not enough for its {num_agents} agents and up to {num_berries} berries; use a bigger grid or fewer allotments")

class MixedSocietyException(HarvestException):
    def __init__(self, reason):
        super().__init__(f"Can't set up a mixed society: {reason}")

class AgentTypeException(HarvestException):
    def __init__(self, expected_type, agent_type):
        super().__init__(f"Expected type {expected_type} and got {agent_type}")

class OutOfBounds(HarvestException):
    def __init__(self, agent_id, coordinates):
        super().__init__(f"Coordinates {coordinates} are out of bounds for agent {agent_id}")

class NoEmptyCells(HarvestException):
    def __init__(self):
        super().__init__("No empty cells in grid")

class NoAllocationException(HarvestException):
    def __init__(self, agent_id):
        super().__init__(f"No allocation found for agent {agent_id}")

class UnrecognisedPrinciple(HarvestException):
    def __init__(self, principle):
        super().__init__(f"Do not recognise principle {principle}")

class UnrecognisedNormsModule(HarvestException):
    def __init__(self, norms_module_type, valid_types):
        super().__init__(f"Do not recognise norms module type {norms_module_type}; choose from {valid_types}")

class UnmappedActionException(HarvestException):
    def __init__(self, missing_actions):
        super().__init__(f"action_name_to_index is missing entries for: {sorted(missing_actions)}")

class NoPathFound(HarvestException):
    def __init__(self, agent_id, agent_coordinates, berry_coordinates):
        super().__init__(f"Agent {agent_id} couldn't find a path from {agent_coordinates} to {berry_coordinates}")

class IllegalBerry(HarvestException):
    def __init__(self, agent_id, illegal_move):
        super().__init__(f"Agent {agent_id} is trying to access illegal berry: {illegal_move}")

class NumBerriesException(HarvestException):
    def __init__(self, num_expected_berries, num_berries):
        super().__init__(f"Expected {num_expected_berries} berries and got {num_berries}")

class NumFeaturesException(HarvestException):
    def __init__(self, num_expected_features, num_features):
        super().__init__(f"Expected {num_expected_features} features and got {num_features}")

class CheckpointNotFoundException(HarvestException):
    def __init__(self, checkpoint_path, agent_types=None):
        message = f"No checkpoint at {checkpoint_path}; test mode loads a checkpoint trained with the same agent type and number of agents"
        if agent_types is not None:
            message += (f". These agents were trained together as {agent_types}, so test them in a society with the same type for each agent "
                        "(the same scenario layout and mixed allotments)")
        super().__init__(message)

class CheckpointExistsException(HarvestException):
    def __init__(self, checkpoint_path):
        super().__init__(f"A checkpoint already exists at {checkpoint_path}; training never overwrites checkpoints, so use other settings or another seed, or move it")

class CheckpointMismatchException(HarvestException):
    def __init__(self, checkpoint_path, expected_features, expected_actions, checkpoint_features, checkpoint_actions):
        #agents observe 4 features of their own plus one per other agent (plus any the scenario adds), and have move, eat and one throw per other agent
        if checkpoint_actions == expected_actions:
            #the same number of agents, so the scenarios observe different features (e.g. agents in the commons also observe its stock)
            advice = "it was trained in a scenario whose agents observe different features (agents in the commons also observe its stock); test it in the scenario it was trained in"
        else:
            advice = "test with the number of agents it was trained with"
        super().__init__(f"Checkpoint {checkpoint_path} was trained with {checkpoint_actions - 1} agents ({checkpoint_features} inputs, "
                         f"{checkpoint_actions} actions), but this run has {expected_actions - 1} agents ({expected_features} inputs, "
                         f"{expected_actions} actions); {advice}")

class SeasonsException(HarvestException):
    def __init__(self, reason):
        super().__init__(f"Can't use seasons: {reason}")

class RegrowthDelaysException(HarvestException):
    def __init__(self, num_allotments, regrowth_delays):
        super().__init__(f"Expected {num_allotments} allotment regrowth delays (one per allotment, each a whole number of steps of at least 0) and got {regrowth_delays}")

class RegrowthDelaysNotSupported(HarvestException):
    def __init__(self, scenario):
        super().__init__(f"The {scenario} scenario has no allotments; allotment regrowth delays can only be used with the allotment scenario")

class CommonsParameterException(HarvestException):
    def __init__(self, reason):
        super().__init__(f"Can't set up the commons: {reason}")

class CommonsGridException(HarvestException):
    def __init__(self, carrying_capacity, num_agents, num_cells):
        super().__init__(f"The commons needs a free cell for every berry up to its carrying capacity and for every agent: carrying capacity "
                         f"{carrying_capacity} and {num_agents} agents need {carrying_capacity + num_agents} cells, but the grid has {num_cells}")

class BerryOffGrid(HarvestException):
    def __init__(self, berry_id, day):
        super().__init__(f"Berry {berry_id} was off the grid at the season change on day {day}; every berry should have regrown or be waiting to regrow by then")
