class HarvestException(Exception):
    def __init__(self):
        super().__init__()

class FileExistsException(HarvestException):
    def __init__(self, file_name):
        super().__init__()
        self.file_name= file_name
    
    def __str__(self):
        return (f"File {self.file_name} already exists")

class NoBerriesException(HarvestException):
    def __init__(self, agent_id=None, coordinates=None):
        super().__init__()
        self.agent_id = agent_id
        self.coordinates = coordinates
    
    def __str__(self):
        if self.coordinates == None:
            return (f"Agent {self.agent_id} has found no berries in the grid")
        elif self.agent_id == None:
            return (f"Coordinates {self.coordinates} has no berries")
        else:
            return (f"Agent {self.agent_id} trying to forage at {self.coordinates} which has no berry")
        
class NumAgentsException(HarvestException):
    def __init__(self, num_expected_agents, num_agents):
        super().__init__()
        self.num_expected_agents = num_expected_agents
        self.num_agents = num_agents
    
    def __str__(self):
        return (f"Expected {self.num_expected_agents} agents and got {self.num_agents}")

class NumAllotmentsException(HarvestException):
    def __init__(self, num_agents, num_allotments):
        super().__init__()
        self.num_agents = num_agents
        self.num_allotments = num_allotments
    
    def __str__(self):
        return (f"The number of allotments must be between 1 and the number of agents: currently asking for {self.num_agents} agents and {self.num_allotments} allotments")

class AllotmentWidthException(HarvestException):
    def __init__(self, max_width, num_allotments):
        super().__init__()
        self.max_width = max_width
        self.num_allotments = num_allotments

    def __str__(self):
        return (f"Every allotment must be at least 1 column wide: a grid {self.max_width} columns wide can't be split into {self.num_allotments} allotments")

class AllotmentCapacityException(HarvestException):
    def __init__(self, allotment_id, allotment, num_cells, num_agents, num_berries):
        super().__init__()
        self.allotment_id = allotment_id
        self.allotment = allotment
        self.num_cells = num_cells
        self.num_agents = num_agents
        self.num_berries = num_berries

    def __str__(self):
        return (f"Allotment {self.allotment_id} (columns {self.allotment[0]} to {self.allotment[1]-1}, rows {self.allotment[2]} to {self.allotment[3]-1}) has {self.num_cells} cells, "
                f"not enough for its {self.num_agents} agents and up to {self.num_berries} berries; use a bigger grid or fewer allotments")

class MixedSocietyException(HarvestException):
    def __init__(self, reason):
        super().__init__()
        self.reason = reason

    def __str__(self):
        return (f"Can't set up a mixed society: {self.reason}")

class AgentTypeException(HarvestException):
    def __init__(self, expected_type, agent_type):
        super().__init__()
        self.expected_type = expected_type
        self.agent_type = agent_type
    
    def __str__(self):
        return (f"Expected type {self.expected_type} and got {self.agent_type}")

class OutOfBounds(HarvestException):
    def __init__(self, agent_id, coordinates):
        super().__init__()
        self.agent_id = agent_id
        self.coordinates = coordinates
    
    def __str__(self):
        return(f"Coordinates {self.coordinates} are out of bounds for agent {self.agent_id}")
    
class NoEmptyCells(HarvestException):
    def __init__(self):
        pass

    def __str__(self):
        return ("No empty cells in grid")
    
class NoAllocationException(HarvestException):
    def __init__(self, agent_id):
        self.agent_id = agent_id

    def __str__(self):
        return (f"No allocation found for agent {self.agent_id}")
    
class UnrecognisedPrinciple(HarvestException):
    def __init__(self, principle):
        super().__init__()
        self.principle = principle

    def __str__(self):
        return(f"Do not recognise principle {self.principle}")

class UnrecognisedNormsModule(HarvestException):
    def __init__(self, norms_module_type, valid_types):
        super().__init__()
        self.norms_module_type = norms_module_type
        self.valid_types = valid_types

    def __str__(self):
        return(f"Do not recognise norms module type {self.norms_module_type}; choose from {self.valid_types}")

class UnmappedActionException(HarvestException):
    def __init__(self, missing_actions):
        super().__init__()
        self.missing_actions = missing_actions

    def __str__(self):
        return (f"action_name_to_index is missing entries for: {sorted(self.missing_actions)}")
    
class NoPathFound(HarvestException):
    def __init__(self, agent_id, agent_coordinates, berry_coordinates):
        self.agent_id = agent_id
        self.agent_coordinates = agent_coordinates
        self.berry_coordinates = berry_coordinates

    def __str__(self):
        return (f"Agent {self.agent_id} couldn't find a path from {self.agent_coordinates} to {self.berry_coordinates}")

class IllegalBerry(HarvestException):
    def __init__(self, agent_id, illegal_move):
        self.agent_id = agent_id
        self.illegal_move = illegal_move

    def __str__(self):
        return (f"Agent {self.agent_id} is trying to access illegal berry: {self.illegal_move}")
    
class NumBerriesException(HarvestException):
    def __init__(self, num_expected_berries, num_berries):
        super().__init__()
        self.num_expected_berries = num_expected_berries
        self.num_berries = num_berries
    
    def __str__(self):
        return (f"Expected {self.num_expected_berries} berries and got {self.num_berries}")
    
class NumFeaturesException(HarvestException):
    def __init__(self, num_expected_features, num_features):
        super().__init__()
        self.num_expected_features = num_expected_features
        self.num_features = num_features

    def __str__(self):
        return (f"Expected {self.num_expected_features} features and got {self.num_features}")

class ImpossibleNormException(HarvestException):
    def __init__(self, day, agent_id, antecedent, action, reward):
        super().__init__()
        self.day = day
        self.agent_id = agent_id
        self.antecedent = antecedent
        self.action = action
        self.reward = reward
    
    def __str__(self):
        return (f"Day {self.day}, agent {self.agent_id} saving an impossible behaviour! Antecedent {self.antecedent} consequent {self.action}; got reward {self.reward}")

class SeasonsNotSupported(HarvestException):
    def __init__(self, scenario):
        super().__init__()
        self.scenario = scenario

    def __str__(self):
        return (f"The {self.scenario} scenario has no berry allocations to change; seasons can only be used with the colours, capabilities and allotment scenarios")

class SeasonsException(HarvestException):
    def __init__(self, reason):
        super().__init__()
        self.reason = reason

    def __str__(self):
        return (f"Can't use seasons: {self.reason}")

class RegrowthDelaysException(HarvestException):
    def __init__(self, num_allotments, regrowth_delays):
        super().__init__()
        self.num_allotments = num_allotments
        self.regrowth_delays = regrowth_delays

    def __str__(self):
        return (f"Expected {self.num_allotments} allotment regrowth delays (one per allotment, each a whole number of steps of at least 0) and got {self.regrowth_delays}")

class RegrowthDelaysNotSupported(HarvestException):
    def __init__(self, scenario):
        super().__init__()
        self.scenario = scenario

    def __str__(self):
        return (f"The {self.scenario} scenario has no allotments; allotment regrowth delays can only be used with the allotment scenario")

class BerryOffGrid(HarvestException):
    def __init__(self, berry_id, day):
        super().__init__()
        self.berry_id = berry_id
        self.day = day

    def __str__(self):
        return (f"Berry {self.berry_id} was off the grid at the season change on day {self.day}; every berry should have regrown or be waiting to regrow by then")