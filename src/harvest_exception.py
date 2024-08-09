class HarvestException(Exception):
    def __init__(self):
        super().__init__

class FileExistsException(HarvestException):
    def __init__(self, file_name):
        super().__init__
        self.file_name= file_name
    
    def __str__(self):
        return (f"File {self.file_name} already exists")

class NoBerriesException(HarvestException):
    def __init__(self, agent_id=None, coordinates=None):
        super().__init__
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
        super().__init__
        self.num_expected_agents = num_expected_agents
        self.num_agents = num_agents
    
    def __str__(self):
        return (f"Expected {self.num_expected_agents} agents and got {self.num_agents}")

class AgentTypeException(HarvestException):
    def __init__(self, expected_type, agent_type):
        super().__init__
        self.expected_type = expected_type
        self.agent_type = agent_type
    
    def __str__(self):
        return (f"Expected type {self.expected_type} and got {self.agent_type}")

class OutOfBounds(HarvestException):
    def __init__(self, agent_id, coordinates):
        super().__init__
        self.agent_id = agent_id
        self.coordinates = coordinates
    
    def __str__(self):
        return(f"Coordinates {self.coordinates} are out of bounds for agent {self.agent_id}")
    
class NoEmptyCells(HarvestException):
    def __init__(self):
        pass

    def __str__(self):
        return ("No empty cells in grid")
    
class UnrecognisedPrinciple(HarvestException):
    def __init__(self, principle):
        super().__init__
        self.principle = principle
    
    def __str__(self):
        return(f"Do not recognise principle {self.principle}")
    
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
        super().__init__
        self.num_expected_berries = num_expected_berries
        self.num_berries = num_berries
    
    def __str__(self):
        return (f"Expected {self.num_expected_berries} berries and got {self.num_berries}")
    
class NumFeaturesException(HarvestException):
    def __init__(self, num_expected_features, num_features):
        super().__init__
        self.num_expected_features = num_expected_features
        self.num_features = num_features
    
    def __str__(self):
        return (f"Expected {self.num_expected_features} berries and got {self.num_features}")