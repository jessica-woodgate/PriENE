class HarvestException(Exception):
    def __init__(self):
        super().__init__

class FileExistsException(HarvestException):
    def __init__(self, file_name):
        super().__init__
        self.file_name= file_name
    
    def __str__(self):
        return ("File {f} already exists".format(f=self.file_name))

class NoBerriesException(HarvestException):
    def __init__(self, agent_id=None, coordinates=None):
        super().__init__
        self.agent_id = agent_id
        self.coordinates = coordinates
    
    def __str__(self):
        if self.coordinates == None:
            return ("Agent {a} has found no berries in the grid".format(a=self.agent_id))
        elif self.agent_id == None:
            return ("Coordinates {c} has no berries".format(c=self.coordinates))
        else:
            return ("Agent {a} trying to forage at {c} which has no berry".format(a=self.agent_id, c=self.coordinates))
        
class NumAgentsException(HarvestException):
    def __init__(self, num_agents, num_expected_agents):
        super().__init__
        self.num_agents = num_agents
        self.num_expected_agents = num_expected_agents
    
    def __str__(self):
        return ("Expected {n} agents and got {m}".format(n=self.num_agents, m=self.num_expected_agents))

class AgentTypeException(HarvestException):
    def __init__(self, agent_type, expected_type):
        super().__init__
        self.agent_type = agent_type
        self.expected_type = expected_type
    
    def __str__(self):
        return (f"Expected type {self.expected_type} and got {self.agent_type}")

class OutOfBounds(HarvestException):
    def __init__(self, agent_id, coordinates):
        super().__init__
        self.agent_id = agent_id
        self.coordinates = coordinates
    
    def __str__(self):
        return("Coordinates {c} are out of bounds for agent {a}".format(c=self.coordinates,a=self.agent_id))
    
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
        return("Do not recognise principle {p}".format(p=self.principle))
    
class NoPathFound(HarvestException):
    def __init__(self, agent_id, agent_coordinates, berry_coordinates):
        self.agent_id = agent_id
        self.agent_coordinates = agent_coordinates
        self.berry_coordinates = berry_coordinates

    def __str__(self):
        return ("Agent {a} couldn't find a path from {m} to {n}".format(a=self.agent_id,m=self.agent_coordinates,n=self.berry_coordinates))

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
        return ("Expected {n} berries and got {m}".format(n=self.num_berries, m=self.num_expected_berries))