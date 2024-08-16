from mesa import Agent

class Berry(Agent):
    """
    Berry object can be foraged by agents; in testing scenarios, a berry can be allocated to a specifc agent or specific part of the grid
    Instance variables:
        agent_type -- type of agent (berry)
        allocated_agent_id -- id of agent allocated to (None for training)
        min/max width/height -- dimensions of grid berry can be assigned to (whole grid for training)
    """
    def __init__(self,unique_id,model,min_width,max_width,min_height,max_height,allocated_agent_id=None):
        super().__init__(unique_id, model)
        self.agent_type = "berry"
        self.foraged = False
        self.allocated_agent_id = allocated_agent_id
        self.min_width = min_width
        self.max_width = max_width
        self.min_height = min_height
        self.max_height = max_height
    
    def step(self):
        pass

    def reset(self):
        self.foraged = False