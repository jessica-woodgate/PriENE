from mesa import Agent

class Berry(Agent):
    """
    Berry object can be foraged by agents; in testing scenarios, a berry can be allocated to a specifc agent or specific part of the grid
    Instance variables:
        agent_type -- type of agent (berry)
        allocated_agent_id -- id of agent allocated to (None for training)
        min/max width/height -- dimensions of grid berry can be assigned to (whole grid for training)
    """
    def __init__(self,unique_id,model,allotment,allotment_id=None):
        super().__init__(unique_id, model)
        self.agent_type = "berry"
        self.foraged = False
        self.allotment_id = allotment_id
        self.min_width = allotment[0]
        self.max_width = allotment[1]
        self.min_height = allotment[2]
        self.max_height = allotment[3]
    
    def step(self):
        pass

    def reset(self):
        self.foraged = False