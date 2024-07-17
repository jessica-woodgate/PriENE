from mesa import Agent

class Berry(Agent):
    def __init__(self,unique_id,model,min_width,max_width,min_height,max_height,allocated_agent_id=None):
        super().__init__(unique_id, model)
        self.agent_type = "berry"
        self.foraged = False
        #self.marked = False
        self.allocated_agent_id = allocated_agent_id
        self.min_width = min_width
        self.max_width = max_width
        self.min_height = min_height
        self.max_height = max_height
    
    def step(self):
        pass

    def reset(self):
        self.foraged = False
        #self.marked = False