import numpy as np
from src.harvest_model import HarvestModel

class BasicHarvest(HarvestModel):
    def __init__(self,num_agents,num_start_berries,agent_type,max_width,max_height,max_episodes,training,write_data,write_norms,file_string=""):
        super().__init__(num_agents,max_width,max_height,max_episodes,training,write_data,write_norms,file_string)
        self.num_start_berries = num_start_berries
        self.init_agents(agent_type)
        self.berries = self.init_berries()

    def init_berries(self):
        berries = []
        self.num_berries = 0
        for i in range(self.num_start_berries):
            b = self.new_berry(0, self.max_width, 0, self.max_height)
            self.place_agent_in_allotment(b)
            self.num_berries += 1
            berries.append(b)
        assert(self.num_berries == self.num_start_berries)
        return berries