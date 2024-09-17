from src.harvest_model import HarvestModel

class BasicHarvest(HarvestModel):
    """
    Basic harvest scenario for training agents in; any agent can access any berry
    Instance variables:
        num_start_berries -- the number of berries initiated at the beginning of an episode
        berries -- list of active berry objects
    """
    def __init__(self,num_agents,num_start_berries,agent_type,max_width,max_height,max_episodes,max_days,training,checkpoint_path,write_data,write_norms,filepath=""):
        super().__init__(num_agents,max_width,max_height,max_episodes,max_days,training,write_data,write_norms,filepath)
        self.num_start_berries = num_start_berries
        self._init_agents(agent_type, checkpoint_path)
        self.berries = self._init_berries()

    def _init_berries(self):
        berries = []
        self.num_berries = 0
        for i in range(self.num_start_berries):
            b = self._new_berry(0, self.max_width, 0, self.max_height)
            self._place_agent_in_allotment(b)
            self.num_berries += 1
            berries.append(b)
        assert(self.num_berries == self.num_start_berries)
        return berries