from src.harvest_model import HarvestModel
from src.harvest_exception import NumBerriesException
from src.harvest_exception import RegrowthDelaysNotSupported

class ColoursHarvest(HarvestModel):
    """
    Colours harvest scenario agents have only access to berries of a particular colour
    Instance variables:
        num_start_berries -- the number of berries initiated at the beginning of an episode
        allocations -- dictionary of colours, the berries of each, and the agent holding it (agents are given colours afresh each episode)
        berries -- list of active berry objects
    """
    SHUFFLES_ALLOCATIONS = True

    def __init__(self,society_mix,num_agents,num_start_berries,agent_type,max_width,max_height,max_episodes,max_days,training,checkpoint_path,write_data,write_norms,filepath="",seed=None,norms_module_type="rule_based",seasons=False,season_length=25,allotment_regrowth_delays=None,start_episode=1):
        if allotment_regrowth_delays is not None:
            raise RegrowthDelaysNotSupported("colours")
        super().__init__(num_agents,max_width,max_height,max_episodes,max_days,training,write_data,write_norms,filepath,seed,norms_module_type,seasons,season_length,start_episode=start_episode)
        self.num_start_berries = num_start_berries
        self.allocations = self._assign_allocations()
        self._init_agents(society_mix, agent_type, checkpoint_path)
        self.berries = self._init_berries()
        self._init_seasons()

    def _assign_allocations(self):
        resources = self._generate_resource_allocations(self.num_agents)
        allocations = {}
        for i in range(self.num_agents):
            key = "allocation_"+str(i)
            #each colour is a group of one agent, so seasons change each colour by one berry
            allocations[key] = {"id": i, "berry_allocation": resources[i], "agent_ids": [i]}
        return allocations

    def _init_berries(self):
        berries = []
        self.num_berries = 0
        allotment = [0,self.max_width,0,self.max_height]
        for allocation_data in self.allocations.values():
            berry_allocation = allocation_data["berry_allocation"]
            for i in range(berry_allocation):
                b = self._new_berry(allotment,allocation_data["id"])
                self._place_agent_in_allotment(b)
                self.num_berries += 1
                berries.append(b)
        if self.num_berries != self.num_start_berries:
            raise NumBerriesException(self.num_start_berries, self.num_berries)
        return berries