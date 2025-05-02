from src.harvest_model import HarvestModel
from src.harvest_exception import NumBerriesException

class ColoursHarvest(HarvestModel):
    """
    Colours harvest scenario agents have only access to berries of a particular colour
    Instance variables:
        num_start_berries -- the number of berries initiated at the beginning of an episode
        allocations -- dictionary of agent ids and the berries assigned to that agent
        berries -- list of active berry objects
    """
    def __init__(self,society_mix,num_agents,num_start_berries,agent_type,max_width,max_height,max_episodes,max_days,training,checkpoint_path,write_data,write_norms,filepath=""):
        super().__init__(num_agents,max_width,max_height,max_episodes,max_days,training,write_data,write_norms,filepath)
        self.num_start_berries = num_start_berries
        print(num_start_berries)
        self.allocations = self._assign_allocations()
        print(self.allocations.values())
        print(self.allocations.keys())
        self._init_agents(society_mix, agent_type, checkpoint_path)
        self.berries = self._init_berries()
    
    def _assign_allocations(self):
        resources = self._generate_resource_allocations(self.num_agents)
        print(resources)
        allocations = {}
        for i in range(self.num_agents):
            key = "allocation_"+str(i)
            allocations[key] = {"id": i, "berry_allocation": resources[i]}
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
    
    def _init_agents(self, society_mix, agent_type, checkpoint_path):
        self.living_agents = []
        allotment = [0,self.max_width,0,self.max_height]
        if society_mix == "homogeneous":
            for i in range(self.num_agents):
                self._add_agent(i, agent_type, allotment, checkpoint_path, allocation_id=f"allocation_{i}")
        else:
            assert self.num_agents%2 == 0
            half_pop = int(self.num_agents/2)
            for i in range(half_pop):
                self._add_agent(i, agent_type, allotment, checkpoint_path, allocation_id=f"allocation_{i}")
            for i in range(half_pop):
                self._add_agent(i+half_pop, "baseline", allotment, checkpoint_path, allocation_id=f"allocation_{i}")
        assert self.num_agents == len(self.living_agents)
        self.berry_id = len(self.living_agents) + 1