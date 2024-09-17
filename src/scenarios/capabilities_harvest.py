from src.harvest_model import HarvestModel
from src.harvest_exception import NumBerriesException

class CapabilitiesHarvest(HarvestModel):
    """
    Capabilities harvest scenario agents have only access to specific berries: some agents are tall, and can access berries on trees; some agents are small and can access berries on the ground
    Instance variables:
        num_start_berries -- the number of berries initiated at the beginning of an episode
        allocations -- dictionary of agent ids and the berries assigned to that agent
        berries -- list of active berry objects
    """
    def __init__(self,num_agents,num_start_berries,agent_type,max_width,max_height,max_episodes,max_days,training,checkpoint_path,write_data,write_norms,filepath=""):
        super().__init__(num_agents,max_width,max_height,max_episodes,max_days,training,write_data,write_norms,filepath)
        self.num_start_berries = num_start_berries
        self.allocations = self._assign_allocations()
        self._init_agents(agent_type, checkpoint_path)
        self.berries = self._init_berries()
    
    def _assign_allocations(self):
        resources = self._generate_resource_allocations(self.num_agents)
        allocations = {}
        for i in range(self.num_agents):
            key = "agent_"+str(i)
            allocations[key] = {"id": i, "berry_allocation": resources[i]}
        return allocations

    def _init_berries(self):
        berries = []
        self.num_berries = 0
        for agent_data in self.allocations.values():
            berry_allocation = agent_data["berry_allocation"]
            for i in range(berry_allocation):
                b = self._new_berry(0,self.max_width,0,self.max_height,agent_data["id"])
                self._place_agent_in_allotment(b)
                self.num_berries += 1
                berries.append(b)
        if self.num_berries != self.num_start_berries:
            raise NumBerriesException(self.num_start_berries, self.num_berries)
        return berries