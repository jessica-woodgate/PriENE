from src.harvest_model import HarvestModel
from ..agent.harvest_agent import HarvestAgent
from src.harvest_exception import NumBerriesException
from src.harvest_exception import NumAgentsException
from src.harvest_exception import NoAllocationException

class CapabilitiesHarvest(HarvestModel):
    """
    Capabilities harvest scenario agents have only access to specific berries: some agents are tall, and can access berries on trees; some agents are small and can access berries on the ground
    Instance variables:
        num_start_berries -- the number of berries initiated at the beginning of an episode
        allocations -- dictionary of agent ids and the berries assigned to that agent
        berries -- list of active berry objects
    """
    def __init__(self,num_agents,num_start_berries,num_capabilities,agent_type,max_width,max_height,max_episodes,max_days,training,checkpoint_path,write_data,write_norms,filepath=""):
        super().__init__(num_agents,max_width,max_height,max_episodes,max_days,training,write_data,write_norms,filepath)
        self.num_start_berries = num_start_berries
        self.num_capabilities = num_capabilities
        self.allocations = self._assign_allocations()
        self._init_agents(agent_type, checkpoint_path)
        self.berries = self._init_berries()
    
    def _assign_allocations(self):
        resources = self._generate_resource_allocations(self.num_agents)
        allocations = {}
        group_size = self.num_agents // self.num_capabilities
        remainder = self.num_agents % self.num_capabilities
        start_index = 0
        for i in range(self.num_capabilities):
            end_index = start_index + group_size + (1 if i < remainder else 0)
            current_resource = sum(resources[start_index:end_index])
            agent_ids = list(range(start_index, end_index))
            start_index = end_index
            key = "allocation_"+str(i)
            allocations[key] = {"id": i, "berry_allocation": current_resource, "agent_ids": agent_ids}
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
    
    def _init_agents(self, agent_type, checkpoint_path):
        self.living_agents = []
        allotment = [0,self.max_width,0,self.max_height]
        for id in range(self.num_agents):
            allocation_id = self._get_allocation_id(id)
            a = HarvestAgent(id,self,agent_type,allotment,self.training,checkpoint_path,self.epsilon,self.write_norms,shared_replay_buffer=self.shared_replay_buffer,allocation_id=allocation_id)
            self._add_agent(a)
        self.num_living_agents = len(self.living_agents)
        self.berry_id = self.num_living_agents + 1
        if self.num_living_agents != self.num_agents:
            raise NumAgentsException(self.num_agents, self.num_living_agents)
    
    def _get_allocation_id(self, agent_id):
        for allocation_id, allocation_data in self.allocations.items():
            if agent_id in allocation_data["agent_ids"]:
                return allocation_id
        raise NoAllocationException(agent_id)