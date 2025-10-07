from src.harvest_model import HarvestModel
from ..agent.harvest_agent import HarvestAgent
from src.harvest_exception import NumBerriesException
from src.harvest_exception import NumAgentsException
from src.harvest_exception import NumAllotmentsException
from src.harvest_exception import NoAllocationException

class AllotmentHarvest(HarvestModel):
    """
    Allotment harvest scenario agents have only access to specific parts of the grid within which different amounts of berries grow
        num_start_berries -- the number of berries initiated at the beginning of an episode
        allocations -- dictionary of agent ids, the part of the grid they have access to, and the berries assigned to that agent
        berries -- list of active berry objects
    """
    def __init__(self,society_mix,num_agents,num_start_berries,num_allotments,agent_type,max_width,max_height,max_episodes,max_days,training,checkpoint_path,write_data,write_norms,filepath=""):
        super().__init__(num_agents,max_width,max_height,max_episodes,max_days,training,write_data,write_norms,filepath)
        self.num_start_berries = num_start_berries
        if num_allotments > num_agents or num_allotments < 1:
            raise NumAllotmentsException(num_agents, num_allotments)
        self.num_allotments = num_allotments
        allotment_interval = int(max_width // num_allotments)
        self.allocations = self._assign_allocations(allotment_interval)
        print(self.allocations.values())
        print(self.allocations.keys())
        self._init_agents(society_mix, agent_type, checkpoint_path)
        self.berries = self._init_berries()

    def _assign_allocations(self, allotment_interval):
        resources = self._generate_resource_allocations(self.num_agents)
        allocations = {}
        allotment_start = 0
        allotment_end = allotment_interval
        group_size = self.num_agents // self.num_allotments
        remainder = self.num_agents % self.num_allotments
        start_index = 0
        for i in range(self.num_allotments):
            end_index = start_index + group_size + (1 if i < remainder else 0)
            current_resource = sum(resources[start_index:end_index])
            agent_ids = list(range(start_index, end_index))
            start_index = end_index
            key = "allocation_"+str(i)
            allocations[key] = {"id": i, "berry_allocation": current_resource, "allotment":[allotment_start,allotment_end,0,self.max_height], "agent_ids": agent_ids}
            allotment_start += allotment_interval
            allotment_end += allotment_interval
        return allocations

    def _init_berries(self):
        self.num_berries = 0
        berries = []
        for allocation_data in self.allocations.values():
            for i in range(allocation_data["berry_allocation"]):
                b = self._new_berry(allocation_data["allotment"],allocation_data["id"])
                self._place_agent_in_allotment(b)
                self.num_berries += 1
                berries.append(b)
        if self.num_berries != self.num_start_berries:
            raise NumBerriesException(self.num_start_berries, self.num_berries)
        return berries
      
    def _init_agents(self, society_mix, agent_type, checkpoint_path):
        self.living_agents = []
        if society_mix == "homogeneous":
            for id in range(self.num_agents):
                allocation_id = self._get_allocation_id(id)
                allotment = self.allocations[allocation_id]["allotment"]
                self._add_agent(id, agent_type, allotment, checkpoint_path, allocation_id=allocation_id)
        else:
            assert self.num_agents%2 == 0
            half_pop = int(self.num_agents/2)
            for id in range(half_pop):
                allocation_id = self._get_allocation_id(id)
                allotment = self.allocations[allocation_id]["allotment"]
                self._add_agent(id, agent_type, allotment, checkpoint_path, allocation_id=allocation_id)
            for id in range(half_pop):
                allocation_id = self._get_allocation_id(id)
                allotment = self.allocations[allocation_id]["allotment"]
                self._add_agent(id+half_pop, "baseline", allotment, checkpoint_path, allocation_id=allocation_id)
        self.num_living_agents = len(self.living_agents)
        self.berry_id = self.num_living_agents + 1
        if self.num_living_agents != self.num_agents:
            raise NumAgentsException(self.num_agents, self.num_living_agents)
    
    def _get_allocation_id(self, agent_id):
        for allocation_id, allocation_data in self.allocations.items():
            if agent_id in allocation_data["agent_ids"]:
                return allocation_id
        raise NoAllocationException(agent_id)