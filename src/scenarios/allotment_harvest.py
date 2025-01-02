from src.harvest_model import HarvestModel
from ..agent.harvest_agent import HarvestAgent
from src.harvest_exception import NumBerriesException
from src.harvest_exception import NumAgentsException
from src.harvest_exception import NumAllotmentsException
from src.harvest_exception import NoAllotmentException

class AllotmentHarvest(HarvestModel):
    """
    Allotment harvest scenario agents have only access to specific parts of the grid within which different amounts of berries grow
        num_start_berries -- the number of berries initiated at the beginning of an episode
        allocations -- dictionary of agent ids, the part of the grid they have access to, and the berries assigned to that agent
        berries -- list of active berry objects
    """
    def __init__(self,num_agents,num_start_berries,num_allotments,agent_type,aggregation,max_width,max_height,max_episodes,max_days,training,checkpoint_path,write_data,write_norms,filepath=""):
        super().__init__(num_agents,max_width,max_height,max_episodes,max_days,training,write_data,write_norms,filepath)
        self.num_start_berries = num_start_berries
        if num_allotments > num_agents or num_allotments < 1:
            raise NumAllotmentsException(num_agents, num_allotments)
        self.num_allotments = num_allotments
        allotment_interval = int(max_width // num_allotments)
        self.allocations = self._assign_allocations(allotment_interval)
        self._init_agents(agent_type, aggregation, checkpoint_path)
        self.berries = self._init_berries()

    def _assign_allocations(self, allotment_interval):
        resources = self._generate_resource_allocations(self.num_agents)
        allocations = {}
        allotment_start = 0
        allotment_end = allotment_interval
        group_size = self.num_agents // self.num_allotments
        print("group size", group_size)
        remainder = self.num_agents % self.num_allotments
        start_index = 0
        for i in range(self.num_allotments):
            end_index = start_index + group_size + (1 if i < remainder else 0)
            current_resource = sum(resources[start_index:end_index])
            print("start", start_index, "end", end_index)
            agent_ids = list(range(start_index, end_index))
            print("agent_ids", agent_ids)
            start_index = end_index
            key = "allotment_"+str(i)
            allocations[key] = {"id": i, "berry_allocation": current_resource, "allotment":[allotment_start,allotment_end,0,self.max_height], "agent_ids": agent_ids}
            allotment_start += allotment_interval
            allotment_end += allotment_interval
        print("allocations", allocations)
        return allocations

    def _init_berries(self):
        self.num_berries = 0
        berries = []
        for allotment_data in self.allocations.values():
            allotment_id = allotment_data["id"]
            allotment = allotment_data["allotment"]
            berry_allocation = allotment_data["berry_allocation"]
            for i in range(berry_allocation):
                b = self._new_berry(allotment,allotment_id)
                self._place_agent_in_allotment(b)
                self.num_berries += 1
                berries.append(b)
        if self.num_berries != self.num_start_berries:
            raise NumBerriesException(self.num_start_berries, self.num_berries)
        return berries
      
    def _init_agents(self, agent_type, aggregation, checkpoint_path):
        self.living_agents = []
        for id in range(self.num_agents):
            agent_id = "agent_"+str(id)
            allotment_id = self._get_allotment_id(id)
            allotment = self.allocations[allotment_id]["allotment"]
            print("agent", id, "allotment id", allotment_id, "allotment", allotment)
            a = HarvestAgent(id,self,agent_type,aggregation,allotment,self.training,checkpoint_path,self.epsilon,self.write_norms,shared_replay_buffer=self.shared_replay_buffer,allotment_id=allotment_id)
            self._add_agent(a)
        self.num_living_agents = len(self.living_agents)
        self.berry_id = self.num_living_agents + 1
        if self.num_living_agents != self.num_agents:
            raise NumAgentsException(self.num_agents, self.num_living_agents)
    
    def _get_allotment_id(self, agent_id):
        for allotment_id, allocation_data in self.allocations.items():
            if agent_id in allocation_data["agent_ids"]:
                return allotment_id
        raise NoAllotmentException(agent_id)