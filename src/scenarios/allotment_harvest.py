from src.harvest_model import HarvestModel
from src.agent.harvest_agent import HarvestAgent
from src.harvest_exception import NumAgentsException
from src.harvest_exception import NumBerriesException

class AllotmentHarvest(HarvestModel):
    """
    Allotment harvest scenario agents have only access to specific parts of the grid within which different amounts of berries grow
        num_start_berries -- the number of berries initiated at the beginning of an episode
        allocations -- dictionary of agent ids, the part of the grid they have access to, and the berries assigned to that agent
        berries -- list of active berry objects
    """
    def __init__(self,num_agents,num_start_berries,agent_type,max_width,max_height,max_episodes,max_days,training,checkpoint_path,write_data,write_norms,filepath=""):
        super().__init__(num_agents,max_width,max_height,max_episodes,max_days,training,write_data,write_norms,filepath)
        self.num_start_berries = num_start_berries
        allotment_interval = int(max_width / num_agents)
        self.allocations = self._assign_allocations(allotment_interval)
        self._init_agents(agent_type, checkpoint_path)
        self.berries = self._init_berries()

    def _assign_allocations(self, allotment_interval):
        resources = self._generate_resource_allocations(self.num_agents)
        allocations = {}
        allotment_start = 0
        allotment_end = allotment_interval
        for i in range(self.num_agents):
            key = "agent_"+str(i)
            allocations[key] = {"id": i, "berry_allocation": resources[i], "allotment":[allotment_start,allotment_end,0,self.max_height]}
            allotment_start += allotment_interval
            allotment_end += allotment_interval
        return allocations

    def _init_berries(self):
        self.num_berries = 0
        berries = []
        for agent_data in self.allocations.values():
            agent_id = agent_data["id"]
            allotment = agent_data["allotment"]
            berry_allocation = agent_data["berry_allocation"]
            for i in range(berry_allocation):
                b = self._new_berry(allotment[0],allotment[1],allotment[2],allotment[3],agent_id)
                self._place_agent_in_allotment(b)
                self.num_berries += 1
                berries.append(b)
        if self.num_berries != self.num_start_berries:
            raise NumBerriesException(self.num_start_berries, self.num_berries)
        return berries
      
    def _init_agents(self, agent_type, checkpoint_path):
        self.living_agents = []
        for id in range(self.num_agents):
            agent_id = "agent_"+str(id)
            allotment = self.allocations[agent_id]["allotment"]
            a = HarvestAgent(id,self,agent_type,self.max_days,allotment[0],allotment[1],allotment[2],allotment[3],self.training,checkpoint_path,self.epsilon,self.write_norms,shared_replay_buffer=self.shared_replay_buffer)
            self._add_agent(a)
        self.num_living_agents = len(self.living_agents)
        self.berry_id = self.num_living_agents + 1
        if self.num_living_agents != self.num_agents:
            raise NumAgentsException(self.num_agents, self.num_living_agents)