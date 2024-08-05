from src.harvest_model import HarvestModel
from src.agent.harvest_agent import HarvestAgent

class AllotmentHarvest(HarvestModel):
    def __init__(self,num_agents,num_start_berries,agent_type,max_width,max_height,max_episodes,training,write_data,write_norms,file_string=""):
        super().__init__(num_agents,max_width,max_height,max_episodes,training,write_data,write_norms,file_string)
        self.num_start_berries = num_start_berries
        #allocations is a nested dictionary with allotments for each agent (list of coordinates for max/min width/height) and berry allocation;
        allocation_interval = int(max_width / num_agents)
        # self.allocations = {"agent_0": {
        #                         "id": 0,
        #                         "berry_allocation": 5,
        #                         "allotment": [0,allocation_interval,0,self.max_height]},
        #                     "agent_1": {
        #                         "id": 1,
        #                         "berry_allocation": 2,
        #                         "allotment": [allocation_interval,allocation_interval*2,0,self.max_height]},
        #                     "agent_2": {
        #                         "id": 2,
        #                         "berry_allocation": 3,
        #                         "allotment": [allocation_interval*2,allocation_interval*3,0,self.max_height]},
        #                     "agent_3": {
        #                         "id": 3,
        #                         "berry_allocation": 2,
        #                         "allotment": [allocation_interval*3,self.max_width,0,self.max_height]},
        #                     }
        self.allocations = {"agent_0": {
                                "id": 0,
                                "berry_allocation": 3,
                                "allotment": [0,4,0,self.max_height]},
                            "agent_1": {
                                "id": 1,
                                "berry_allocation": 1,
                                "allotment": [4,self.max_width,0,self.max_height]}
                        }
        self.init_agents(agent_type)
        self.berries = self.init_berries()

    def init_berries(self):
        self.num_berries = 0
        berries = []
        for agent_data in self.allocations.values():
            agent_id = agent_data["id"]
            allotment = agent_data["allotment"]
            berry_allocation = agent_data["berry_allocation"]
            for i in range(berry_allocation):
                b = self.new_berry(allotment[0],allotment[1],allotment[2],allotment[3],agent_id)
                self.place_agent_in_allotment(b)
                self.num_berries += 1
                berries.append(b)
        assert(self.num_berries==self.num_start_berries)
        return berries
      
    def init_agents(self, agent_type):
        self._living_agents = []
        for id in range(self._num_agents):
            agent_id = "agent_"+str(id)
            allotment = self.allocations[agent_id]["allotment"]
            a = HarvestAgent(id,self,agent_type,self._max_days,allotment[0],allotment[1],allotment[2],allotment[3],self.training,self.epsilon,self.write_norms,shared_replay_buffer=self.shared_replay_buffer)
            self.add_agent(a)
        self._num_living_agents = len(self._living_agents)
        self.berry_id = self._num_living_agents + 1
        assert self._num_living_agents == self._num_agents, f"init {self.num_living_agents} instead of {self._num_agents}"