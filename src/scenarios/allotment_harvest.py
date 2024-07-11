import numpy as np
from src.harvest_model import HarvestModel
from src.agent.harvest_agent import HarvestAgent

class AllotmentHarvest(HarvestModel):
    def __init__(self,num_agents, agent_type,max_episodes,training,write_data,write_norms,file_string=""):
        self.max_width = 8
        super().__init__(num_agents,self.max_width,max_episodes,training,write_data,write_norms,file_string)
        self.num_start_berries = 8
        #allocations is a nested dictionary with allotments for each agent (list of coordinates for max/min width/height) and berry allocation;
        self.allocations = {"agent_0": {
                                "id": 0,
                                "berry_allocation": 6,
                                "allotment": [0,4,0,self.max_height]},
                            "agent_1": {
                                "id": 1,
                                "berry_allocation": 2,
                                "allotment": [4,self.max_width,0,self.max_height]}
                        }
        self.init_agents(self.n_features, agent_type)
        self.init_berries()

    def init_berries(self):
        self.num_berries = 0
        for agent_data in self.allocations.values():
            agent_id = agent_data["id"]
            allotment = agent_data["allotment"]
            berry_allocation = agent_data["berry_allocation"]
            for i in range(berry_allocation):
                b = self.new_berry(allotment[0],allotment[1],allotment[2],allotment[3],agent_id)
                self.place_agent_in_allotment(b)
                self.num_berries += 1
        assert(self.num_berries==self.num_start_berries)
      
    def init_agents(self,n_features, agent_type):
        self.living_agents = []
        for id in range(self.num_agents):
            agent_id = "agent_"+str(id)
            allotment = self.allocations[agent_id]["allotment"]
            n_features = self.get_n_features(allotment[1]-allotment[0],allotment[3]-allotment[2])
            a = HarvestAgent(id,self,agent_type,allotment[0],allotment[1],allotment[2],allotment[3],n_features,self.training,self.epsilon,shared_replay_buffer=self.shared_replay_buffer)
            self.add_agent(a)
        self.num_living_agents = len(self.living_agents)
        self.berry_id = self.num_living_agents + 1
        assert self.num_living_agents == self.num_agents, f"init {self.num_living_agents} instead of {self.num_agents}"
    
    #agents can see their coords and attributes,coords of berries they can reach,coords of other agents+how many days they have to live
    #coords of other agents and rest of the grid is masked
    def observe(self, observer):
        x, y = observer.pos
        observer_features = np.array([x, y, observer.health, observer.berries, observer.days_left_to_live])
        agent_coords = np.array([])
        berry_coords = np.zeros(((observer.max_width-observer.min_width), (observer.max_height-observer.min_height)))
        agent_days_left_to_live = np.array([])
        #agent dictionary is ordered
        for a in self.schedule.agents:
            if a.unique_id == observer.unique_id:
                continue
            if a.agent_type == "berry":
                if a.allocated_agent_id == observer.unique_id:
                    x, y = a.pos
                    berry_coords[x-observer.min_width, y-observer.min_height] = 1
                else:
                    continue
            else:
                #mask coords of other agents
                agent_coords = np.append(agent_coords, 0)
                agent_coords = np.append(agent_coords, 0)
                agent_days_left_to_live = np.append(agent_days_left_to_live, a.days_left_to_live)
        berry_coords = berry_coords.flatten()
        coords = np.append(observer_features, agent_coords)
        coords = np.append(coords, berry_coords)
        observation = np.append(coords, agent_days_left_to_live)
        return observation
    
    def get_n_features(self, max_width=0, max_height=0):
        #agent coords
        n_features = self.num_agents * 2
        #agent's own health and num berries
        n_features += 2
        #feature for each cell that could have berries in
        n_features += (max_width * max_height)
        #feature for how many days each agent has left to live
        n_features += self.num_agents
        return n_features