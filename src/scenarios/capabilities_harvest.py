import numpy as np
from src.harvest_model import HarvestModel

class CapabilitiesHarvest(HarvestModel):
    def __init__(self,num_agents,agent_type,max_episodes,training,write_data,write_norms,file_string=""):
        self.max_width = 4
        super().__init__(num_agents,self.max_width,max_episodes,training,write_data,write_norms,file_string)
        self.num_start_berries = 8
        self.allocations = {"agent_0": {
                                "id": 0,
                                "berry_allocation": 6},
                            "agent_1": {
                                "id": 1,
                                "berry_allocation": 2}
                            }
        self.init_agents(self.n_features, agent_type)
        self.berries = self.init_berries()

    def init_berries(self):
        berries = []
        self.num_berries = 0
        for agent_data in self.allocations.values():
            berry_allocation = agent_data["berry_allocation"]
            for i in range(berry_allocation):
                b = self.new_berry(0,self.max_width,0,self.max_height,agent_data["id"])
                self.place_agent_in_allotment(b)
                self.num_berries += 1
                berries.append(b)
        assert(self.num_berries == self.num_start_berries)
        return berries
    
    #agents can see their coords and attributes,coords of berries they can reach,coords of other agents+how many days they have to live
    def observe(self, observer):
        x, y = observer.pos
        observer_features = np.array([x, y, observer.health, observer.berries, observer.days_left_to_live])
        agent_coords = np.array([])
        berry_coords = np.zeros((self.max_width, self.max_height))
        agent_days_left_to_live = np.array([])
        for a in self.schedule.agents:
            if a.unique_id == observer.unique_id:
                continue
            if a.agent_type == "berry":
                if a.allocated_agent_id == observer.unique_id:
                    x, y = a.pos
                    berry_coords[x, y] = 1
                else:
                    continue
            elif a.done == False:
                #observe agent's coords and how many days they have left
                x, y = a.pos
                agent_coords = np.append(agent_coords, x)
                agent_coords = np.append(agent_coords, y)
                agent_days_left_to_live = np.append(agent_days_left_to_live, a.days_left_to_live)
            elif a.done == True:
                #if dead, observe 0s
                agent_coords = np.append(agent_coords, 0)
                agent_coords = np.append(agent_coords, 0)
                agent_days_left_to_live = np.append(agent_days_left_to_live, 0)
        berry_coords = berry_coords.flatten()
        coords = np.append(observer_features, agent_coords)
        coords = np.append(coords, berry_coords)
        observation = np.append(coords, agent_days_left_to_live)
        assert len(observation) == self.n_features
        return observation