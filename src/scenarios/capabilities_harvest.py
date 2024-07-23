from src.harvest_model import HarvestModel

class CapabilitiesHarvest(HarvestModel):
    def __init__(self,num_agents,num_start_berries,agent_type,max_width,max_height,max_episodes,training,write_data,write_norms,file_string=""):
        super().__init__(num_agents,max_width,max_height,max_episodes,training,write_data,write_norms,file_string)
        self.num_start_berries = num_start_berries
        self.allocations = {"agent_0": {
                                "id": 0,
                                "berry_allocation": 5},
                            "agent_1": {
                                "id": 1,
                                "berry_allocation": 2},
                            "agent_2": {
                                "id": 2,
                                "berry_allocation": 3},
                            "agent_3": {
                                "id": 3,
                                "berry_allocation": 2},
                            }
        self.init_agents(agent_type)
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