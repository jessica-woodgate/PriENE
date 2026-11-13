from src.harvest_model import HarvestModel
from src.harvest_exception import RegrowthDelaysNotSupported

class CapabilitiesHarvest(HarvestModel):
    """
    Capabilities harvest scenario agents have only access to specific berries: some agents are tall, and can access berries on trees; some agents are small and can access berries on the ground
    Instance variables:
        num_start_berries -- the number of berries initiated at the beginning of an episode
        allocations -- dictionary of capability groups, the berries of each, and the agent ids in it (agents are put in groups afresh each episode)
        berries -- list of active berry objects
    """
    SHUFFLES_ALLOCATIONS = True

    def __init__(self,society_mix,num_agents,num_start_berries,num_capabilities,agent_type,max_width,max_height,max_episodes,max_days,training,checkpoint_path,write_data,write_norms,filepath="",seed=None,norms_module_type="rule_based",seasons=False,season_length=25,allotment_regrowth_delays=None,start_episode=1,training_config=None):
        if allotment_regrowth_delays is not None:
            raise RegrowthDelaysNotSupported("capabilities")
        super().__init__(num_agents,max_width,max_height,max_episodes,max_days,training,write_data,write_norms,filepath,seed,norms_module_type,seasons,season_length,start_episode=start_episode,training_config=training_config)
        self.num_start_berries = num_start_berries
        self.num_capabilities = num_capabilities
        self.allocations = self._assign_allocations()
        self._init_agents(society_mix,agent_type, checkpoint_path)
        self.berries = self._init_berries()
        self._init_seasons()

    def _assign_allocations(self):
        resources = self._generate_resource_allocations(self.num_agents)
        allocations = {}
        for i, agent_ids in enumerate(self._consecutive_groups(range(self.num_agents), self.num_capabilities)):
            allocations["allocation_"+str(i)] = {"id": i, "berry_allocation": sum(resources[agent_id] for agent_id in agent_ids), "agent_ids": agent_ids}
        return allocations