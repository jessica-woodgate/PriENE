from src.scenarios.colours_harvest import ColoursHarvest
from src.harvest_exception import InitialHealthsException

class HealthHarvest(ColoursHarvest):
    """
    Health harvest scenario is the colours harvest scenario with agents starting each episode with different levels of health,
    to test whether agents prioritise those in greater need rather than those with less access to resources
    Instance variables:
        initial_healths -- health each agent starts every episode with, indexed by agent id
    """
    def __init__(self,society_mix,num_agents,num_start_berries,agent_type,max_width,max_height,max_episodes,max_days,training,checkpoint_path,write_data,write_norms,initial_healths,filepath="",seed=None,norms_module_type="rule_based"):
        if len(initial_healths) != num_agents or any(health <= 0 for health in initial_healths):
            raise InitialHealthsException(num_agents, initial_healths)
        self.initial_healths = list(initial_healths)
        super().__init__(society_mix,num_agents,num_start_berries,agent_type,max_width,max_height,max_episodes,max_days,training,checkpoint_path,write_data,write_norms,filepath,seed,norms_module_type)
        self._apply_initial_healths()

    def _apply_initial_healths(self):
        """
        Set each agent's start health, which agent.reset() restores at the start of every episode,
        and its current health for the first episode
        """
        for a in self.schedule.agents:
            if a.agent_type != "berry":
                a.start_health = self.initial_healths[a.unique_id]
                a.health = a.start_health
                a.days_left_to_live = a.get_days_left_to_live()
                a.total_days_left_to_live = a.days_left_to_live
