"""
Names run.py gives a harvest run's outputs and, with --output-root, its folder's condition; shared with the experiment runner
(src/experiments.py), which must know where each run it launches will be.
"""


def scenario_file_string(scenario, seasons, allotment_regrowth_delays=None, mixed_allotments="split", society_mix="homogeneous"):
    """
    scenario part of output filenames; options that are on are added after the scenario so runs don't overwrite each other. Allotment
    is always "_throwbetween" (agents can throw berries to other allotments), apart from runs made when they could only throw within one
    """
    file_scenario = scenario
    if seasons:
        file_scenario += "_seasons"
    if allotment_regrowth_delays is not None:
        file_scenario += "_regrow" + "-".join(str(delay) for delay in allotment_regrowth_delays)
    if society_mix != "homogeneous":
        file_scenario += "_mixed"
        if scenario == "allotment" and mixed_allotments == "separate":
            file_scenario += "_separate"
    if scenario == "allotment":
        file_scenario += "_throwbetween"
    return file_scenario


def harvest_condition(scenario, seasons, season_length, num_allocations, mixed_allotments, society_mix, num_agents, max_days,
                      commons_parameters=None, config_name=None):
    """
    A harvest run's condition, as its folder (--output-root) names it: the scenario and its options, e.g. colours_seasons_len25_4agents_200days
    """
    parts = [scenario_file_string(scenario, seasons, None, mixed_allotments, society_mix)]
    if seasons:
        parts.append(f"len{season_length}")
    if scenario in ["allotment", "capabilities"]:
        parts.append(f"{num_allocations}groups")
    if commons_parameters:
        parts.append(f"K{commons_parameters['carrying_capacity']}_r{commons_parameters['growth_rate']}_R0{commons_parameters['initial_stock']}")
    parts += [f"{num_agents}agents", f"{max_days}days"]
    if config_name:
        parts.append(config_name)
    return "_".join(parts)
