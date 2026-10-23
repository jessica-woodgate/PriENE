from src.data_handling.norm_processing import NormProcessing

NORM = "IF,low health,THEN,eat"


def norm_entry(numerosity, **held):
    return {NORM: {"reward": 1, "numerosity": numerosity, "fitness": 1, "adoption": 4, **held}}


def test_merge_norms_summarises_days_held_across_episodes(tmp_path):
    data = {"1": [norm_entry(2, first_day_held=10, last_day_held=50, days_held=40)],
            "2": [norm_entry(3, first_day_held=20, last_day_held=200, days_held=150)]}
    merged = NormProcessing()._merge_norms(data, str(tmp_path / "out"))[NORM]
    assert merged["numerosity"] == 5 and merged["num_instances_across_episodes"] == 2
    assert merged["total_days_held"] == 190
    assert merged["mean_days_held"] == 95
    assert merged["mean_first_day_held"] == 15
    assert merged["mean_last_day_held"] == 125
    assert not {"instances_held", "sum_first_day_held", "sum_last_day_held"} & merged.keys()
    cooperative = NormProcessing()._get_cooperative_norms({"IF,high health,THEN,throw": merged}, str(tmp_path / "out"))
    assert cooperative.loc[0, "mean_days_held"] == 95


def test_merge_norms_without_days_held_fields(tmp_path):
    """Emerged norms files from runs before held days were tracked"""
    merged = NormProcessing()._merge_norms({"1": [norm_entry(2)], "2": [norm_entry(3)]}, str(tmp_path / "out"))[NORM]
    assert merged == {"reward": 2, "numerosity": 5, "fitness": 2, "adoption": 8, "num_instances_across_episodes": 2}
