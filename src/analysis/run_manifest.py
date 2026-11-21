"""
A run manifest: one row per file every run of an experiment wrote (runs made with run.py's --output-root, e.g. by run.py experiments),
saying which run it came from: train or test, scenario, society, training seed, the reward config the agents were trained with and its
sanction settings. Report files have the same names in every run (e.g. model_episode_reports_train_basic_maximin.csv whatever the seed or
rewards), and a report's seed column is the run's own seed (a test run's is the test seed, the same for every training seed), so the
manifest is how files are told apart once they leave their folders.

    python -m src.analysis.run_manifest runs/reward_trials            # writes runs/reward_trials/manifest.csv
    python -m src.analysis.run_manifest runs/reward_trials --out manifest.csv
    python -m src.analysis.run_manifest runs/reward_trials --export runs/reward_trials/results   # and every file by reward, seed and society

Everything comes from each run's run.json (src/run_record.py), not from folder names. A test run's reward config is its training run's
(the training run whose checkpoint it tested).
"""
import argparse
import glob
import json
import os
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN_FILE = "run.json"
COLUMNS = ["file", "file_type", "agent_id", "mode", "scenario", "condition", "society", "training_seed", "test_seed", "reward_config",
           "sanction", "positive_sanction_only", "death_reward", "throw_reward", "forage_reward", "eat_reward", "ethics_rewards", "status",
           "learning", "last_episode", "run_folder"]
#what each file is, by its name
FILE_TYPES = [("model_episode_reports_", "model_episode_report"), ("fm_model_episode_reports_", "model_episode_report"),
              ("agent_reports_", "agent_report"), ("fm_agent_reports_", "agent_report"), ("agent_norms_", "agent_norms"),
              ("q_model_variables.keras", "q_network"), ("target_model_variables.keras", "target_network"),
              ("optimiser.npz", "optimiser_state"), ("replay_buffer.npz", "replay_buffer"), ("training_state.json", "training_state"),
              ("progress.json", "checkpoint_progress"), ("settings.json", "checkpoint_settings"), ("agent_types.json", "agent_types"),
              ("run.json", "run_record"), ("status.json", "run_status"), ("run.log", "run_log"), ("convergence_", "convergence")]


def file_type(name):
    for pattern, kind in FILE_TYPES:
        if name.startswith(pattern) or name == pattern:
            return kind
    if name.endswith("_emerged_norms.json"):
        return "emerged_norms"
    return "other"


def read_json(path):
    with open(path) as file:
        return json.load(file)


def reward_config_name(record):
    """The training config file a run was given (--training-config), or the scenario's own (harvest, or fruit_market) if none"""
    command = record.get("command", [])
    if "--training-config" in command:
        return os.path.splitext(os.path.basename(command[command.index("--training-config") + 1]))[0]
    return "fruit_market" if record["configuration"].get("scenario") == "fruit_market" else "harvest"


def training_record(record, records_by_folder):
    """A test run's training run: the run folder holding the checkpoint it tested (None if it isn't one of the runs read)"""
    checkpoint = record["configuration"].get("checkpoint")
    if not checkpoint:
        return None
    path = os.path.abspath(checkpoint)
    while path != os.path.dirname(path):
        if path in records_by_folder:
            return records_by_folder[path]
        path = os.path.dirname(path)
    return None


def learning(folder):
    """
    A training run's learning, from its convergence file (run.py --until-converged, src/convergence.py): "converged", "stopped at max
    episodes" (trained to the cap without converging), or "not assessed" (no convergence file: e.g. trained until exploration decayed,
    or a test run); and the last episode its convergence was assessed after (None without one)
    """
    paths = glob.glob(os.path.join(folder, "data", "results", "**", "convergence_*.json"), recursive=True)
    if not paths:
        return {"learning": "not assessed", "last_episode": None}
    record = read_json(paths[0])
    outcome = "converged" if record.get("converged") else ("stopped at max episodes" if "stopped" in record else "not converged yet")
    return {"learning": outcome, "last_episode": record.get("after_episode")}


def run_rows(folder, record, trained_with, status):
    configuration = record["configuration"]
    #a test run's agents were trained with its training run's config (the one recorded beside the checkpoint, which the test uses)
    source = trained_with or record
    training_config = source["configuration"].get("training_config") or {}
    rewards = training_config.get("rewards", {})
    common = {"mode": configuration.get("mode"), "scenario": configuration.get("scenario"), "condition": configuration.get("condition"),
              "society": configuration.get("society"), "training_seed": configuration.get("seed"),
              "test_seed": configuration.get("test_seed"), "reward_config": reward_config_name(source),
              "sanction": training_config.get("sanction"), "positive_sanction_only": bool(training_config.get("positive_sanction_only", False)),
              "death_reward": rewards.get("death"), "throw_reward": rewards.get("throw"), "forage_reward": rewards.get("forage"),
              "eat_reward": rewards.get("eat"),
              #the rewards ethics agents get instead of the baseline's (e.g. harvest.yaml's eat 0.8), as JSON; {} when they get the same
              "ethics_rewards": json.dumps(training_config.get("ethics_rewards", {}), sort_keys=True), "status": status,
              **learning(folder), "run_folder": os.path.relpath(folder, REPO)}
    rows = []
    for root, _, files in os.walk(folder):
        for name in sorted(files):
            if name.startswith("."):
                #hidden files the system adds (e.g. Finder's .DS_Store), not the run's
                continue
            path = os.path.join(root, name)
            parent = os.path.basename(root)
            rows.append({"file": os.path.relpath(path, REPO), "file_type": file_type(name),
                         "agent_id": int(parent.split("_")[1]) if parent.startswith("agent_") else None, **common})
    return rows


def build(experiment_folder):
    """The manifest of every run under experiment_folder (one row per file), as a DataFrame"""
    folders = sorted(os.path.dirname(path) for path in glob.glob(os.path.join(os.path.abspath(experiment_folder), "**", RUN_FILE), recursive=True))
    records = {folder: read_json(os.path.join(folder, RUN_FILE)) for folder in folders}
    rows = []
    for folder, record in records.items():
        status_path = os.path.join(folder, "status.json")
        status = read_json(status_path).get("status") if os.path.exists(status_path) else None
        trained_with = training_record(record, records) if record["configuration"].get("mode") == "test" else None
        rows += run_rows(folder, record, trained_with, status)
    manifest = pd.DataFrame(rows, columns=COLUMNS)
    manifest["agent_id"] = manifest["agent_id"].astype("Int64")
    manifest["training_seed"] = manifest["training_seed"].astype("Int64")
    manifest["test_seed"] = manifest["test_seed"].astype("Int64")
    return manifest


def reward_label(reward_config):
    """A reward config as the export names it: the harvest scenarios' own config (configs/training/harvest.yaml) is the current rewards"""
    return "current" if reward_config == "harvest" else reward_config


def export_name(row):
    """
    A file's name in the export: <reward>_seed<seed>_<society>_<train|test[_<test condition>]>_<what it is>[_agent<id>].<extension>, e.g.
    suggested_death5_s1_seed101_maximin_train_episode_report.csv or current_seed102_veto_train_agent3_q_network.keras
    """
    original = os.path.basename(row["file"])
    stem, extension = os.path.splitext(original)
    kind = {"model_episode_report": "episode_report"}.get(row["file_type"], row["file_type"])
    if kind == "other":
        kind = stem
    mode = row["mode"] if row["mode"] == "train" else f"{row['mode']}_{row['condition']}"
    agent = [] if pd.isna(row["agent_id"]) else [f"agent{int(row['agent_id'])}"]
    return "_".join([reward_label(row["reward_config"]), f"seed{row['training_seed']}", row["society"], mode, *agent, kind]) + extension


def export(manifest, out):
    """
    Every file in the manifest, under out/<reward>/seed_<training seed>/<society>/ with a name saying the same (export_name), as a hard
    link to the run's file (no copy: the same file under a second name; the runs' own folders are untouched, as testing and the
    experiment runner need them as they are). Returns the manifest with each file's exported path (export_file); writes it to
    out/manifest.csv
    """
    exported = manifest.copy()
    exported["export_file"] = [os.path.join(out, reward_label(row["reward_config"]), f"seed_{row['training_seed']}", row["society"],
                                            export_name(row)) for _, row in manifest.iterrows()]
    duplicates = exported["export_file"][exported["export_file"].duplicated()]
    if len(duplicates):
        raise ValueError(f"two files would be exported under the same name, e.g. {duplicates.iloc[0]}")
    for source, target in zip(exported["file"], exported["export_file"]):
        os.makedirs(os.path.dirname(target), exist_ok=True)
        source = os.path.join(REPO, source)
        if os.path.exists(target):
            if os.path.samefile(source, target):
                continue
            raise FileExistsError(f"{target} exists and isn't {source}")
        os.link(source, target)
    exported["export_file"] = [os.path.relpath(path, REPO) for path in exported["export_file"]]
    exported = exported[["export_file"] + [column for column in exported.columns if column != "export_file"]]
    exported.to_csv(os.path.join(out, "manifest.csv"), index=False)
    return exported


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment_folder", help="An experiment's folder, e.g. runs/reward_trials")
    parser.add_argument("--out", help="Where to write the manifest (default: manifest.csv in the experiment's folder)")
    parser.add_argument("--export", help="Also link every file into this folder by reward, seed and society, named after them (see export)")
    args = parser.parse_args()
    manifest = build(args.experiment_folder)
    if args.export:
        #the export's own files are no run's
        export_root = os.path.abspath(args.export)
        manifest = manifest[[not os.path.join(REPO, path).startswith(export_root + os.sep) for path in manifest["file"]]]
    out = args.out or os.path.join(args.experiment_folder, "manifest.csv")
    manifest.to_csv(out, index=False)
    if args.export:
        exported = export(manifest, args.export)
        print(f"{len(exported)} files linked into {args.export}, by reward, seed and society (its manifest: {args.export}/manifest.csv)")
    runs = manifest.drop_duplicates("run_folder")
    print(f"{len(manifest)} files from {len(runs)} runs written to {out}")
    print(runs.groupby(["mode", "reward_config", "sanction", "positive_sanction_only"], dropna=False)["society"].agg(
        runs="count", societies=lambda s: ", ".join(sorted(set(s)))).to_string())


if __name__ == "__main__":
    main()
