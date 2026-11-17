"""
Each run launched with run.py's --output-root has a folder of its own (see run_folder), holding everything it writes (its data/, as
run.py writes to relative paths and runs inside the folder) and:

    run.json -- the run's full configuration: mode, settings, seeds, the command, the machine (hostname, CPU and GPU model,
        architecture) and the Python and package versions, and the git commit
    status.json -- "running" (with the process id and host), "complete" or "failed" (with the error); a run whose status says
        running but whose process is gone stopped partway, and is "incomplete" (see read_status)
    run.log -- everything the run printed

Relaunching a run's command continues an incomplete training run from its checkpoint, restarts an incomplete test run, and does
nothing for a complete one.
"""
import datetime
import json
import os
import platform
import socket
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_FILE = "run.json"
STATUS_FILE = "status.json"
LOG_FILE = "run.log"
PACKAGES = ["numpy", "pandas", "scipy", "statsmodels", "tensorflow", "keras", "mesa", "matplotlib", "seaborn", "yaml"]


def run_folder(output_root, experiment, mode, condition, society, seed_label):
    """
    <output-root>/<experiment>/<train|test>/<condition>/<society>/<seed_label>, e.g. runs/main/train/colours_4agents/maximin/seed_3
    or runs/main/test/allotment_2allotments_4agents/maximin/train_seed_3
    """
    return os.path.join(output_root, experiment, mode, condition, society, seed_label)


def _write_json(path, value):
    writing = path + ".writing"
    with open(writing, "w") as file:
        json.dump(value, file, indent=2, default=str)
    os.replace(writing, path)


def _read_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as file:
        return json.load(file)


def _command_output(command):
    try:
        return subprocess.run(command, capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def cpu_model():
    if sys.platform == "darwin":
        return _command_output(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor()
    if os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo") as file:
            for line in file:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    return platform.processor() or None


def gpu_models():
    """The GPUs TensorFlow can use (empty when it runs on CPU only)"""
    try:
        import tensorflow as tf
        devices = tf.config.list_physical_devices("GPU")
        return [tf.config.experimental.get_device_details(device).get("device_name", device.name) for device in devices]
    except Exception:
        return []


def package_versions():
    versions = {"python": platform.python_version()}
    for name in PACKAGES:
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", None)
        except ImportError:
            versions[name] = None
    return versions


def machine():
    return {"hostname": socket.gethostname(), "cpu": cpu_model(), "gpus": gpu_models(), "architecture": platform.machine(),
            "system": f"{platform.system()} {platform.release()}"}


def git_commit():
    return _command_output(["git", "-C", REPO, "rev-parse", "HEAD"])


def write_run_record(folder, configuration):
    """run.json: the configuration given, with the command, the machine, the package versions and the git commit"""
    record = {"configuration": configuration, "command": [sys.executable] + sys.argv, "machine": machine(),
              "packages": package_versions(), "git_commit": git_commit(), "written": datetime.datetime.now().isoformat(timespec="seconds")}
    _write_json(os.path.join(folder, RUN_FILE), record)
    return record


def read_run_record(folder):
    return _read_json(os.path.join(folder, RUN_FILE))


def write_status(folder, status, **details):
    _write_json(os.path.join(folder, STATUS_FILE), {"status": status, "time": datetime.datetime.now().isoformat(timespec="seconds"),
                                                     "pid": os.getpid(), "host": socket.gethostname(), **details})


def _process_alive(pid):
    try:
        os.kill(pid, 0)
    except (OSError, TypeError):
        return False
    return True


def read_status(folder):
    """
    The run's status: "not started" (no status file), "running", "incomplete" (it says running, but its process on this host is
    gone: stopped partway), "complete" or "failed"
    """
    status = _read_json(os.path.join(folder, STATUS_FILE))
    if status is None:
        return "not started"
    if status["status"] == "running" and status.get("host") == socket.gethostname() and not _process_alive(status.get("pid")):
        return "incomplete"
    return status["status"]


class Tee:
    """Writes what a run prints to the console and to its log"""
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text):
        self.stream.write(text)
        self.log.write(text)
        self.log.flush()

    def flush(self):
        self.stream.flush()
        self.log.flush()

    def isatty(self):
        return False


def log_to_folder(folder):
    """Copy everything the run prints (stdout and stderr) into its run.log, appending, so a relaunch adds to it"""
    log = open(os.path.join(folder, LOG_FILE), "a")
    log.write(f"\n=== {datetime.datetime.now().isoformat(timespec='seconds')} {' '.join(sys.argv)}\n")
    sys.stdout = Tee(sys.stdout, log)
    sys.stderr = Tee(sys.stderr, log)
