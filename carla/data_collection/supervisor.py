import subprocess
import time
import glob
import os
import h5py
import psutil
import threading
import msvcrt
from tqdm import tqdm

# --- Configuration ---
MAIN_SCRIPT = 'python collector/run_collector.py --runs=10 --no-progress'
BACKUP_PATTERN = 'backup_run_*.hdf5'
AGG_FILE = r'D:/marathon.hdf5'
TARGET_RUNS = 1000
RELAUNCH_DELAY = 2  # seconds between relaunches

# --- Utility Functions ---
def kill_carla_processes():
    """Terminate any running Carla Unreal Engine processes."""
    for proc in psutil.process_iter(['name']):
        try:
            if proc.info['name'] and 'CarlaUnreal' in proc.info['name']:
                proc.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def count_agg_runs():
    """Return number of runs in the aggregate file."""
    if not os.path.exists(AGG_FILE):
        return 0
    with h5py.File(AGG_FILE, 'r') as f:
        return len(f['runs'])


def merge_backup(backup_path):
    """Merge runs from a backup HDF5 into the aggregate file with incremental run IDs,
    keeping only runs where 'control' has 500 entries."""
    if not os.path.exists(AGG_FILE):
        with h5py.File(AGG_FILE, 'w') as f:
            f.create_group('runs')
    with h5py.File(AGG_FILE, 'a') as agg, h5py.File(backup_path, 'r') as src:
        current_count = len(agg['runs'])
        sorted_keys = sorted(src['runs'].keys(), key=lambda k: int(k.split('_')[-1]))
        valid_keys = [
            k for k in sorted_keys
            if len(src[f"runs/{k}/vehicles/0/waypoint"]) == 500
        ]
        for idx, run_key in enumerate(valid_keys, start=1):
            new_run_id = f"{current_count + idx}"
            src.copy(f'runs/{run_key}', agg['runs'], name=new_run_id)


def cleanup_local():
    """Remove all local HDF5 backup and temp files."""
    for path in glob.glob('*.hdf5'):
        try:
            os.remove(path)
        except OSError:
            pass

# --- Key Listener (Windows) ---
def start_key_listener(callback):
    """Listen for single-key presses and invoke callback."""
    def listen():
        while True:
            if msvcrt.kbhit():
                ch = msvcrt.getch().decode('utf-8', errors='ignore')
                callback(ch)
            time.sleep(0.1)
    t = threading.Thread(target=listen, daemon=True)
    t.start()
    return t

# --- Supervisor Loop ---
def run_supervisor():
    aggregated = count_agg_runs()
    ##print(f"Starting supervisor: {aggregated}/{TARGET_RUNS} runs aggregated.")
    start_key_listener(on_key)

    # Use tqdm progress bar for aggregation
    pbar = tqdm(total=TARGET_RUNS, initial=aggregated, desc='Aggregated runs')

    try:
        while aggregated < TARGET_RUNS:
            ##print("Launching collector...")
            with open(os.devnull, 'w') as devnull:
                proc = subprocess.Popen(MAIN_SCRIPT, shell=True, stdout=devnull, stderr=devnull)
                ret = proc.wait()

            # if ret == 0:
            #     #print("Collector exited cleanly. Supervisor stopping.")
            #     break

            #print(f"Collector crashed (code {ret}). Merging latest backup.")
            kill_carla_processes()

            backups = sorted(glob.glob(BACKUP_PATTERN), key=os.path.getmtime)
            if backups:
                latest = backups[-1]
                merge_backup(latest)
                old = aggregated
                aggregated = count_agg_runs()
                pbar.update(aggregated - old)
                #print(f"Merged {os.path.basename(latest)} => {aggregated}/{TARGET_RUNS}")
            else:
                #print("No backups found to merge. Retrying...")
                continue

            cleanup_local()
            #print("Cleaned local .hdf5 files.")

            for i in range(RELAUNCH_DELAY, 0, -1):
                #print(f"Relaunch in {i}s (press Q to quit)", end='\r')
                time.sleep(1)
            #print()

    except KeyboardInterrupt:
        #print("\nKeyboard interrupt received. Finalizing merge and exiting...")
        backups = sorted(glob.glob(BACKUP_PATTERN), key=os.path.getmtime)
        if backups:
            merge_backup(backups[-1])
            aggregated = count_agg_runs()
            pbar.update(aggregated - pbar.n)
        kill_carla_processes()
    finally:
        pbar.close()
        print("Finalizing merge and exiting...")
        print("Supervisor terminated.")

# --- Key Callback ---
def on_key(ch):
    if ch.lower() == 'q':
        raise KeyboardInterrupt

# --- Entry Point ---
if __name__ == '__main__':
    print("Starting supervisor...")
    print("Press 'Q' to quit at any time.")
    kill_carla_processes()
    cleanup_local()
    run_supervisor()
