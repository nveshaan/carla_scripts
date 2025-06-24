import os
import h5py

master_path = "D:/marathon.hdf5"
sample_path = "data/run.hdf5"

os.makedirs(os.path.dirname(sample_path), exist_ok=True)

with h5py.File(master_path, 'r') as master, h5py.File(sample_path, 'a') as sample:
    if 'runs' not in sample:
        runs = sample.create_group("runs")
    else:
        runs = sample['runs']
    
    for i in range(1):
        group_name = f"{i+1}"
        master_group_path = f"runs/{group_name}"
        master.copy(master_group_path, runs, name=group_name)
