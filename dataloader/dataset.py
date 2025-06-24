import h5py
import torch
from torch.utils.data import Dataset
from typing import Sequence, Tuple, Optional
from functools import lru_cache
import os


def convert_hdf5_with_chunking(
    input_path: str,
    output_path: str,
    compression: str = "gzip",
    compression_level: int = 4,
    chunk_size: int = 64
):
    """
    Converts an existing HDF5 file to a chunked and compressed version for efficient lazy loading.

    Args:
        input_path (str): Path to original HDF5 file.
        output_path (str): Path to save the converted HDF5 file.
        compression (str): 'gzip', 'lzf', or None.
        compression_level (int): Compression level (only for 'gzip').
        chunk_size (int): Chunk size along time dimension.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with h5py.File(input_path, "r") as fin, h5py.File(output_path, "w") as fout:
        data_group_in = fin["runs"]
        data_group_out = fout.create_group("runs")

        for run_key in data_group_in:
            run_in = data_group_in[run_key]
            run_out = data_group_out.create_group(run_key)

            for key in run_in['vehicles/0']:
                data = run_in[key][:]
                shape = data.shape
                run_out.create_dataset(
                    key,
                    data=data,
                    compression=compression,
                    compression_opts=compression_level if compression == "gzip" else None,
                    chunks=(chunk_size, *shape[1:])
                )

    print(f"Converted {len(data_group_in)} runs to {output_path} with chunking & compression.")


class SampleData(Dataset):
    """
    Lazily loads multimodal sequences from a large HDF5 file for efficient training.

    Parameters
    ----------
    file_path : str
        Path to the HDF5 file.
    obs_keys : Sequence[str]
        List of keys under the 'obs' group to use as observation modalities (e.g., ['rgb', 'lidar']).
    act_keys : Sequence[str]
        List of keys under each trajectory (outside 'obs') to use as action modalities (e.g., ['actions', 'velocity']).
    obs_horizon : int, optional
        Number of past frames to stack as observation (default is 2).
    act_horizon : int, optional
        Number of future actions to predict (default is 16).
    gap : int, optional
        Number of timesteps between the last observation and first predicted action (default is 0).
    obs_stride : int, optional
        Temporal stride for observation sampling (default is 1).
    act_stride : int, optional
        Temporal stride for predicted action sampling (default is 1).
    cache_size : int or None, optional
        If set, enables LRU caching of recent samples (default is 0 = no caching).

    Returns (per sample)
    ---------------------
    actions : list of torch.Tensor
        Each tensor is shaped (act_horizon, D), where D is action dimension for that modality.
    observations : list of torch.Tensor
        Each tensor is shaped (obs_horizon, H, W, C) for image modalities or (obs_horizon, ...) for others.

    Output from DataLoader
    ----------------------
    When wrapped in a DataLoader:

    >>> for batch_idx, (actions, observations) in enumerate(dataloader):

    - actions : list of batched tensors, each of shape (B, act_horizon, D)
    - observations : list of batched tensors, each of shape (B, obs_horizon, D) # D depends the type of data(image, lidar, control, etc.) batched

    Notes
    -----
    - Each modality is returned as a separate tensor in the list.
    - The batching is handled by PyTorch's `default_collate`, which stacks each modality independently.
    - This structure is compatible with multimodal training pipelines (e.g., image + lidar + actions).
    - To simplify access, consider modifying `__getitem__()` to return dictionaries instead of lists.
    """

    def __init__(self, file_path: str, obs_horizon: int, act_horizon: int, gap: int, obs_stride: int, act_stride: int, obs_keys: Sequence[str], act_keys: Sequence[str], compressed: bool=False, cache_size: Optional[int] = 0) -> Dataset:
        super().__init__()
        self.file_path = file_path
        self.file = None
        self.cache_size = cache_size

        self.obs_horizon = obs_horizon
        self.act_horizon = act_horizon
        self.gap = gap
        self.obs_stride = obs_stride
        self.act_stride = act_stride
        self.obs_keys = obs_keys
        self.act_keys = act_keys

        if compressed: self.diff = ''
        else: self.diff = 'vehicles/0/'

        self.index_map = []
        with h5py.File(self.file_path, 'r') as f:
            for run_key in f['runs']:
                run = f[f'runs/{run_key}']
                total_frames = len(run[f'{self.diff}control'])
                max_index = total_frames - (
                    (obs_horizon - 1) * obs_stride +
                    gap +
                    (act_horizon - 1) * act_stride + 1)

                for i in range(max_index):
                    self.index_map.append((run_key, i))
                
        if self.cache_size:
            self._load_sample = lru_cache(maxsize = self.cache_size)(self._load_sample)

    def __len__(self) -> int:
        return len(self.index_map)
    
    def __getitem__(self, idx:int) -> Tuple[list[torch.Tensor], list[torch.Tensor]]:
        if self.file is None:
            self.file = h5py.File(self.file_path, 'r')
        run_key, start_idx = self.index_map[idx]
        try:
            result = self._load_sample(run_key, start_idx)
            return result
        except Exception as e:
            print(f"Error loading sample {idx} (run_key={run_key}, start_idx={start_idx}): {e}")
            raise
    
    def _load_sample(self, run_key: str, start_idx: int) -> Tuple[list[torch.Tensor], list[torch.Tensor]]:
        run = self.file[f'runs/{run_key}']
        obs_tensors, act_tensors = [], []

        for key in self.obs_keys:
            data = run[self.diff+key]
            idxs = [start_idx + i * self.obs_stride for i in range(self.obs_horizon)]
            if key == 'image':
                obs_tensors.append(torch.stack([torch.flip(torch.tensor(data[i], dtype=torch.float32)[:, :, :3], dims=[-1]).permute(2, 0, 1)
                                                    for i in idxs]).squeeze())
            elif key == 'velocity':
                obs_tensors.append(torch.tensor(data[idxs], dtype=torch.float32).norm().unsqueeze(0))
            else:
                obs_tensors.append(torch.tensor(data[idxs], dtype=torch.float32).squeeze())

        pred_start = start_idx + (self.obs_horizon - 1) * self.obs_stride + self.gap
        for key in self.act_keys:
            data = run[self.diff+key]
            idxs = [pred_start + i * self.act_stride for i in range(self.act_horizon)]
            if key == 'location' or key == 'waypoint':
                act_tensors.append(torch.abs(torch.tensor(data[idxs,:2], dtype=torch.float32).squeeze()) - torch.abs(torch.tensor(run[self.diff+'location'][start_idx, :2], dtype=torch.float32).squeeze()))
            else:
                act_tensors.append(torch.tensor(data[idxs], dtype=torch.float32).squeeze())

        return obs_tensors, act_tensors

    def __del__(self):
        if self.file:
            try:
                self.file.close()
            except Exception:
                pass

if __name__ == "__main__":
    sample = SampleData(
        file_path='D:/marathon.hdf5',
        obs_horizon=1,
        act_horizon=5,
        gap=0,
        obs_stride=1,
        act_stride=1,
        obs_keys=['image', 'velocity', 'command'],
        act_keys=['location'])
    
    print(f"Length of dataset: {len(sample)}")
    print(f"Content of one instance: {next(iter(sample))[0][2].item()}")
    