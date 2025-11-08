import torch
import torch.distributed as dist
from typing import List, Optional


class DeviceMesh:
    """
    Handles device mapping and process topology for multi-GPU or multi-node setups.
    Provides utilities for collective operations and device-aware tensor placement.
    """

    def __init__(self, devices: Optional[List[str]] = None):
        """
        Initialize a device mesh.

        Args:
            devices (List[str], optional): List of device strings like ["cuda:0", "cuda:1"].
                                           If None, automatically detects available GPUs.
        """
        if devices is None:
            if torch.cuda.is_available():
                num_devices = torch.cuda.device_count()
                self.devices = [f"cuda:{i}" for i in range(num_devices)]
            else:
                self.devices = ["cpu"]
        else:
            self.devices = devices

        self.world_size = len(self.devices)
        self.initialized = dist.is_initialized()

        if self.initialized:
            self.rank = dist.get_rank()
        else:
            self.rank = 0

    def setup_distributed(self, backend: str = "nccl"):
        """
        Initialize torch.distributed if not already initialized.
        """
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
            self.initialized = True
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def get_device_for_rank(self, rank: Optional[int] = None) -> torch.device:
        """
        Get the device corresponding to the given rank.
        """
        if rank is None:
            rank = self.rank
        if rank < len(self.devices):
            return torch.device(self.devices[rank])
        return torch.device("cpu")

    def barrier(self):
        """
        Synchronize all ranks.
        """
        if self.initialized:
            dist.barrier()

    def scatter_tensor(self, tensor: torch.Tensor, dim: int = 0):
        """
        Example stub for tensor scattering across devices.
        """
        if self.world_size == 1:
            return [tensor.to(self.devices[0])]

        chunks = torch.chunk(tensor, self.world_size, dim=dim)
        return [chunk.to(self.devices[i]) for i, chunk in enumerate(chunks)]

    def __repr__(self):
        return f"<DeviceMesh world_size={self.world_size}, devices={self.devices}>"
