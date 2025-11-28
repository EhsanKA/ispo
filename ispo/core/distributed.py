#!/usr/bin/env python3
"""
Distributed Data Parallel (DDP) utilities for multi-GPU inference.

This module provides utilities for setting up and using PyTorch DDP for
distributed inference across multiple GPUs.
"""

import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def setup_ddp(rank: int, world_size: int, backend: str = "nccl") -> None:
    """
    Initialize the distributed process group for DDP.
    
    Args:
        rank: Process rank (0 to world_size-1)
        world_size: Total number of processes
        backend: Communication backend ('nccl' for GPU, 'gloo' for CPU)
    """
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    
    # Initialize the process group
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size
    )
    
    # Set the device for this process
    torch.cuda.set_device(rank)
    
    logger.info(f"DDP initialized: rank={rank}, world_size={world_size}, backend={backend}")


def cleanup_ddp() -> None:
    """Clean up the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("DDP process group destroyed")


def is_ddp_available() -> bool:
    """Check if DDP is available and properly configured."""
    return torch.cuda.is_available() and torch.cuda.device_count() > 1


def get_world_size() -> int:
    """Get the world size (number of processes) if DDP is initialized."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """Get the current process rank if DDP is initialized."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def wrap_model_with_ddp(model: nn.Module, device: torch.device, 
                        find_unused_parameters: bool = False) -> nn.Module:
    """
    Wrap a model with DistributedDataParallel.
    
    Args:
        model: PyTorch model to wrap
        device: Device to use (should be cuda:rank)
        find_unused_parameters: Whether to find unused parameters (slower but more flexible)
        
    Returns:
        DDP-wrapped model
    """
    if not dist.is_initialized():
        raise RuntimeError("DDP not initialized. Call setup_ddp() first.")
    
    model = model.to(device)
    ddp_model = DDP(
        model,
        device_ids=[device.index] if device.type == 'cuda' else None,
        output_device=device.index if device.type == 'cuda' else None,
        find_unused_parameters=find_unused_parameters
    )
    
    logger.info(f"Model wrapped with DDP on device {device}")
    return ddp_model


def all_reduce_tensor(tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
    """
    All-reduce a tensor across all processes.
    
    Args:
        tensor: Tensor to reduce
        op: Reduction operation (default: SUM)
        
    Returns:
        Reduced tensor (same on all processes)
    """
    if dist.is_initialized():
        dist.all_reduce(tensor, op=op)
    return tensor


def gather_tensors(tensor: torch.Tensor) -> list:
    """
    Gather tensors from all processes to a list.
    
    Args:
        tensor: Tensor to gather
        
    Returns:
        List of tensors from all processes (only on rank 0 if gather_list is None)
    """
    if not dist.is_initialized():
        return [tensor]
    
    world_size = dist.get_world_size()
    gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gather_list, tensor)
    return gather_list


def split_data_for_rank(data, world_size: int, rank: int):
    """
    Split data across processes for DDP.
    
    Args:
        data: Data to split (list, tuple, or dataset)
        world_size: Number of processes
        rank: Current process rank
        
    Returns:
        Data chunk for this process
    """
    if isinstance(data, (list, tuple)):
        chunk_size = len(data) // world_size
        start_idx = rank * chunk_size
        end_idx = start_idx + chunk_size if rank < world_size - 1 else len(data)
        return data[start_idx:end_idx]
    elif hasattr(data, '__len__') and hasattr(data, '__getitem__'):
        # For datasets with length and indexing
        total_len = len(data)
        chunk_size = total_len // world_size
        start_idx = rank * chunk_size
        end_idx = start_idx + chunk_size if rank < world_size - 1 else total_len
        return data.select(range(start_idx, end_idx)) if hasattr(data, 'select') else data[start_idx:end_idx]
    else:
        # If we can't split, return all data (not ideal but works)
        logger.warning(f"Could not split data type {type(data)}. Returning all data for rank {rank}.")
        return data

