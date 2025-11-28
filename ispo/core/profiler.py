#!/usr/bin/env python3
"""
Performance Profiler Module

This module provides performance profiling capabilities for ISP inference,
tracking CPU, GPU, and memory usage during execution.
"""

import time
import numpy as np
import psutil
import GPUtil
import logging
import torch
from typing import Dict, Optional
from pathlib import Path

# Load environment variables from .env file for wandb API key
try:
    from dotenv import load_dotenv
    # Try to load .env from project root (assuming this file is in ispo/core/)
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # Try loading from current directory
        load_dotenv()
except ImportError:
    # python-dotenv not installed, continue without it
    pass

# Configure logging
logger = logging.getLogger(__name__)

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class PerformanceProfiler:
    """Class to profile CPU, GPU, and memory usage during inference."""

    def __init__(self, use_wandb: bool = False, wandb_run=None):
        """
        Initialize profiler.
        
        Args:
            use_wandb: Whether to log to wandb
            wandb_run: wandb run object for logging
        """
        self.start_time = None
        self.end_time = None
        self.cpu_usage = []
        self.cpu_per_core = []
        self.memory_usage = []
        self.memory_used_gb = []
        self.memory_available_gb = []
        self.gpu_usage = []
        self.gpu_memory = []
        self.gpu_memory_total = []
        self.gpu_temperature = []
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_run = wandb_run
        self.step = 0

    def start_profiling(self):
        """Start profiling system resources."""
        self.start_time = time.time()
        logger.info("Started performance profiling")

    def update_metrics(self):
        """Update current resource usage metrics."""
        # CPU usage - overall and per-core
        cpu_percent = psutil.cpu_percent(interval=None)
        self.cpu_usage.append(cpu_percent)
        
        # Per-core CPU usage
        cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
        self.cpu_per_core.append(cpu_per_core)
        avg_cpu_per_core = np.mean(cpu_per_core) if cpu_per_core else 0
        max_cpu_per_core = np.max(cpu_per_core) if cpu_per_core else 0

        # Memory usage - detailed breakdown
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / (1024**3)
        memory_available_gb = memory.available / (1024**3)
        memory_total_gb = memory.total / (1024**3)
        memory_cached_gb = getattr(memory, 'cached', 0) / (1024**3) if hasattr(memory, 'cached') else 0
        
        self.memory_usage.append(memory_percent)
        self.memory_used_gb.append(memory_used_gb)
        self.memory_available_gb.append(memory_available_gb)

        # GPU usage (if available) - detailed metrics
        gpu_percent = None
        gpu_mem_used = None
        gpu_mem_total = None
        gpu_temp = None
        gpu_mem_percent = None
        
        # Try GPUtil first
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                gpu_percent = gpu.load * 100
                gpu_mem_used = gpu.memoryUsed
                gpu_mem_total = gpu.memoryTotal
                gpu_mem_percent = (gpu_mem_used / gpu_mem_total) * 100 if gpu_mem_total > 0 else 0
                gpu_temp = gpu.temperature
                self.gpu_usage.append(gpu_percent)
                self.gpu_memory.append(gpu_mem_used)
                self.gpu_memory_total.append(gpu_mem_total)
                if gpu_temp is not None:
                    self.gpu_temperature.append(gpu_temp)
        except Exception as e:
            logger.debug(f"GPUtil failed: {e}")
        
        # Also try PyTorch CUDA if available
        if torch.cuda.is_available():
            try:
                gpu_mem_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
                gpu_mem_reserved = torch.cuda.memory_reserved() / (1024**3)  # GB
                gpu_mem_max_allocated = torch.cuda.max_memory_allocated() / (1024**3)  # GB
            except:
                gpu_mem_allocated = None
                gpu_mem_reserved = None
                gpu_mem_max_allocated = None
        else:
            gpu_mem_allocated = None
            gpu_mem_reserved = None
            gpu_mem_max_allocated = None
        
        # Log to wandb in real-time
        if self.use_wandb and self.wandb_run is not None:
            log_dict = {
                # CPU metrics
                'runtime/cpu_percent': cpu_percent,
                'runtime/cpu_percent_avg_per_core': avg_cpu_per_core,
                'runtime/cpu_percent_max_per_core': max_cpu_per_core,
                'runtime/cpu_count': psutil.cpu_count(),
                # Memory metrics
                'runtime/memory_percent': memory_percent,
                'runtime/memory_used_gb': memory_used_gb,
                'runtime/memory_available_gb': memory_available_gb,
                'runtime/memory_total_gb': memory_total_gb,
                'runtime/memory_cached_gb': memory_cached_gb,
            }
            
            # GPU metrics from GPUtil
            if gpu_percent is not None:
                log_dict['runtime/gpu_percent'] = gpu_percent
            if gpu_mem_used is not None:
                log_dict['runtime/gpu_memory_used_mb'] = gpu_mem_used
                log_dict['runtime/gpu_memory_used_gb'] = gpu_mem_used / 1024
            if gpu_mem_total is not None:
                log_dict['runtime/gpu_memory_total_gb'] = gpu_mem_total / 1024
            if gpu_mem_percent is not None:
                log_dict['runtime/gpu_memory_percent'] = gpu_mem_percent
            if gpu_temp is not None:
                log_dict['runtime/gpu_temperature_c'] = gpu_temp
            
            # GPU metrics from PyTorch
            if gpu_mem_allocated is not None:
                log_dict['runtime/gpu_memory_allocated_gb'] = gpu_mem_allocated
            if gpu_mem_reserved is not None:
                log_dict['runtime/gpu_memory_reserved_gb'] = gpu_mem_reserved
            if gpu_mem_max_allocated is not None:
                log_dict['runtime/gpu_memory_max_allocated_gb'] = gpu_mem_max_allocated
            
            self.wandb_run.log(log_dict, step=self.step)
            self.step += 1

    def stop_profiling(self) -> Dict[str, float]:
        """Stop profiling and return summary statistics."""
        self.end_time = time.time()
        total_time = self.end_time - self.start_time

        metrics = {
            'total_time_seconds': total_time,
            'avg_cpu_percent': np.mean(self.cpu_usage) if self.cpu_usage else 0,
            'max_cpu_percent': np.max(self.cpu_usage) if self.cpu_usage else 0,
            'min_cpu_percent': np.min(self.cpu_usage) if self.cpu_usage else 0,
            'avg_memory_percent': np.mean(self.memory_usage) if self.memory_usage else 0,
            'max_memory_percent': np.max(self.memory_usage) if self.memory_usage else 0,
            'min_memory_percent': np.min(self.memory_usage) if self.memory_usage else 0,
        }
        
        # Per-core CPU statistics
        if self.cpu_per_core:
            all_cores = np.array(self.cpu_per_core)
            metrics.update({
                'avg_cpu_per_core': np.mean(all_cores),
                'max_cpu_per_core': np.max(all_cores),
                'min_cpu_per_core': np.min(all_cores),
            })
        
        # Memory statistics in GB
        if self.memory_used_gb:
            metrics.update({
                'avg_memory_used_gb': np.mean(self.memory_used_gb),
                'max_memory_used_gb': np.max(self.memory_used_gb),
                'min_memory_used_gb': np.min(self.memory_used_gb),
                'avg_memory_available_gb': np.mean(self.memory_available_gb),
                'min_memory_available_gb': np.min(self.memory_available_gb),
            })

        if self.gpu_usage:
            metrics.update({
                'avg_gpu_percent': np.mean(self.gpu_usage),
                'max_gpu_percent': np.max(self.gpu_usage),
                'min_gpu_percent': np.min(self.gpu_usage),
                'avg_gpu_memory_mb': np.mean(self.gpu_memory),
                'max_gpu_memory_mb': np.max(self.gpu_memory),
                'min_gpu_memory_mb': np.min(self.gpu_memory),
            })
            
            if self.gpu_memory_total:
                metrics.update({
                    'avg_gpu_memory_total_gb': np.mean(self.gpu_memory_total) / 1024,
                })
            
            if self.gpu_temperature:
                metrics.update({
                    'avg_gpu_temperature_c': np.mean(self.gpu_temperature),
                    'max_gpu_temperature_c': np.max(self.gpu_temperature),
                    'min_gpu_temperature_c': np.min(self.gpu_temperature),
                })
        
        # PyTorch CUDA memory stats if available
        if torch.cuda.is_available():
            try:
                metrics.update({
                    'pytorch_gpu_memory_allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                    'pytorch_gpu_memory_reserved_gb': torch.cuda.memory_reserved() / (1024**3),
                    'pytorch_gpu_memory_max_allocated_gb': torch.cuda.max_memory_allocated() / (1024**3),
                })
            except:
                pass

        # Log summary to wandb
        if self.use_wandb and self.wandb_run is not None:
            summary_log = {}
            for key, value in metrics.items():
                summary_log[f'summary/{key}'] = value
            self.wandb_run.log(summary_log)

        logger.info(f"Performance profiling completed. Total time: {total_time:.2f}s")
        return metrics



