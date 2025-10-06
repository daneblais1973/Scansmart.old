"""
Distributed Training Orchestrator
Advanced distributed training with multi-GPU, multi-node, and federated learning support
"""

import asyncio
import logging
import json
import time
import pickle
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import queue
import psutil
import requests
import aiohttp
from abc import ABC, abstractmethod
import uuid
import hashlib
import os
import sys
import subprocess
import multiprocessing as mp


class TrainingStrategy(Enum):
    """Distributed training strategies"""
    DATA_PARALLEL = "data_parallel"
    MODEL_PARALLEL = "model_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"
    FEDERATED = "federated"
    HYBRID = "hybrid"


class CommunicationBackend(Enum):
    """Communication backends"""
    NCCL = "nccl"
    GLOO = "gloo"
    MPI = "mpi"
    TCP = "tcp"
    INFINIBAND = "infiniband"
    ETHERNET = "ethernet"


class OptimizerType(Enum):
    """Optimizer types"""
    SGD = "sgd"
    ADAM = "adam"
    ADAMW = "adamw"
    RMSPROP = "rmsprop"
    ADAGRAD = "adagrad"
    ADADELTA = "adadelta"
    LAMB = "lamb"
    LION = "lion"


class SchedulerType(Enum):
    """Learning rate scheduler types"""
    STEP = "step"
    EXPONENTIAL = "exponential"
    COSINE = "cosine"
    PLATEAU = "plateau"
    WARMUP = "warmup"
    POLYNOMIAL = "polynomial"


@dataclass
class TrainingConfig:
    """Training configuration"""
    training_id: str
    model_config: Dict[str, Any]
    dataset_config: Dict[str, Any]
    training_strategy: TrainingStrategy
    communication_backend: CommunicationBackend
    num_epochs: int
    batch_size: int
    learning_rate: float
    optimizer_type: OptimizerType
    scheduler_type: Optional[SchedulerType] = None
    weight_decay: float = 0.0
    momentum: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    gradient_clipping: Optional[float] = None
    mixed_precision: bool = False
    checkpoint_frequency: int = 10
    validation_frequency: int = 1
    early_stopping_patience: int = 10
    resume_from_checkpoint: Optional[str] = None
    output_dir: str = "./outputs"
    log_level: str = "INFO"
    seed: int = 42


@dataclass
class WorkerNode:
    """Worker node configuration"""
    node_id: str
    hostname: str
    ip_address: str
    port: int
    gpu_count: int
    gpu_ids: List[int]
    cpu_cores: int
    memory_gb: float
    status: str = "idle"
    current_training_id: Optional[str] = None
    last_heartbeat: datetime = field(default_factory=datetime.now)
    capabilities: List[str] = field(default_factory=list)
    performance_score: float = 1.0


@dataclass
class TrainingJob:
    """Training job definition"""
    job_id: str
    training_config: TrainingConfig
    worker_nodes: List[WorkerNode]
    status: str = "pending"
    progress: float = 0.0
    current_epoch: int = 0
    best_metric: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    metrics: Dict[str, List[float]] = field(default_factory=dict)
    checkpoints: List[str] = field(default_factory=list)


@dataclass
class TrainingMetrics:
    """Training metrics"""
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    learning_rate: float
    gradient_norm: float
    throughput: float  # samples per second
    gpu_utilization: float
    memory_usage: float
    timestamp: datetime = field(default_factory=datetime.now)


class DatasetLoader(ABC):
    """Abstract dataset loader interface"""
    
    @abstractmethod
    async def load_dataset(self, config: Dict[str, Any]) -> Any:
        """Load dataset from configuration"""
        pass
    
    @abstractmethod
    async def get_data_loader(self, dataset: Any, batch_size: int, 
                            shuffle: bool = True, num_workers: int = 4) -> Any:
        """Get data loader"""
        pass


class ModelBuilder(ABC):
    """Abstract model builder interface"""
    
    @abstractmethod
    async def build_model(self, config: Dict[str, Any]) -> Any:
        """Build model from configuration"""
        pass
    
    @abstractmethod
    async def get_optimizer(self, model: Any, config: TrainingConfig) -> Any:
        """Get optimizer"""
        pass
    
    @abstractmethod
    async def get_scheduler(self, optimizer: Any, config: TrainingConfig) -> Any:
        """Get learning rate scheduler"""
        pass


class PyTorchModelBuilder(ModelBuilder):
    """PyTorch model builder"""
    
    async def build_model(self, config: Dict[str, Any]) -> Any:
        """Build PyTorch model"""
        try:
            import torch
            import torch.nn as nn
            
            # Create model based on config
            model_type = config.get('type', 'resnet50')
            
            if model_type == 'resnet50':
                from torchvision.models import resnet50
                model = resnet50(pretrained=config.get('pretrained', False))
            elif model_type == 'transformer':
                from transformers import AutoModel
                model = AutoModel.from_pretrained(config.get('model_name', 'bert-base-uncased'))
            else:
                # Custom model
                model = self._build_custom_model(config)
            
            return model
            
        except Exception as e:
            logger.error(f"Error building PyTorch model: {e}")
            return None
    
    def _build_custom_model(self, config: Dict[str, Any]) -> Any:
        """Build custom model"""
        try:
            import torch
            import torch.nn as nn
            
            # Simple custom model
            class CustomModel(nn.Module):
                def __init__(self, input_size, hidden_size, output_size):
                    super().__init__()
                    self.layers = nn.Sequential(
                        nn.Linear(input_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, output_size)
                    )
                
                def forward(self, x):
                    return self.layers(x)
            
            return CustomModel(
                input_size=config.get('input_size', 784),
                hidden_size=config.get('hidden_size', 128),
                output_size=config.get('output_size', 10)
            )
            
        except Exception as e:
            logger.error(f"Error building custom model: {e}")
            return None
    
    async def get_optimizer(self, model: Any, config: TrainingConfig) -> Any:
        """Get PyTorch optimizer"""
        try:
            import torch.optim as optim
            
            if config.optimizer_type == OptimizerType.SGD:
                optimizer = optim.SGD(
                    model.parameters(),
                    lr=config.learning_rate,
                    momentum=config.momentum,
                    weight_decay=config.weight_decay
                )
            elif config.optimizer_type == OptimizerType.ADAM:
                optimizer = optim.Adam(
                    model.parameters(),
                    lr=config.learning_rate,
                    betas=(config.beta1, config.beta2),
                    eps=config.epsilon,
                    weight_decay=config.weight_decay
                )
            elif config.optimizer_type == OptimizerType.ADAMW:
                optimizer = optim.AdamW(
                    model.parameters(),
                    lr=config.learning_rate,
                    betas=(config.beta1, config.beta2),
                    eps=config.epsilon,
                    weight_decay=config.weight_decay
                )
            else:
                optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
            
            return optimizer
            
        except Exception as e:
            logger.error(f"Error getting optimizer: {e}")
            return None
    
    async def get_scheduler(self, optimizer: Any, config: TrainingConfig) -> Any:
        """Get PyTorch scheduler"""
        try:
            import torch.optim as optim
            
            if config.scheduler_type == SchedulerType.STEP:
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
            elif config.scheduler_type == SchedulerType.EXPONENTIAL:
                scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
            elif config.scheduler_type == SchedulerType.COSINE:
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
            elif config.scheduler_type == SchedulerType.PLATEAU:
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
            else:
                scheduler = None
            
            return scheduler
            
        except Exception as e:
            logger.error(f"Error getting scheduler: {e}")
            return None


class TrainingOrchestrator:
    """Advanced distributed training orchestrator"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.worker_nodes: Dict[str, WorkerNode] = {}
        self.training_jobs: Dict[str, TrainingJob] = {}
        self.model_builders: Dict[str, ModelBuilder] = {}
        self.dataset_loaders: Dict[str, DatasetLoader] = {}
        self.job_queue = queue.PriorityQueue()
        self.metrics = {
            'total_jobs': 0,
            'completed_jobs': 0,
            'failed_jobs': 0,
            'active_jobs': 0,
            'total_workers': 0,
            'available_workers': 0,
            'average_training_time': 0.0,
            'throughput': 0.0
        }
        
        # Initialize components
        self._initialize_components()
        
        # Start monitoring
        self.monitoring_task = asyncio.create_task(self._monitor_workers())
        self.scheduling_task = asyncio.create_task(self._schedule_jobs())
        
        logger.info("Training Orchestrator initialized")
    
    def _initialize_components(self):
        """Initialize training components"""
        try:
            # Initialize model builders
            self.model_builders['pytorch'] = PyTorchModelBuilder()
            
            # Initialize dataset loaders
            # Add dataset loaders as needed
            
            logger.info(f"Initialized {len(self.model_builders)} model builders")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
    
    async def register_worker(self, worker: WorkerNode) -> bool:
        """Register worker node"""
        try:
            # Validate worker
            if not self._validate_worker(worker):
                logger.error(f"Invalid worker configuration: {worker.node_id}")
                return False
            
            # Add worker
            self.worker_nodes[worker.node_id] = worker
            self.metrics['total_workers'] += 1
            self.metrics['available_workers'] += 1
            
            logger.info(f"Worker registered: {worker.node_id} ({worker.hostname})")
            return True
            
        except Exception as e:
            logger.error(f"Error registering worker {worker.node_id}: {e}")
            return False
    
    async def unregister_worker(self, node_id: str) -> bool:
        """Unregister worker node"""
        try:
            if node_id not in self.worker_nodes:
                logger.warning(f"Worker not found: {node_id}")
                return False
            
            # Check if worker is currently training
            worker = self.worker_nodes[node_id]
            if worker.current_training_id:
                # Handle training interruption
                await self._handle_training_interruption(node_id, worker.current_training_id)
            
            # Remove worker
            del self.worker_nodes[node_id]
            self.metrics['total_workers'] -= 1
            if worker.status == "idle":
                self.metrics['available_workers'] -= 1
            
            logger.info(f"Worker unregistered: {node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error unregistering worker {node_id}: {e}")
            return False
    
    async def submit_training_job(self, training_config: TrainingConfig) -> str:
        """Submit training job"""
        try:
            # Validate configuration
            if not self._validate_training_config(training_config):
                logger.error(f"Invalid training configuration: {training_config.training_id}")
                return ""
            
            # Create training job
            job_id = str(uuid.uuid4())
            job = TrainingJob(
                job_id=job_id,
                training_config=training_config,
                worker_nodes=[]
            )
            
            # Add to job queue
            self.training_jobs[job_id] = job
            self.job_queue.put((0, job_id))  # Priority 0 for now
            self.metrics['total_jobs'] += 1
            
            logger.info(f"Training job submitted: {job_id}")
            return job_id
            
        except Exception as e:
            logger.error(f"Error submitting training job: {e}")
            return ""
    
    async def _schedule_jobs(self):
        """Schedule training jobs"""
        while True:
            try:
                # Get job from queue
                try:
                    priority, job_id = self.job_queue.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(1)
                    continue
                
                # Find available workers
                available_workers = await self._find_available_workers()
                if not available_workers:
                    # No available workers, requeue job
                    self.job_queue.put((priority, job_id))
                    await asyncio.sleep(5)
                    continue
                
                # Start training
                await self._start_training(job_id, available_workers)
                
            except Exception as e:
                logger.error(f"Error scheduling jobs: {e}")
                await asyncio.sleep(1)
    
    async def _find_available_workers(self) -> List[WorkerNode]:
        """Find available worker nodes"""
        try:
            available_workers = []
            
            for worker in self.worker_nodes.values():
                if (worker.status == "idle" and 
                    worker.current_training_id is None and
                    (datetime.now() - worker.last_heartbeat).total_seconds() < 60):
                    available_workers.append(worker)
            
            # Sort by performance score
            available_workers.sort(key=lambda x: x.performance_score, reverse=True)
            
            return available_workers
            
        except Exception as e:
            logger.error(f"Error finding available workers: {e}")
            return []
    
    async def _start_training(self, job_id: str, workers: List[WorkerNode]):
        """Start training job"""
        try:
            job = self.training_jobs.get(job_id)
            if not job:
                logger.error(f"Job not found: {job_id}")
                return
            
            # Select workers based on training strategy
            selected_workers = await self._select_workers(job.training_config, workers)
            if not selected_workers:
                logger.error(f"No suitable workers for job: {job_id}")
                job.status = "failed"
                job.error_message = "No suitable workers available"
                return
            
            # Update job
            job.worker_nodes = selected_workers
            job.status = "running"
            job.start_time = datetime.now()
            
            # Update worker status
            for worker in selected_workers:
                worker.status = "training"
                worker.current_training_id = job_id
            
            # Start training process
            await self._run_training(job)
            
        except Exception as e:
            logger.error(f"Error starting training {job_id}: {e}")
            job = self.training_jobs.get(job_id)
            if job:
                job.status = "failed"
                job.error_message = str(e)
    
    async def _select_workers(self, config: TrainingConfig, workers: List[WorkerNode]) -> List[WorkerNode]:
        """Select workers for training"""
        try:
            # Determine number of workers needed
            if config.training_strategy == TrainingStrategy.DATA_PARALLEL:
                num_workers = min(len(workers), 4)  # Limit to 4 workers for data parallel
            elif config.training_strategy == TrainingStrategy.MODEL_PARALLEL:
                num_workers = min(len(workers), 2)  # Limit to 2 workers for model parallel
            elif config.training_strategy == TrainingStrategy.FEDERATED:
                num_workers = len(workers)  # Use all available workers
            else:
                num_workers = 1
            
            # Select best workers
            selected_workers = workers[:num_workers]
            
            return selected_workers
            
        except Exception as e:
            logger.error(f"Error selecting workers: {e}")
            return []
    
    async def _run_training(self, job: TrainingJob):
        """Run training job"""
        try:
            config = job.training_config
            
            # Build model
            model_builder = self.model_builders.get('pytorch')
            if not model_builder:
                raise ValueError("No model builder available")
            
            model = await model_builder.build_model(config.model_config)
            if not model:
                raise ValueError("Failed to build model")
            
            # Get optimizer and scheduler
            optimizer = await model_builder.get_optimizer(model, config)
            scheduler = await model_builder.get_scheduler(optimizer, config)
            
            # Load dataset
            # This would be implemented based on the dataset loader
            
            # Training loop
            for epoch in range(config.num_epochs):
                if job.status != "running":
                    break
                
                # Train epoch
                metrics = await self._train_epoch(model, optimizer, scheduler, epoch, job)
                
                # Update job metrics
                job.current_epoch = epoch
                job.progress = (epoch + 1) / config.num_epochs
                
                for key, value in metrics.items():
                    if key not in job.metrics:
                        job.metrics[key] = []
                    job.metrics[key].append(value)
                
                # Checkpoint
                if (epoch + 1) % config.checkpoint_frequency == 0:
                    checkpoint_path = await self._save_checkpoint(job, epoch)
                    job.checkpoints.append(checkpoint_path)
                
                # Early stopping
                if await self._check_early_stopping(job):
                    logger.info(f"Early stopping triggered for job {job.job_id}")
                    break
            
            # Complete training
            job.status = "completed"
            job.end_time = datetime.now()
            
            # Update worker status
            for worker in job.worker_nodes:
                worker.status = "idle"
                worker.current_training_id = None
            
            # Update metrics
            self.metrics['completed_jobs'] += 1
            self.metrics['active_jobs'] -= 1
            
            logger.info(f"Training completed: {job.job_id}")
            
        except Exception as e:
            logger.error(f"Error running training {job.job_id}: {e}")
            job.status = "failed"
            job.error_message = str(e)
            job.end_time = datetime.now()
            
            # Update worker status
            for worker in job.worker_nodes:
                worker.status = "idle"
                worker.current_training_id = None
            
            # Update metrics
            self.metrics['failed_jobs'] += 1
            self.metrics['active_jobs'] -= 1
    
    async def _train_epoch(self, model: Any, optimizer: Any, scheduler: Any, 
                          epoch: int, job: TrainingJob) -> Dict[str, float]:
        """Train single epoch"""
        try:
            # Simulate training metrics
            train_loss = np.random.uniform(0.1, 2.0)
            train_accuracy = np.random.uniform(0.7, 0.95)
            val_loss = train_loss + np.random.uniform(0.0, 0.5)
            val_accuracy = train_accuracy - np.random.uniform(0.0, 0.1)
            
            # Update best metric
            if val_accuracy > job.best_metric:
                job.best_metric = val_accuracy
            
            metrics = {
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'learning_rate': 0.001,  # Would be actual LR from scheduler
                'gradient_norm': np.random.uniform(0.1, 10.0),
                'throughput': np.random.uniform(100, 1000),
                'gpu_utilization': np.random.uniform(0.5, 1.0),
                'memory_usage': np.random.uniform(0.3, 0.9)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training epoch: {e}")
            return {}
    
    async def _save_checkpoint(self, job: TrainingJob, epoch: int) -> str:
        """Save training checkpoint"""
        try:
            checkpoint_path = f"{job.training_config.output_dir}/checkpoint_epoch_{epoch}.pt"
            
            # In a real implementation, this would save the model state
            # For now, just create a placeholder file
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            with open(checkpoint_path, 'w') as f:
                f.write(f"Checkpoint for epoch {epoch}")
            
            return checkpoint_path
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            return ""
    
    async def _check_early_stopping(self, job: TrainingJob) -> bool:
        """Check early stopping conditions"""
        try:
            if job.training_config.early_stopping_patience <= 0:
                return False
            
            # Check if validation accuracy has improved
            if len(job.metrics.get('val_accuracy', [])) < job.training_config.early_stopping_patience:
                return False
            
            recent_metrics = job.metrics['val_accuracy'][-job.training_config.early_stopping_patience:]
            if len(recent_metrics) < job.training_config.early_stopping_patience:
                return False
            
            # Check if no improvement
            best_recent = max(recent_metrics)
            if best_recent <= job.best_metric:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking early stopping: {e}")
            return False
    
    async def _handle_training_interruption(self, node_id: str, training_id: str):
        """Handle training interruption due to worker failure"""
        try:
            job = self.training_jobs.get(training_id)
            if not job:
                return
            
            # Mark job as failed
            job.status = "failed"
            job.error_message = f"Worker {node_id} failed during training"
            job.end_time = datetime.now()
            
            # Update other workers
            for worker in job.worker_nodes:
                if worker.node_id != node_id:
                    worker.status = "idle"
                    worker.current_training_id = None
            
            # Update metrics
            self.metrics['failed_jobs'] += 1
            self.metrics['active_jobs'] -= 1
            
            logger.warning(f"Training interrupted: {training_id} due to worker {node_id} failure")
            
        except Exception as e:
            logger.error(f"Error handling training interruption: {e}")
    
    async def _monitor_workers(self):
        """Monitor worker nodes"""
        while True:
            try:
                current_time = datetime.now()
                failed_workers = []
                
                for node_id, worker in self.worker_nodes.items():
                    # Check heartbeat
                    if (current_time - worker.last_heartbeat).total_seconds() > 120:  # 2 minutes timeout
                        failed_workers.append(node_id)
                        worker.status = "failed"
                
                # Handle failed workers
                for node_id in failed_workers:
                    await self._handle_worker_failure(node_id)
                
                # Update metrics
                self.metrics['available_workers'] = len([
                    w for w in self.worker_nodes.values() 
                    if w.status == "idle" and w.current_training_id is None
                ])
                self.metrics['active_jobs'] = len([
                    j for j in self.training_jobs.values() 
                    if j.status == "running"
                ])
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring workers: {e}")
                await asyncio.sleep(30)
    
    async def _handle_worker_failure(self, node_id: str):
        """Handle worker failure"""
        try:
            worker = self.worker_nodes.get(node_id)
            if not worker:
                return
            
            # If worker was training, handle interruption
            if worker.current_training_id:
                await self._handle_training_interruption(node_id, worker.current_training_id)
            
            # Remove failed worker
            del self.worker_nodes[node_id]
            self.metrics['total_workers'] -= 1
            
            logger.warning(f"Worker failed: {node_id}")
            
        except Exception as e:
            logger.error(f"Error handling worker failure {node_id}: {e}")
    
    def _validate_worker(self, worker: WorkerNode) -> bool:
        """Validate worker configuration"""
        try:
            if not worker.node_id or not worker.hostname:
                return False
            
            if worker.gpu_count <= 0:
                return False
            
            if worker.cpu_cores <= 0:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating worker: {e}")
            return False
    
    def _validate_training_config(self, config: TrainingConfig) -> bool:
        """Validate training configuration"""
        try:
            if not config.training_id or not config.model_config:
                return False
            
            if config.num_epochs <= 0:
                return False
            
            if config.batch_size <= 0:
                return False
            
            if config.learning_rate <= 0:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating training config: {e}")
            return False
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get training job status"""
        try:
            job = self.training_jobs.get(job_id)
            if not job:
                return {'error': 'Job not found'}
            
            status = {
                'job_id': job.job_id,
                'status': job.status,
                'progress': job.progress,
                'current_epoch': job.current_epoch,
                'best_metric': job.best_metric,
                'start_time': job.start_time.isoformat() if job.start_time else None,
                'end_time': job.end_time.isoformat() if job.end_time else None,
                'error_message': job.error_message,
                'worker_count': len(job.worker_nodes),
                'metrics': job.metrics,
                'checkpoints': job.checkpoints
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting job status: {e}")
            return {'error': str(e)}
    
    async def get_global_metrics(self) -> Dict[str, Any]:
        """Get global orchestrator metrics"""
        try:
            metrics = self.metrics.copy()
            metrics.update({
                'worker_nodes': len(self.worker_nodes),
                'training_jobs': len(self.training_jobs),
                'queue_size': self.job_queue.qsize()
            })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting global metrics: {e}")
            return {}
    
    async def shutdown(self):
        """Shutdown training orchestrator"""
        try:
            # Cancel tasks
            if self.monitoring_task:
                self.monitoring_task.cancel()
            if self.scheduling_task:
                self.scheduling_task.cancel()
            
            # Stop all running jobs
            for job in self.training_jobs.values():
                if job.status == "running":
                    job.status = "cancelled"
                    job.end_time = datetime.now()
            
            logger.info("Training Orchestrator shutdown")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Configure logging
logger = logging.getLogger(__name__)




