"""
Model Serving Infrastructure
Advanced model serving with load balancing, auto-scaling, and performance optimization
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


class ModelType(Enum):
    """Model type categories"""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    ONNX = "onnx"
    SKLEARN = "sklearn"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    TRANSFORMERS = "transformers"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


class ServingStrategy(Enum):
    """Model serving strategies"""
    SINGLE_INSTANCE = "single_instance"
    LOAD_BALANCED = "load_balanced"
    AUTO_SCALING = "auto_scaling"
    CANARY = "canary"
    BLUE_GREEN = "blue_green"
    A_B_TESTING = "a_b_testing"


class ScalingPolicy(Enum):
    """Auto-scaling policies"""
    CPU_BASED = "cpu_based"
    MEMORY_BASED = "memory_based"
    REQUEST_BASED = "request_based"
    CUSTOM_METRICS = "custom_metrics"
    HYBRID = "hybrid"


@dataclass
class ModelConfig:
    """Model configuration"""
    model_id: str
    name: str
    model_type: ModelType
    version: str
    model_path: str
    config_path: Optional[str] = None
    weights_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    requirements: List[str] = field(default_factory=list)
    dependencies: Dict[str, str] = field(default_factory=dict)


@dataclass
class ServingConfig:
    """Model serving configuration"""
    serving_id: str
    model_config: ModelConfig
    serving_strategy: ServingStrategy
    scaling_policy: ScalingPolicy
    min_instances: int = 1
    max_instances: int = 10
    target_cpu_utilization: float = 70.0
    target_memory_utilization: float = 80.0
    target_request_rate: float = 100.0
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    cooldown_period: int = 300  # seconds
    health_check_interval: int = 30  # seconds
    timeout: float = 30.0
    batch_size: int = 1
    max_batch_wait_time: float = 0.1


@dataclass
class ModelInstance:
    """Model instance configuration"""
    instance_id: str
    serving_id: str
    model: Any
    status: str = "starting"
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    request_count: int = 0
    error_count: int = 0
    last_request: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_health_check: datetime = field(default_factory=datetime.now)


@dataclass
class PredictionRequest:
    """Prediction request"""
    request_id: str
    serving_id: str
    input_data: Any
    batch_id: Optional[str] = None
    priority: int = 0
    timeout: float = 30.0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionResponse:
    """Prediction response"""
    response_id: str
    request_id: str
    serving_id: str
    predictions: Any
    confidence_scores: Optional[Dict[str, float]] = None
    processing_time: float = 0.0
    model_version: str = ""
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelLoader(ABC):
    """Abstract model loader interface"""
    
    @abstractmethod
    async def load_model(self, config: ModelConfig) -> Any:
        """Load model from configuration"""
        pass
    
    @abstractmethod
    async def unload_model(self, model: Any) -> bool:
        """Unload model"""
        pass
    
    @abstractmethod
    async def predict(self, model: Any, input_data: Any) -> Any:
        """Run prediction"""
        pass


class PyTorchModelLoader(ModelLoader):
    """PyTorch model loader"""
    
    async def load_model(self, config: ModelConfig) -> Any:
        """Load PyTorch model"""
        try:
            import torch
            
            # Load model
            model = torch.load(config.model_path, map_location='cpu')
            model.eval()
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading PyTorch model: {e}")
            return None
    
    async def unload_model(self, model: Any) -> bool:
        """Unload PyTorch model"""
        try:
            del model
            return True
        except Exception as e:
            logger.error(f"Error unloading PyTorch model: {e}")
            return False
    
    async def predict(self, model: Any, input_data: Any) -> Any:
        """Run PyTorch prediction"""
        try:
            import torch
            
            with torch.no_grad():
                if isinstance(input_data, np.ndarray):
                    input_tensor = torch.from_numpy(input_data)
                else:
                    input_tensor = input_data
                
                predictions = model(input_tensor)
                
                if isinstance(predictions, torch.Tensor):
                    return predictions.cpu().numpy()
                else:
                    return predictions
                    
        except Exception as e:
            logger.error(f"Error running PyTorch prediction: {e}")
            return None


class TensorFlowModelLoader(ModelLoader):
    """TensorFlow model loader"""
    
    async def load_model(self, config: ModelConfig) -> Any:
        """Load TensorFlow model"""
        try:
            import tensorflow as tf
            
            # Load model
            model = tf.keras.models.load_model(config.model_path)
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading TensorFlow model: {e}")
            return None
    
    async def unload_model(self, model: Any) -> bool:
        """Unload TensorFlow model"""
        try:
            del model
            return True
        except Exception as e:
            logger.error(f"Error unloading TensorFlow model: {e}")
            return False
    
    async def predict(self, model: Any, input_data: Any) -> Any:
        """Run TensorFlow prediction"""
        try:
            predictions = model.predict(input_data)
            return predictions
            
        except Exception as e:
            logger.error(f"Error running TensorFlow prediction: {e}")
            return None


class SklearnModelLoader(ModelLoader):
    """Scikit-learn model loader"""
    
    async def load_model(self, config: ModelConfig) -> Any:
        """Load scikit-learn model"""
        try:
            import pickle
            
            with open(config.model_path, 'rb') as f:
                model = pickle.load(f)
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading scikit-learn model: {e}")
            return None
    
    async def unload_model(self, model: Any) -> bool:
        """Unload scikit-learn model"""
        try:
            del model
            return True
        except Exception as e:
            logger.error(f"Error unloading scikit-learn model: {e}")
            return False
    
    async def predict(self, model: Any, input_data: Any) -> Any:
        """Run scikit-learn prediction"""
        try:
            predictions = model.predict(input_data)
            return predictions
            
        except Exception as e:
            logger.error(f"Error running scikit-learn prediction: {e}")
            return None


class ModelServer:
    """Advanced model server"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.servings: Dict[str, ServingConfig] = {}
        self.instances: Dict[str, ModelInstance] = {}
        self.loaders: Dict[ModelType, ModelLoader] = {}
        self.request_queue = queue.PriorityQueue()
        self.response_queue = queue.Queue()
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_latency': 0.0,
            'throughput': 0.0,
            'active_instances': 0,
            'total_servings': 0
        }
        
        # Initialize model loaders
        self._initialize_loaders()
        
        # Start monitoring
        self.monitoring_task = asyncio.create_task(self._monitor_instances())
        self.scaling_task = asyncio.create_task(self._auto_scaling())
        self.processing_task = asyncio.create_task(self._process_requests())
        
        logger.info("Model Server initialized")
    
    def _initialize_loaders(self):
        """Initialize model loaders"""
        try:
            self.loaders[ModelType.PYTORCH] = PyTorchModelLoader()
            self.loaders[ModelType.TENSORFLOW] = TensorFlowModelLoader()
            self.loaders[ModelType.SKLEARN] = SklearnModelLoader()
            
            logger.info(f"Initialized {len(self.loaders)} model loaders")
            
        except Exception as e:
            logger.error(f"Error initializing loaders: {e}")
    
    async def deploy_model(self, serving_config: ServingConfig) -> bool:
        """Deploy model for serving"""
        try:
            # Validate configuration
            if not self._validate_serving_config(serving_config):
                logger.error(f"Invalid serving configuration: {serving_config.serving_id}")
                return False
            
            # Store serving configuration
            self.servings[serving_config.serving_id] = serving_config
            
            # Create initial instances
            await self._create_initial_instances(serving_config)
            
            self.metrics['total_servings'] += 1
            logger.info(f"Model deployed: {serving_config.serving_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deploying model {serving_config.serving_id}: {e}")
            return False
    
    async def _create_initial_instances(self, serving_config: ServingConfig):
        """Create initial model instances"""
        try:
            min_instances = serving_config.min_instances
            
            for i in range(min_instances):
                instance_id = f"{serving_config.serving_id}_instance_{i}"
                await self._create_instance(instance_id, serving_config)
            
            logger.info(f"Created {min_instances} initial instances for {serving_config.serving_id}")
            
        except Exception as e:
            logger.error(f"Error creating initial instances: {e}")
    
    async def _create_instance(self, instance_id: str, serving_config: ServingConfig) -> bool:
        """Create model instance"""
        try:
            # Get model loader
            loader = self.loaders.get(serving_config.model_config.model_type)
            if not loader:
                logger.error(f"No loader for model type: {serving_config.model_config.model_type}")
                return False
            
            # Load model
            model = await loader.load_model(serving_config.model_config)
            if not model:
                logger.error(f"Failed to load model: {serving_config.model_config.model_id}")
                return False
            
            # Create instance
            instance = ModelInstance(
                instance_id=instance_id,
                serving_id=serving_config.serving_id,
                model=model
            )
            
            self.instances[instance_id] = instance
            self.metrics['active_instances'] += 1
            
            logger.info(f"Instance created: {instance_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating instance {instance_id}: {e}")
            return False
    
    async def predict(self, serving_id: str, input_data: Any, 
                     priority: int = 0, timeout: float = 30.0) -> PredictionResponse:
        """Run prediction"""
        try:
            # Validate serving
            if serving_id not in self.servings:
                return PredictionResponse(
                    response_id="",
                    request_id="",
                    serving_id=serving_id,
                    predictions=None,
                    success=False,
                    error_message="Serving not found"
                )
            
            # Create request
            request_id = str(uuid.uuid4())
            request = PredictionRequest(
                request_id=request_id,
                serving_id=serving_id,
                input_data=input_data,
                priority=priority,
                timeout=timeout
            )
            
            # Add to queue
            self.request_queue.put((priority, request))
            self.metrics['total_requests'] += 1
            
            # Wait for response
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    response = self.response_queue.get_nowait()
                    if response.request_id == request_id:
                        return response
                except queue.Empty:
                    await asyncio.sleep(0.01)
            
            # Timeout
            return PredictionResponse(
                response_id="",
                request_id=request_id,
                serving_id=serving_id,
                predictions=None,
                success=False,
                error_message="Request timeout"
            )
            
        except Exception as e:
            logger.error(f"Error running prediction: {e}")
            return PredictionResponse(
                response_id="",
                request_id="",
                serving_id=serving_id,
                predictions=None,
                success=False,
                error_message=str(e)
            )
    
    async def _process_requests(self):
        """Process prediction requests"""
        while True:
            try:
                # Get request from queue
                try:
                    priority, request = self.request_queue.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.01)
                    continue
                
                # Find available instance
                instance = await self._find_available_instance(request.serving_id)
                if not instance:
                    # No available instance, requeue request
                    self.request_queue.put((priority, request))
                    await asyncio.sleep(0.1)
                    continue
                
                # Process request
                response = await self._process_single_request(request, instance)
                
                # Add response to queue
                self.response_queue.put(response)
                
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                await asyncio.sleep(0.1)
    
    async def _find_available_instance(self, serving_id: str) -> Optional[ModelInstance]:
        """Find available model instance"""
        try:
            available_instances = [
                instance for instance in self.instances.values()
                if (instance.serving_id == serving_id and 
                    instance.status == "ready" and
                    instance.cpu_usage < 90.0 and
                    instance.memory_usage < 90.0)
            ]
            
            if not available_instances:
                return None
            
            # Select instance with least load
            return min(available_instances, key=lambda x: x.cpu_usage + x.memory_usage)
            
        except Exception as e:
            logger.error(f"Error finding available instance: {e}")
            return None
    
    async def _process_single_request(self, request: PredictionRequest, 
                                    instance: ModelInstance) -> PredictionResponse:
        """Process single prediction request"""
        try:
            start_time = time.time()
            
            # Get model loader
            serving_config = self.servings[request.serving_id]
            loader = self.loaders[serving_config.model_config.model_type]
            
            # Run prediction
            predictions = await loader.predict(instance.model, request.input_data)
            
            processing_time = time.time() - start_time
            
            # Update instance metrics
            instance.request_count += 1
            instance.last_request = datetime.now()
            
            # Update global metrics
            self.metrics['successful_requests'] += 1
            self.metrics['average_latency'] = (
                (self.metrics['average_latency'] * (self.metrics['successful_requests'] - 1) + processing_time) /
                self.metrics['successful_requests']
            )
            
            return PredictionResponse(
                response_id=str(uuid.uuid4()),
                request_id=request.request_id,
                serving_id=request.serving_id,
                predictions=predictions,
                processing_time=processing_time,
                model_version=serving_config.model_config.version,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error processing request {request.request_id}: {e}")
            
            # Update metrics
            self.metrics['failed_requests'] += 1
            instance.error_count += 1
            
            return PredictionResponse(
                response_id=str(uuid.uuid4()),
                request_id=request.request_id,
                serving_id=request.serving_id,
                predictions=None,
                success=False,
                error_message=str(e)
            )
    
    async def _monitor_instances(self):
        """Monitor model instances"""
        while True:
            try:
                for instance in self.instances.values():
                    # Update instance status
                    await self._update_instance_status(instance)
                    
                    # Health check
                    if instance.status == "ready":
                        await self._health_check_instance(instance)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring instances: {e}")
                await asyncio.sleep(30)
    
    async def _update_instance_status(self, instance: ModelInstance):
        """Update instance status"""
        try:
            # Get system metrics
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent
            
            instance.cpu_usage = cpu_usage
            instance.memory_usage = memory_usage
            
            # Update status based on metrics
            if cpu_usage > 95 or memory_usage > 95:
                instance.status = "overloaded"
            elif cpu_usage < 10 and memory_usage < 10:
                instance.status = "idle"
            else:
                instance.status = "ready"
                
        except Exception as e:
            logger.error(f"Error updating instance status: {e}")
    
    async def _health_check_instance(self, instance: ModelInstance):
        """Health check for instance"""
        try:
            # Simple health check - try to run a dummy prediction
            serving_config = self.servings[instance.serving_id]
            loader = self.loaders[serving_config.model_config.model_type]
            
            # Create dummy input
            dummy_input = np.random.rand(1, 10)  # Adjust based on model requirements
            
            # Run dummy prediction
            start_time = time.time()
            await loader.predict(instance.model, dummy_input)
            health_check_time = time.time() - start_time
            
            # Update last health check
            instance.last_health_check = datetime.now()
            
            # If health check takes too long, mark as unhealthy
            if health_check_time > 5.0:  # 5 seconds timeout
                instance.status = "unhealthy"
                logger.warning(f"Instance {instance.instance_id} health check timeout")
            
        except Exception as e:
            logger.error(f"Health check failed for instance {instance.instance_id}: {e}")
            instance.status = "unhealthy"
    
    async def _auto_scaling(self):
        """Auto-scaling logic"""
        while True:
            try:
                for serving_id, serving_config in self.servings.items():
                    await self._scale_serving(serving_config)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in auto-scaling: {e}")
                await asyncio.sleep(60)
    
    async def _scale_serving(self, serving_config: ServingConfig):
        """Scale serving based on metrics"""
        try:
            # Get instances for this serving
            serving_instances = [
                instance for instance in self.instances.values()
                if instance.serving_id == serving_config.serving_id
            ]
            
            if not serving_instances:
                return
            
            # Calculate average metrics
            avg_cpu = np.mean([i.cpu_usage for i in serving_instances])
            avg_memory = np.mean([i.memory_usage for i in serving_instances])
            total_requests = sum(i.request_count for i in serving_instances)
            
            # Scale up conditions
            if (len(serving_instances) < serving_config.max_instances and
                (avg_cpu > serving_config.scale_up_threshold * 100 or
                 avg_memory > serving_config.scale_up_threshold * 100 or
                 total_requests > serving_config.target_request_rate)):
                
                # Create new instance
                instance_id = f"{serving_config.serving_id}_instance_{len(serving_instances)}"
                await self._create_instance(instance_id, serving_config)
                logger.info(f"Scaled up serving {serving_config.serving_id}")
            
            # Scale down conditions
            elif (len(serving_instances) > serving_config.min_instances and
                  avg_cpu < serving_config.scale_down_threshold * 100 and
                  avg_memory < serving_config.scale_down_threshold * 100 and
                  total_requests < serving_config.target_request_rate * 0.5):
                
                # Remove least used instance
                least_used = min(serving_instances, key=lambda x: x.request_count)
                await self._remove_instance(least_used.instance_id)
                logger.info(f"Scaled down serving {serving_config.serving_id}")
            
        except Exception as e:
            logger.error(f"Error scaling serving {serving_config.serving_id}: {e}")
    
    async def _remove_instance(self, instance_id: str) -> bool:
        """Remove model instance"""
        try:
            instance = self.instances.get(instance_id)
            if not instance:
                return False
            
            # Get model loader
            serving_config = self.servings[instance.serving_id]
            loader = self.loaders[serving_config.model_config.model_type]
            
            # Unload model
            await loader.unload_model(instance.model)
            
            # Remove instance
            del self.instances[instance_id]
            self.metrics['active_instances'] -= 1
            
            logger.info(f"Instance removed: {instance_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing instance {instance_id}: {e}")
            return False
    
    def _validate_serving_config(self, serving_config: ServingConfig) -> bool:
        """Validate serving configuration"""
        try:
            if not serving_config.serving_id or not serving_config.model_config:
                return False
            
            if serving_config.min_instances < 1:
                return False
            
            if serving_config.max_instances < serving_config.min_instances:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating serving config: {e}")
            return False
    
    async def get_serving_metrics(self, serving_id: str) -> Dict[str, Any]:
        """Get serving metrics"""
        try:
            serving_instances = [
                instance for instance in self.instances.values()
                if instance.serving_id == serving_id
            ]
            
            if not serving_instances:
                return {}
            
            metrics = {
                'serving_id': serving_id,
                'instance_count': len(serving_instances),
                'total_requests': sum(i.request_count for i in serving_instances),
                'total_errors': sum(i.error_count for i in serving_instances),
                'average_cpu_usage': np.mean([i.cpu_usage for i in serving_instances]),
                'average_memory_usage': np.mean([i.memory_usage for i in serving_instances]),
                'healthy_instances': len([i for i in serving_instances if i.status == "ready"]),
                'unhealthy_instances': len([i for i in serving_instances if i.status == "unhealthy"])
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting serving metrics: {e}")
            return {}
    
    async def get_global_metrics(self) -> Dict[str, Any]:
        """Get global server metrics"""
        try:
            metrics = self.metrics.copy()
            metrics.update({
                'active_servings': len(self.servings),
                'total_instances': len(self.instances),
                'queue_size': self.request_queue.qsize(),
                'response_queue_size': self.response_queue.qsize()
            })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting global metrics: {e}")
            return {}
    
    async def shutdown(self):
        """Shutdown model server"""
        try:
            # Cancel tasks
            if self.monitoring_task:
                self.monitoring_task.cancel()
            if self.scaling_task:
                self.scaling_task.cancel()
            if self.processing_task:
                self.processing_task.cancel()
            
            # Unload all models
            for instance in self.instances.values():
                serving_config = self.servings[instance.serving_id]
                loader = self.loaders[serving_config.model_config.model_type]
                await loader.unload_model(instance.model)
            
            logger.info("Model Server shutdown")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Configure logging
logger = logging.getLogger(__name__)




