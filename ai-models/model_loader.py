"""
Model Loader
============
Enterprise-grade AI model loading service for dynamic model management
"""

import asyncio
import logging
import json
import pickle
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Type
from dataclasses import dataclass, field
from enum import Enum
import os
import warnings
warnings.filterwarnings('ignore')

# AI/ML imports with graceful fallback
try:
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer, AutoConfig
    from sklearn.base import BaseEstimator
    import joblib
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    torch = None

logger = logging.getLogger(__name__)

class LoaderStatus(Enum):
    """Loader status levels"""
    IDLE = "idle"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"
    UNLOADING = "unloading"

class ModelFormat(Enum):
    """Model format types"""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    SKLEARN = "sklearn"
    TRANSFORMERS = "transformers"
    ONNX = "onnx"
    PICKLE = "pickle"
    CUSTOM = "custom"

@dataclass
class ModelInfo:
    """Model information container"""
    model_id: str
    name: str
    format: ModelFormat
    file_path: str
    model_size: int
    loaded_at: datetime
    status: LoaderStatus
    memory_usage: float
    inference_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LoaderMetrics:
    """Loader metrics"""
    total_models_loaded: int
    active_models: int
    average_load_time: float
    average_memory_usage: float
    average_inference_time: float
    cache_hit_rate: float
    error_rate: float
    throughput: float

class ModelLoader:
    """Enterprise-grade AI model loading service"""
    
    def __init__(self):
        self.loaded_models: Dict[str, Any] = {}
        self.model_info: Dict[str, ModelInfo] = {}
        self.model_cache: Dict[str, Any] = {}
        self.status = LoaderStatus.IDLE
        
        # Performance tracking
        self.metrics = LoaderMetrics(
            total_models_loaded=0, active_models=0, average_load_time=0.0,
            average_memory_usage=0.0, average_inference_time=0.0,
            cache_hit_rate=0.0, error_rate=0.0, throughput=0.0
        )
        
        # Loader configuration
        self.config = {
            'max_models_in_memory': 10,
            'cache_size_mb': 1000,
            'auto_unload_timeout': 3600,  # 1 hour
            'enable_caching': True,
            'enable_quantization': True,
            'enable_optimization': True
        }
        
        logger.info("Model Loader initialized")
    
    async def load_model(self, model_id: str, file_path: str, 
                        model_format: ModelFormat, 
                        model_class: Optional[Type] = None,
                        device: str = 'cpu',
                        quantization: bool = False,
                        optimization: bool = False) -> bool:
        """Load a model into memory"""
        try:
            start_time = datetime.now()
            self.status = LoaderStatus.LOADING
            
            # Check if model is already loaded
            if model_id in self.loaded_models:
                logger.info(f"Model {model_id} already loaded")
                return True
            
            # Check cache first
            if self.config['enable_caching'] and model_id in self.model_cache:
                logger.info(f"Loading model {model_id} from cache")
                self.loaded_models[model_id] = self.model_cache[model_id]
                self._update_model_info(model_id, file_path, model_format, start_time)
                return True
            
            # Load model based on format
            model_object = await self._load_model_by_format(
                file_path, model_format, model_class, device, quantization, optimization
            )
            
            # Store loaded model
            self.loaded_models[model_id] = model_object
            
            # Update model info
            self._update_model_info(model_id, file_path, model_format, start_time)
            
            # Update cache
            if self.config['enable_caching']:
                self.model_cache[model_id] = model_object
            
            # Update metrics
            self._update_metrics()
            
            self.status = LoaderStatus.LOADED
            logger.info(f"Model loaded successfully: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            self.status = LoaderStatus.ERROR
            return False
    
    async def unload_model(self, model_id: str) -> bool:
        """Unload a model from memory"""
        try:
            self.status = LoaderStatus.UNLOADING
            
            if model_id not in self.loaded_models:
                logger.warning(f"Model {model_id} not loaded")
                return False
            
            # Remove from loaded models
            del self.loaded_models[model_id]
            
            # Update model info
            if model_id in self.model_info:
                self.model_info[model_id].status = LoaderStatus.IDLE
            
            # Update metrics
            self._update_metrics()
            
            self.status = LoaderStatus.IDLE
            logger.info(f"Model unloaded: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error unloading model {model_id}: {e}")
            return False
    
    async def get_model(self, model_id: str) -> Optional[Any]:
        """Get loaded model by ID"""
        return self.loaded_models.get(model_id)
    
    async def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get model information"""
        return self.model_info.get(model_id)
    
    async def get_loaded_models(self) -> List[str]:
        """Get list of loaded model IDs"""
        return list(self.loaded_models.keys())
    
    async def is_model_loaded(self, model_id: str) -> bool:
        """Check if model is loaded"""
        return model_id in self.loaded_models
    
    async def reload_model(self, model_id: str) -> bool:
        """Reload a model"""
        try:
            if model_id not in self.model_info:
                raise ValueError(f"Model {model_id} not found")
            
            model_info = self.model_info[model_id]
            
            # Unload if currently loaded
            if model_id in self.loaded_models:
                await self.unload_model(model_id)
            
            # Reload model
            success = await self.load_model(
                model_id, 
                model_info.file_path, 
                model_info.format
            )
            
            return success
            
        except Exception as e:
            logger.error(f"Error reloading model {model_id}: {e}")
            return False
    
    async def preload_models(self, model_configs: List[Dict[str, Any]]) -> Dict[str, bool]:
        """Preload multiple models"""
        try:
            results = {}
            
            for config in model_configs:
                model_id = config.get('model_id')
                file_path = config.get('file_path')
                model_format = ModelFormat(config.get('format', 'pytorch'))
                
                success = await self.load_model(model_id, file_path, model_format)
                results[model_id] = success
            
            return results
            
        except Exception as e:
            logger.error(f"Error preloading models: {e}")
            return {}
    
    async def optimize_model(self, model_id: str, optimization_type: str = 'inference') -> bool:
        """Optimize a loaded model"""
        try:
            if model_id not in self.loaded_models:
                raise ValueError(f"Model {model_id} not loaded")
            
            model = self.loaded_models[model_id]
            
            if optimization_type == 'inference' and AI_AVAILABLE and torch:
                # PyTorch optimization
                if hasattr(model, 'eval'):
                    model.eval()
                if hasattr(model, 'half'):
                    model.half()  # Convert to half precision
            
            logger.info(f"Model optimized: {model_id} ({optimization_type})")
            return True
            
        except Exception as e:
            logger.error(f"Error optimizing model {model_id}: {e}")
            return False
    
    async def get_model_performance(self, model_id: str) -> Dict[str, float]:
        """Get model performance metrics"""
        try:
            if model_id not in self.model_info:
                return {}
            
            model_info = self.model_info[model_id]
            
            return {
                'memory_usage': model_info.memory_usage,
                'inference_time': model_info.inference_time,
                'model_size': model_info.model_size,
                'load_time': (datetime.now() - model_info.loaded_at).total_seconds()
            }
            
        except Exception as e:
            logger.error(f"Error getting model performance: {e}")
            return {}
    
    async def _load_model_by_format(self, file_path: str, model_format: ModelFormat,
                                   model_class: Optional[Type], device: str,
                                   quantization: bool, optimization: bool) -> Any:
        """Load model based on format"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Model file not found: {file_path}")
            
            if model_format == ModelFormat.PYTORCH:
                return await self._load_pytorch_model(file_path, device, quantization)
            elif model_format == ModelFormat.TRANSFORMERS:
                return await self._load_transformers_model(file_path, device, quantization)
            elif model_format == ModelFormat.SKLEARN:
                return await self._load_sklearn_model(file_path)
            elif model_format == ModelFormat.PICKLE:
                return await self._load_pickle_model(file_path)
            else:
                raise ValueError(f"Unsupported model format: {model_format}")
                
        except Exception as e:
            logger.error(f"Error loading model by format: {e}")
            raise
    
    async def _load_pytorch_model(self, file_path: str, device: str, quantization: bool) -> Any:
        """Load PyTorch model"""
        try:
            if not AI_AVAILABLE or not torch:
                raise ImportError("PyTorch not available")
            
            # Load model
            model = torch.load(file_path, map_location=device)
            
            # Apply quantization if requested
            if quantization and hasattr(torch.quantization, 'quantize_dynamic'):
                model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
            
            model.eval()
            return model
            
        except Exception as e:
            logger.error(f"Error loading PyTorch model: {e}")
            raise
    
    async def _load_transformers_model(self, file_path: str, device: str, quantization: bool) -> Any:
        """Load Transformers model"""
        try:
            if not AI_AVAILABLE:
                raise ImportError("Transformers not available")
            
            # Load model and tokenizer
            model = AutoModel.from_pretrained(file_path)
            tokenizer = AutoTokenizer.from_pretrained(file_path)
            
            # Move to device
            if device != 'cpu' and torch:
                model = model.to(device)
            
            return {'model': model, 'tokenizer': tokenizer}
            
        except Exception as e:
            logger.error(f"Error loading Transformers model: {e}")
            raise
    
    async def _load_sklearn_model(self, file_path: str) -> Any:
        """Load scikit-learn model"""
        try:
            if not AI_AVAILABLE:
                raise ImportError("scikit-learn not available")
            
            return joblib.load(file_path)
            
        except Exception as e:
            logger.error(f"Error loading scikit-learn model: {e}")
            raise
    
    async def _load_pickle_model(self, file_path: str) -> Any:
        """Load pickle model"""
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
                
        except Exception as e:
            logger.error(f"Error loading pickle model: {e}")
            raise
    
    def _update_model_info(self, model_id: str, file_path: str, 
                          model_format: ModelFormat, start_time: datetime):
        """Update model information"""
        try:
            load_time = (datetime.now() - start_time).total_seconds()
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            
            model_info = ModelInfo(
                model_id=model_id,
                name=os.path.basename(file_path),
                format=model_format,
                file_path=file_path,
                model_size=file_size,
                loaded_at=start_time,
                status=LoaderStatus.LOADED,
                memory_usage=0.0,  # Would need psutil for actual measurement
                inference_time=0.0,  # Would need actual inference testing
                metadata={'load_time': load_time}
            )
            
            self.model_info[model_id] = model_info
            
        except Exception as e:
            logger.error(f"Error updating model info: {e}")
    
    def _update_metrics(self):
        """Update loader metrics"""
        try:
            self.metrics.active_models = len(self.loaded_models)
            self.metrics.total_models_loaded = len(self.model_info)
            
            # Calculate average memory usage
            memory_usages = [info.memory_usage for info in self.model_info.values()]
            self.metrics.average_memory_usage = sum(memory_usages) / len(memory_usages) if memory_usages else 0.0
            
            # Calculate average inference time
            inference_times = [info.inference_time for info in self.model_info.values()]
            self.metrics.average_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0.0
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    async def get_loader_status(self) -> Dict[str, Any]:
        """Get loader status"""
        return {
            'status': self.status.value,
            'loaded_models': len(self.loaded_models),
            'total_models': len(self.model_info),
            'cache_size': len(self.model_cache),
            'metrics': {
                'active_models': self.metrics.active_models,
                'average_memory_usage': self.metrics.average_memory_usage,
                'average_inference_time': self.metrics.average_inference_time,
                'cache_hit_rate': self.metrics.cache_hit_rate,
                'error_rate': self.metrics.error_rate
            },
            'config': self.config,
            'ai_available': AI_AVAILABLE
        }

# Global instance
model_loader = ModelLoader()




