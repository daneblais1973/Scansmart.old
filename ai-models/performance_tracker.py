"""
Performance Tracker
==================
Enterprise-grade AI model performance tracking service
"""

import asyncio
import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import statistics
import warnings
warnings.filterwarnings('ignore')

# AI/ML imports with graceful fallback
try:
    import torch
    import torch.nn as nn
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    torch = None

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Metric type categories"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    INFERENCE_TIME = "inference_time"
    MEMORY_USAGE = "memory_usage"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    CUSTOM = "custom"

class PerformanceLevel(Enum):
    """Performance level categories"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"

@dataclass
class PerformanceMetric:
    """Performance metric container"""
    metric_id: str
    model_id: str
    metric_type: MetricType
    value: float
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceReport:
    """Performance report container"""
    report_id: str
    model_id: str
    start_time: datetime
    end_time: datetime
    metrics: List[PerformanceMetric]
    summary: Dict[str, Any]
    recommendations: List[str]
    performance_level: PerformanceLevel
    generated_at: datetime = field(default_factory=datetime.now)

@dataclass
class PerformanceThresholds:
    """Performance thresholds container"""
    accuracy_min: float = 0.8
    precision_min: float = 0.7
    recall_min: float = 0.7
    f1_score_min: float = 0.7
    inference_time_max: float = 1.0  # seconds
    memory_usage_max: float = 1000.0  # MB
    throughput_min: float = 100.0  # requests/second
    latency_max: float = 0.5  # seconds

@dataclass
class PerformanceMetrics:
    """Performance tracking metrics"""
    total_models_tracked: int
    total_metrics_collected: int
    average_accuracy: float
    average_inference_time: float
    average_memory_usage: float
    performance_alerts: int
    models_exceeding_thresholds: int
    tracking_accuracy: float

class PerformanceTracker:
    """Enterprise-grade AI model performance tracking service"""
    
    def __init__(self):
        self.metrics: Dict[str, List[PerformanceMetric]] = {}
        self.reports: Dict[str, PerformanceReport] = {}
        self.thresholds = PerformanceThresholds()
        self.alerts: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.tracking_metrics = PerformanceMetrics(
            total_models_tracked=0, total_metrics_collected=0, average_accuracy=0.0,
            average_inference_time=0.0, average_memory_usage=0.0, performance_alerts=0,
            models_exceeding_thresholds=0, tracking_accuracy=0.0
        )
        
        # Tracking configuration
        self.config = {
            'enable_real_time_tracking': True,
            'enable_alerting': True,
            'enable_reporting': True,
            'reporting_interval': 3600,  # 1 hour
            'alert_threshold': 0.8,
            'max_metrics_per_model': 10000,
            'enable_anomaly_detection': True
        }
        
        # Start background tasks
        asyncio.create_task(self._reporting_task())
        asyncio.create_task(self._alerting_task())
        
        logger.info("Performance Tracker initialized")
    
    async def track_metric(self, model_id: str, metric_type: MetricType, 
                         value: float, context: Optional[Dict[str, Any]] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """Track a performance metric"""
        try:
            metric_id = str(uuid.uuid4())
            
            # Create metric
            metric = PerformanceMetric(
                metric_id=metric_id,
                model_id=model_id,
                metric_type=metric_type,
                value=value,
                timestamp=datetime.now(),
                context=context or {},
                metadata=metadata or {}
            )
            
            # Store metric
            if model_id not in self.metrics:
                self.metrics[model_id] = []
            
            self.metrics[model_id].append(metric)
            
            # Limit metrics per model
            if len(self.metrics[model_id]) > self.config['max_metrics_per_model']:
                self.metrics[model_id] = self.metrics[model_id][-self.config['max_metrics_per_model']:]
            
            # Update tracking metrics
            self._update_tracking_metrics()
            
            # Check thresholds
            await self._check_thresholds(model_id, metric)
            
            logger.debug(f"Metric tracked: {model_id} - {metric_type.value}: {value}")
            return metric_id
            
        except Exception as e:
            logger.error(f"Error tracking metric: {e}")
            raise
    
    async def get_model_metrics(self, model_id: str, 
                               metric_type: Optional[MetricType] = None,
                               start_time: Optional[datetime] = None,
                               end_time: Optional[datetime] = None) -> List[PerformanceMetric]:
        """Get metrics for a model"""
        try:
            if model_id not in self.metrics:
                return []
            
            metrics = self.metrics[model_id]
            
            # Filter by metric type
            if metric_type:
                metrics = [m for m in metrics if m.metric_type == metric_type]
            
            # Filter by time range
            if start_time:
                metrics = [m for m in metrics if m.timestamp >= start_time]
            
            if end_time:
                metrics = [m for m in metrics if m.timestamp <= end_time]
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting model metrics: {e}")
            return []
    
    async def get_model_performance_summary(self, model_id: str, 
                                          hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for a model"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            metrics = await self.get_model_metrics(model_id, start_time=start_time, end_time=end_time)
            
            if not metrics:
                return {}
            
            # Group metrics by type
            metrics_by_type = {}
            for metric in metrics:
                if metric.metric_type not in metrics_by_type:
                    metrics_by_type[metric.metric_type] = []
                metrics_by_type[metric.metric_type].append(metric.value)
            
            # Calculate statistics
            summary = {}
            for metric_type, values in metrics_by_type.items():
                if values:
                    summary[metric_type.value] = {
                        'count': len(values),
                        'mean': statistics.mean(values),
                        'median': statistics.median(values),
                        'min': min(values),
                        'max': max(values),
                        'std': statistics.stdev(values) if len(values) > 1 else 0.0
                    }
            
            # Calculate performance level
            performance_level = await self._calculate_performance_level(model_id, summary)
            summary['performance_level'] = performance_level.value
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {}
    
    async def generate_performance_report(self, model_id: str, 
                                        start_time: Optional[datetime] = None,
                                        end_time: Optional[datetime] = None) -> str:
        """Generate performance report for a model"""
        try:
            if start_time is None:
                start_time = datetime.now() - timedelta(hours=24)
            if end_time is None:
                end_time = datetime.now()
            
            # Get metrics
            metrics = await self.get_model_metrics(model_id, start_time=start_time, end_time=end_time)
            
            # Calculate summary
            summary = await self.get_model_performance_summary(model_id)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(model_id, summary)
            
            # Determine performance level
            performance_level = await self._calculate_performance_level(model_id, summary)
            
            # Create report
            report_id = str(uuid.uuid4())
            report = PerformanceReport(
                report_id=report_id,
                model_id=model_id,
                start_time=start_time,
                end_time=end_time,
                metrics=metrics,
                summary=summary,
                recommendations=recommendations,
                performance_level=performance_level
            )
            
            # Store report
            self.reports[report_id] = report
            
            logger.info(f"Performance report generated: {report_id}")
            return report_id
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            raise
    
    async def get_performance_report(self, report_id: str) -> Optional[PerformanceReport]:
        """Get performance report by ID"""
        return self.reports.get(report_id)
    
    async def set_performance_thresholds(self, thresholds: PerformanceThresholds) -> bool:
        """Set performance thresholds"""
        try:
            self.thresholds = thresholds
            logger.info("Performance thresholds updated")
            return True
            
        except Exception as e:
            logger.error(f"Error setting performance thresholds: {e}")
            return False
    
    async def get_performance_alerts(self, model_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get performance alerts"""
        try:
            if model_id:
                return [alert for alert in self.alerts if alert.get('model_id') == model_id]
            else:
                return self.alerts
                
        except Exception as e:
            logger.error(f"Error getting performance alerts: {e}")
            return []
    
    async def _check_thresholds(self, model_id: str, metric: PerformanceMetric):
        """Check if metric exceeds thresholds"""
        try:
            if not self.config['enable_alerting']:
                return
            
            exceeded = False
            threshold_info = {}
            
            # Check specific thresholds
            if metric.metric_type == MetricType.ACCURACY and metric.value < self.thresholds.accuracy_min:
                exceeded = True
                threshold_info['threshold'] = self.thresholds.accuracy_min
                threshold_info['actual'] = metric.value
            
            elif metric.metric_type == MetricType.INFERENCE_TIME and metric.value > self.thresholds.inference_time_max:
                exceeded = True
                threshold_info['threshold'] = self.thresholds.inference_time_max
                threshold_info['actual'] = metric.value
            
            elif metric.metric_type == MetricType.MEMORY_USAGE and metric.value > self.thresholds.memory_usage_max:
                exceeded = True
                threshold_info['threshold'] = self.thresholds.memory_usage_max
                threshold_info['actual'] = metric.value
            
            if exceeded:
                alert = {
                    'alert_id': str(uuid.uuid4()),
                    'model_id': model_id,
                    'metric_type': metric.metric_type.value,
                    'value': metric.value,
                    'threshold': threshold_info['threshold'],
                    'timestamp': metric.timestamp,
                    'severity': 'warning'
                }
                
                self.alerts.append(alert)
                self.tracking_metrics.performance_alerts += 1
                
                logger.warning(f"Performance threshold exceeded: {model_id} - {metric.metric_type.value}")
                
        except Exception as e:
            logger.error(f"Error checking thresholds: {e}")
    
    async def _calculate_performance_level(self, model_id: str, summary: Dict[str, Any]) -> PerformanceLevel:
        """Calculate performance level based on metrics"""
        try:
            score = 0.0
            total_checks = 0
            
            # Check accuracy
            if 'accuracy' in summary:
                accuracy = summary['accuracy']['mean']
                if accuracy >= 0.9:
                    score += 1.0
                elif accuracy >= 0.8:
                    score += 0.8
                elif accuracy >= 0.7:
                    score += 0.6
                else:
                    score += 0.4
                total_checks += 1
            
            # Check inference time
            if 'inference_time' in summary:
                inference_time = summary['inference_time']['mean']
                if inference_time <= 0.1:
                    score += 1.0
                elif inference_time <= 0.5:
                    score += 0.8
                elif inference_time <= 1.0:
                    score += 0.6
                else:
                    score += 0.4
                total_checks += 1
            
            # Check memory usage
            if 'memory_usage' in summary:
                memory_usage = summary['memory_usage']['mean']
                if memory_usage <= 100:
                    score += 1.0
                elif memory_usage <= 500:
                    score += 0.8
                elif memory_usage <= 1000:
                    score += 0.6
                else:
                    score += 0.4
                total_checks += 1
            
            if total_checks == 0:
                return PerformanceLevel.FAIR
            
            average_score = score / total_checks
            
            if average_score >= 0.9:
                return PerformanceLevel.EXCELLENT
            elif average_score >= 0.8:
                return PerformanceLevel.GOOD
            elif average_score >= 0.6:
                return PerformanceLevel.FAIR
            elif average_score >= 0.4:
                return PerformanceLevel.POOR
            else:
                return PerformanceLevel.CRITICAL
                
        except Exception as e:
            logger.error(f"Error calculating performance level: {e}")
            return PerformanceLevel.FAIR
    
    async def _generate_recommendations(self, model_id: str, summary: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations"""
        try:
            recommendations = []
            
            # Check accuracy
            if 'accuracy' in summary and summary['accuracy']['mean'] < 0.8:
                recommendations.append("Consider retraining the model or adjusting hyperparameters to improve accuracy")
            
            # Check inference time
            if 'inference_time' in summary and summary['inference_time']['mean'] > 1.0:
                recommendations.append("Consider model optimization or quantization to reduce inference time")
            
            # Check memory usage
            if 'memory_usage' in summary and summary['memory_usage']['mean'] > 1000:
                recommendations.append("Consider model compression or pruning to reduce memory usage")
            
            # Check throughput
            if 'throughput' in summary and summary['throughput']['mean'] < 100:
                recommendations.append("Consider batch processing or model optimization to improve throughput")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []
    
    def _update_tracking_metrics(self):
        """Update tracking metrics"""
        try:
            self.tracking_metrics.total_models_tracked = len(self.metrics)
            self.tracking_metrics.total_metrics_collected = sum(len(metrics) for metrics in self.metrics.values())
            
            # Calculate averages
            all_metrics = []
            for metrics in self.metrics.values():
                all_metrics.extend(metrics)
            
            if all_metrics:
                accuracy_metrics = [m.value for m in all_metrics if m.metric_type == MetricType.ACCURACY]
                if accuracy_metrics:
                    self.tracking_metrics.average_accuracy = statistics.mean(accuracy_metrics)
                
                inference_metrics = [m.value for m in all_metrics if m.metric_type == MetricType.INFERENCE_TIME]
                if inference_metrics:
                    self.tracking_metrics.average_inference_time = statistics.mean(inference_metrics)
                
                memory_metrics = [m.value for m in all_metrics if m.metric_type == MetricType.MEMORY_USAGE]
                if memory_metrics:
                    self.tracking_metrics.average_memory_usage = statistics.mean(memory_metrics)
            
        except Exception as e:
            logger.error(f"Error updating tracking metrics: {e}")
    
    async def _reporting_task(self):
        """Background reporting task"""
        try:
            while True:
                await asyncio.sleep(self.config['reporting_interval'])
                
                if self.config['enable_reporting']:
                    # Generate reports for all models
                    for model_id in self.metrics.keys():
                        try:
                            await self.generate_performance_report(model_id)
                        except Exception as e:
                            logger.error(f"Error generating report for {model_id}: {e}")
                
        except Exception as e:
            logger.error(f"Error in reporting task: {e}")
    
    async def _alerting_task(self):
        """Background alerting task"""
        try:
            while True:
                await asyncio.sleep(60)  # Check every minute
                
                if self.config['enable_alerting']:
                    # Process alerts
                    recent_alerts = [alert for alert in self.alerts 
                                   if datetime.now() - alert['timestamp'] < timedelta(minutes=5)]
                    
                    if recent_alerts:
                        logger.info(f"Processing {len(recent_alerts)} recent alerts")
                
        except Exception as e:
            logger.error(f"Error in alerting task: {e}")
    
    async def get_tracker_status(self) -> Dict[str, Any]:
        """Get tracker status"""
        return {
            'total_models_tracked': self.tracking_metrics.total_models_tracked,
            'total_metrics_collected': self.tracking_metrics.total_metrics_collected,
            'average_accuracy': self.tracking_metrics.average_accuracy,
            'average_inference_time': self.tracking_metrics.average_inference_time,
            'average_memory_usage': self.tracking_metrics.average_memory_usage,
            'performance_alerts': self.tracking_metrics.performance_alerts,
            'models_exceeding_thresholds': self.tracking_metrics.models_exceeding_thresholds,
            'tracking_accuracy': self.tracking_metrics.tracking_accuracy,
            'config': self.config,
            'ai_available': AI_AVAILABLE
        }

# Global instance
performance_tracker = PerformanceTracker()




