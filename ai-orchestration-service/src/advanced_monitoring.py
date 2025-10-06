"""
Advanced Monitoring and Distributed Tracing System
================================================
Enterprise-grade monitoring, tracing, and observability system.
"""

import asyncio
import logging
import time
import uuid
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import psutil
import os
from collections import defaultdict, deque
import traceback
import sys

logger = logging.getLogger(__name__)

class TraceType(Enum):
    """Types of distributed traces"""
    REQUEST = "request"
    DATABASE = "database"
    API_CALL = "api_call"
    ML_INFERENCE = "ml_inference"
    DATA_PROCESSING = "data_processing"
    CACHE_OPERATION = "cache_operation"
    BACKGROUND_TASK = "background_task"
    ERROR = "error"
    PERFORMANCE = "performance"

class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class TraceSpan:
    """Represents a single span in a distributed trace"""
    span_id: str
    trace_id: str
    parent_span_id: Optional[str]
    operation_name: str
    trace_type: TraceType
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "started"
    error: Optional[str] = None
    
    def finish(self, status: str = "completed", error: Optional[str] = None):
        """Finish the span and calculate duration"""
        self.end_time = datetime.now()
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.status = status
        self.error = error

@dataclass
class Metric:
    """Represents a single metric"""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None

@dataclass
class Alert:
    """Represents an alert"""
    alert_id: str
    name: str
    level: AlertLevel
    message: str
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None

class DistributedTracer:
    """Distributed tracing system"""
    
    def __init__(self):
        self.active_traces: Dict[str, List[TraceSpan]] = defaultdict(list)
        self.completed_traces: Dict[str, List[TraceSpan]] = defaultdict(list)
        self.trace_metrics: Dict[str, Any] = defaultdict(int)
        self.lock = threading.Lock()
        
    def start_trace(self, operation_name: str, trace_type: TraceType, 
                   parent_span_id: Optional[str] = None, tags: Dict[str, Any] = None) -> str:
        """Start a new trace or span"""
        trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())
        
        span = TraceSpan(
            span_id=span_id,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            trace_type=trace_type,
            start_time=datetime.now(),
            tags=tags or {}
        )
        
        with self.lock:
            self.active_traces[trace_id].append(span)
            self.trace_metrics['active_traces'] = len(self.active_traces)
            self.trace_metrics['total_spans'] += 1
        
        return span_id
    
    def finish_span(self, span_id: str, status: str = "completed", error: Optional[str] = None):
        """Finish a span"""
        with self.lock:
            for trace_id, spans in self.active_traces.items():
                for span in spans:
                    if span.span_id == span_id:
                        span.finish(status, error)
                        # Move to completed traces
                        self.completed_traces[trace_id].append(span)
                        spans.remove(span)
                        
                        if not spans:  # No more active spans in this trace
                            del self.active_traces[trace_id]
                        
                        self.trace_metrics['completed_traces'] += 1
                        break
    
    def add_span_log(self, span_id: str, message: str, level: str = "info", 
                    fields: Dict[str, Any] = None):
        """Add a log entry to a span"""
        with self.lock:
            for spans in self.active_traces.values():
                for span in spans:
                    if span.span_id == span_id:
                        span.logs.append({
                            'timestamp': datetime.now(),
                            'message': message,
                            'level': level,
                            'fields': fields or {}
                        })
                        break
    
    def get_trace(self, trace_id: str) -> Optional[List[TraceSpan]]:
        """Get a completed trace"""
        return self.completed_traces.get(trace_id, [])
    
    def get_trace_metrics(self) -> Dict[str, Any]:
        """Get trace metrics"""
        with self.lock:
            return {
                'active_traces': len(self.active_traces),
                'completed_traces': len(self.completed_traces),
                'total_spans': self.trace_metrics['total_spans'],
                'average_span_duration': self._calculate_average_span_duration()
            }
    
    def _calculate_average_span_duration(self) -> float:
        """Calculate average span duration"""
        total_duration = 0
        total_spans = 0
        
        for spans in self.completed_traces.values():
            for span in spans:
                if span.duration_ms is not None:
                    total_duration += span.duration_ms
                    total_spans += 1
        
        return total_duration / total_spans if total_spans > 0 else 0.0

class MetricsCollector:
    """Advanced metrics collection system"""
    
    def __init__(self):
        self.metrics: Dict[str, List[Metric]] = defaultdict(list)
        self.metric_aggregates: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.lock = threading.Lock()
        
    def record_metric(self, name: str, value: Union[int, float], 
                     metric_type: MetricType, tags: Dict[str, str] = None,
                     unit: str = None):
        """Record a metric"""
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=datetime.now(),
            tags=tags or {},
            unit=unit
        )
        
        with self.lock:
            self.metrics[name].append(metric)
            self._update_aggregates(name, metric)
    
    def _update_aggregates(self, name: str, metric: Metric):
        """Update metric aggregates"""
        if name not in self.metric_aggregates:
            self.metric_aggregates[name] = {
                'count': 0,
                'sum': 0.0,
                'min': float('inf'),
                'max': float('-inf'),
                'avg': 0.0
            }
        
        agg = self.metric_aggregates[name]
        agg['count'] += 1
        agg['sum'] += metric.value
        agg['min'] = min(agg['min'], metric.value)
        agg['max'] = max(agg['max'], metric.value)
        agg['avg'] = agg['sum'] / agg['count']
    
    def get_metric_summary(self, name: str) -> Dict[str, Any]:
        """Get metric summary"""
        with self.lock:
            return self.metric_aggregates.get(name, {})
    
    def get_all_metrics(self) -> Dict[str, List[Metric]]:
        """Get all metrics"""
        with self.lock:
            return dict(self.metrics)
    
    def get_metrics_by_time_range(self, start_time: datetime, end_time: datetime) -> Dict[str, List[Metric]]:
        """Get metrics within a time range"""
        with self.lock:
            filtered_metrics = {}
            for name, metrics in self.metrics.items():
                filtered_metrics[name] = [
                    m for m in metrics 
                    if start_time <= m.timestamp <= end_time
                ]
            return filtered_metrics

class AlertManager:
    """Advanced alert management system"""
    
    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.alert_handlers: List[Callable[[Alert], None]] = []
        self.lock = threading.Lock()
        
    def create_alert_rule(self, rule_id: str, name: str, condition: str, 
                         level: AlertLevel, message_template: str):
        """Create an alert rule"""
        self.alert_rules[rule_id] = {
            'name': name,
            'condition': condition,
            'level': level,
            'message_template': message_template,
            'created_at': datetime.now()
        }
    
    def trigger_alert(self, rule_id: str, tags: Dict[str, str] = None, 
                     custom_message: str = None) -> str:
        """Trigger an alert"""
        if rule_id not in self.alert_rules:
            return None
        
        rule = self.alert_rules[rule_id]
        alert_id = str(uuid.uuid4())
        
        message = custom_message or rule['message_template']
        if tags:
            for key, value in tags.items():
                message = message.replace(f"{{{key}}}", str(value))
        
        alert = Alert(
            alert_id=alert_id,
            name=rule['name'],
            level=rule['level'],
            message=message,
            timestamp=datetime.now(),
            tags=tags or {}
        )
        
        with self.lock:
            self.alerts[alert_id] = alert
        
        # Notify handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
        
        return alert_id
    
    def resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        with self.lock:
            if alert_id in self.alerts:
                self.alerts[alert_id].resolved = True
                self.alerts[alert_id].resolved_at = datetime.now()
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add an alert handler"""
        self.alert_handlers.append(handler)
    
    def get_active_alerts(self) -> List[Alert]:
        """Get active (unresolved) alerts"""
        with self.lock:
            return [alert for alert in self.alerts.values() if not alert.resolved]
    
    def get_alerts_by_level(self, level: AlertLevel) -> List[Alert]:
        """Get alerts by severity level"""
        with self.lock:
            return [alert for alert in self.alerts.values() if alert.level == level]

class SystemMonitor:
    """System resource monitoring"""
    
    def __init__(self):
        self.monitoring_active = False
        self.monitoring_thread = None
        self.metrics_collector = MetricsCollector()
        
    def start_monitoring(self, interval_seconds: int = 5):
        """Start system monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("System monitoring stopped")
    
    def _monitoring_loop(self, interval_seconds: int):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                self._collect_system_metrics()
                time.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    def _collect_system_metrics(self):
        """Collect system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            self.metrics_collector.record_metric(
                "system.cpu.percent", cpu_percent, MetricType.GAUGE, unit="%"
            )
            self.metrics_collector.record_metric(
                "system.cpu.count", cpu_count, MetricType.GAUGE
            )
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.metrics_collector.record_metric(
                "system.memory.percent", memory.percent, MetricType.GAUGE, unit="%"
            )
            self.metrics_collector.record_metric(
                "system.memory.used", memory.used, MetricType.GAUGE, unit="bytes"
            )
            self.metrics_collector.record_metric(
                "system.memory.available", memory.available, MetricType.GAUGE, unit="bytes"
            )
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self.metrics_collector.record_metric(
                "system.disk.percent", disk.percent, MetricType.GAUGE, unit="%"
            )
            self.metrics_collector.record_metric(
                "system.disk.used", disk.used, MetricType.GAUGE, unit="bytes"
            )
            self.metrics_collector.record_metric(
                "system.disk.free", disk.free, MetricType.GAUGE, unit="bytes"
            )
            
            # Process metrics
            process = psutil.Process()
            self.metrics_collector.record_metric(
                "process.memory.percent", process.memory_percent(), MetricType.GAUGE, unit="%"
            )
            self.metrics_collector.record_metric(
                "process.memory.rss", process.memory_info().rss, MetricType.GAUGE, unit="bytes"
            )
            self.metrics_collector.record_metric(
                "process.cpu.percent", process.cpu_percent(), MetricType.GAUGE, unit="%"
            )
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

class AdvancedMonitoring:
    """Main monitoring and tracing system"""
    
    def __init__(self):
        self.tracer = DistributedTracer()
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.system_monitor = SystemMonitor()
        
        # Initialize alert rules
        self._setup_default_alert_rules()
        
        # Start system monitoring
        self.system_monitor.start_monitoring()
        
        logger.info("Advanced monitoring system initialized")
    
    def _setup_default_alert_rules(self):
        """Setup default alert rules"""
        self.alert_manager.create_alert_rule(
            "high_cpu", "High CPU Usage", "cpu_percent > 80", 
            AlertLevel.WARNING, "CPU usage is high: {cpu_percent}%"
        )
        self.alert_manager.create_alert_rule(
            "high_memory", "High Memory Usage", "memory_percent > 85", 
            AlertLevel.WARNING, "Memory usage is high: {memory_percent}%"
        )
        self.alert_manager.create_alert_rule(
            "low_disk_space", "Low Disk Space", "disk_percent > 90", 
            AlertLevel.CRITICAL, "Disk space is critically low: {disk_percent}%"
        )
        self.alert_manager.create_alert_rule(
            "error_rate_high", "High Error Rate", "error_rate > 5", 
            AlertLevel.ERROR, "Error rate is high: {error_rate}%"
        )
    
    def start_trace(self, operation_name: str, trace_type: TraceType, 
                   parent_span_id: Optional[str] = None, tags: Dict[str, Any] = None) -> str:
        """Start a distributed trace"""
        return self.tracer.start_trace(operation_name, trace_type, parent_span_id, tags)
    
    def finish_trace(self, span_id: str, status: str = "completed", error: Optional[str] = None):
        """Finish a distributed trace"""
        self.tracer.finish_span(span_id, status, error)
    
    def add_trace_log(self, span_id: str, message: str, level: str = "info", 
                     fields: Dict[str, Any] = None):
        """Add a log to a trace"""
        self.tracer.add_span_log(span_id, message, level, fields)
    
    def record_metric(self, name: str, value: Union[int, float], 
                     metric_type: MetricType, tags: Dict[str, str] = None,
                     unit: str = None):
        """Record a metric"""
        self.metrics_collector.record_metric(name, value, metric_type, tags, unit)
    
    def trigger_alert(self, rule_id: str, tags: Dict[str, str] = None, 
                     custom_message: str = None) -> str:
        """Trigger an alert"""
        return self.alert_manager.trigger_alert(rule_id, tags, custom_message)
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard"""
        return {
            'tracing': self.tracer.get_trace_metrics(),
            'metrics': {
                name: self.metrics_collector.get_metric_summary(name)
                for name in self.metrics_collector.metrics.keys()
            },
            'alerts': {
                'active': len(self.alert_manager.get_active_alerts()),
                'critical': len(self.alert_manager.get_alerts_by_level(AlertLevel.CRITICAL)),
                'error': len(self.alert_manager.get_alerts_by_level(AlertLevel.ERROR)),
                'warning': len(self.alert_manager.get_alerts_by_level(AlertLevel.WARNING))
            },
            'system': {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent
            }
        }
    
    def get_trace(self, trace_id: str) -> Optional[List[TraceSpan]]:
        """Get a specific trace"""
        return self.tracer.get_trace(trace_id)
    
    def get_metrics_by_name(self, name: str) -> List[Metric]:
        """Get metrics by name"""
        return self.metrics_collector.metrics.get(name, [])
    
    def get_active_alerts(self) -> List[Alert]:
        """Get active alerts"""
        return self.alert_manager.get_active_alerts()
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add an alert handler"""
        self.alert_manager.add_alert_handler(handler)
    
    def shutdown(self):
        """Shutdown monitoring system"""
        self.system_monitor.stop_monitoring()
        logger.info("Advanced monitoring system shutdown")

# Global instance
advanced_monitoring = AdvancedMonitoring()




