"""
System Monitoring Service
========================
Comprehensive monitoring system for all ScanSmart services
"""

import asyncio
import aiohttp
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import structlog

logger = structlog.get_logger()

class ServiceStatus(Enum):
    """Service status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class ServiceMetrics:
    """Service metrics data"""
    service_name: str
    status: ServiceStatus
    response_time: float
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    uptime: float
    last_check: datetime
    error_count: int = 0
    success_count: int = 0
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemAlert:
    """System alert data"""
    alert_id: str
    service_name: str
    alert_type: str
    severity: str
    message: str
    timestamp: datetime
    resolved: bool = False

class SystemMonitor:
    """Main system monitoring class"""
    
    def __init__(self):
        self.services = {
            "ai-orchestration": {"host": "localhost", "port": 8001, "path": "/health"},
            "data-service": {"host": "localhost", "port": 8002, "path": "/health"},
            "catalyst-service": {"host": "localhost", "port": 8003, "path": "/health"},
            "screening-service": {"host": "localhost", "port": 8004, "path": "/health"}
        }
        
        self.metrics: Dict[str, ServiceMetrics] = {}
        self.alerts: List[SystemAlert] = []
        self.monitoring_active = False
        
        # Prometheus metrics
        self.service_health_gauge = Gauge('service_health', 'Service health status', ['service'])
        self.service_response_time = Histogram('service_response_time', 'Service response time', ['service'])
        self.system_cpu_usage = Gauge('system_cpu_usage', 'System CPU usage percentage')
        self.system_memory_usage = Gauge('system_memory_usage', 'System memory usage percentage')
        self.system_disk_usage = Gauge('system_disk_usage', 'System disk usage percentage')
        self.alert_counter = Counter('system_alerts_total', 'Total system alerts', ['service', 'type', 'severity'])
        
        # Start Prometheus metrics server
        start_http_server(9090)
        
        logger.info("System Monitor initialized")
    
    async def start_monitoring(self):
        """Start the monitoring system"""
        self.monitoring_active = True
        logger.info("Starting system monitoring")
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._monitor_services()),
            asyncio.create_task(self._monitor_system_resources()),
            asyncio.create_task(self._check_alerts()),
            asyncio.create_task(self._cleanup_old_alerts())
        ]
        
        await asyncio.gather(*tasks)
    
    async def stop_monitoring(self):
        """Stop the monitoring system"""
        self.monitoring_active = False
        logger.info("Stopping system monitoring")
    
    async def _monitor_services(self):
        """Monitor all services"""
        while self.monitoring_active:
            try:
                for service_name, config in self.services.items():
                    await self._check_service_health(service_name, config)
                    await asyncio.sleep(1)  # Stagger checks
            except Exception as e:
                logger.error(f"Error monitoring services: {e}")
            
            await asyncio.sleep(30)  # Check every 30 seconds
    
    async def _check_service_health(self, service_name: str, config: Dict[str, Any]):
        """Check health of a specific service"""
        try:
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                url = f"http://{config['host']}:{config['port']}{config['path']}"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        data = await response.json()
                        status = ServiceStatus.HEALTHY
                        success_count = 1
                        error_count = 0
                    else:
                        status = ServiceStatus.UNHEALTHY
                        success_count = 0
                        error_count = 1
                        data = {"error": f"HTTP {response.status}"}
            
            # Get system metrics
            cpu_usage = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Update metrics
            if service_name in self.metrics:
                self.metrics[service_name].success_count += success_count
                self.metrics[service_name].error_count += error_count
            else:
                self.metrics[service_name] = ServiceMetrics(
                    service_name=service_name,
                    status=status,
                    response_time=response_time,
                    cpu_usage=cpu_usage,
                    memory_usage=memory.percent,
                    disk_usage=disk.percent,
                    uptime=0,
                    last_check=datetime.now(),
                    success_count=success_count,
                    error_count=error_count,
                    details=data
                )
            
            # Update Prometheus metrics
            self.service_health_gauge.labels(service=service_name).set(1 if status == ServiceStatus.HEALTHY else 0)
            self.service_response_time.labels(service=service_name).observe(response_time)
            
            # Check for alerts
            await self._check_service_alerts(service_name, status, response_time, cpu_usage, memory.percent)
            
        except Exception as e:
            logger.error(f"Error checking service {service_name}: {e}")
            
            # Update error metrics
            if service_name in self.metrics:
                self.metrics[service_name].error_count += 1
                self.metrics[service_name].status = ServiceStatus.UNKNOWN
            else:
                self.metrics[service_name] = ServiceMetrics(
                    service_name=service_name,
                    status=ServiceStatus.UNKNOWN,
                    response_time=0,
                    cpu_usage=0,
                    memory_usage=0,
                    disk_usage=0,
                    uptime=0,
                    last_check=datetime.now(),
                    error_count=1,
                    success_count=0,
                    details={"error": str(e)}
                )
            
            # Create alert for service failure
            await self._create_alert(
                service_name=service_name,
                alert_type="service_unreachable",
                severity="critical",
                message=f"Service {service_name} is unreachable: {str(e)}"
            )
    
    async def _monitor_system_resources(self):
        """Monitor system resources"""
        while self.monitoring_active:
            try:
                # Get system metrics
                cpu_usage = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # Update Prometheus metrics
                self.system_cpu_usage.set(cpu_usage)
                self.system_memory_usage.set(memory.percent)
                self.system_disk_usage.set(disk.percent)
                
                # Check for resource alerts
                await self._check_resource_alerts(cpu_usage, memory.percent, disk.percent)
                
            except Exception as e:
                logger.error(f"Error monitoring system resources: {e}")
            
            await asyncio.sleep(60)  # Check every minute
    
    async def _check_service_alerts(self, service_name: str, status: ServiceStatus, 
                                  response_time: float, cpu_usage: float, memory_usage: float):
        """Check for service-specific alerts"""
        # Response time alert
        if response_time > 5.0:  # 5 seconds
            await self._create_alert(
                service_name=service_name,
                alert_type="high_response_time",
                severity="warning",
                message=f"Service {service_name} has high response time: {response_time:.2f}s"
            )
        
        # Service down alert
        if status == ServiceStatus.UNHEALTHY:
            await self._create_alert(
                service_name=service_name,
                alert_type="service_down",
                severity="critical",
                message=f"Service {service_name} is down"
            )
        
        # High error rate alert
        if service_name in self.metrics:
            total_requests = self.metrics[service_name].success_count + self.metrics[service_name].error_count
            if total_requests > 10:  # Only check after some requests
                error_rate = self.metrics[service_name].error_count / total_requests
                if error_rate > 0.1:  # 10% error rate
                    await self._create_alert(
                        service_name=service_name,
                        alert_type="high_error_rate",
                        severity="warning",
                        message=f"Service {service_name} has high error rate: {error_rate:.2%}"
                    )
    
    async def _check_resource_alerts(self, cpu_usage: float, memory_usage: float, disk_usage: float):
        """Check for resource alerts"""
        # CPU usage alert
        if cpu_usage > 80:
            await self._create_alert(
                service_name="system",
                alert_type="high_cpu_usage",
                severity="warning",
                message=f"High CPU usage: {cpu_usage:.1f}%"
            )
        
        # Memory usage alert
        if memory_usage > 85:
            await self._create_alert(
                service_name="system",
                alert_type="high_memory_usage",
                severity="warning",
                message=f"High memory usage: {memory_usage:.1f}%"
            )
        
        # Disk usage alert
        if disk_usage > 90:
            await self._create_alert(
                service_name="system",
                alert_type="high_disk_usage",
                severity="critical",
                message=f"High disk usage: {disk_usage:.1f}%"
            )
    
    async def _create_alert(self, service_name: str, alert_type: str, severity: str, message: str):
        """Create a new alert"""
        alert_id = f"{service_name}_{alert_type}_{int(time.time())}"
        
        # Check if similar alert already exists
        existing_alert = next(
            (alert for alert in self.alerts 
             if alert.service_name == service_name 
             and alert.alert_type == alert_type 
             and not alert.resolved), 
            None
        )
        
        if existing_alert:
            return  # Don't create duplicate alerts
        
        alert = SystemAlert(
            alert_id=alert_id,
            service_name=service_name,
            alert_type=alert_type,
            severity=severity,
            message=message,
            timestamp=datetime.now()
        )
        
        self.alerts.append(alert)
        
        # Update Prometheus counter
        self.alert_counter.labels(service=service_name, type=alert_type, severity=severity).inc()
        
        logger.warning(f"Alert created: {message}")
    
    async def _check_alerts(self):
        """Check and process alerts"""
        while self.monitoring_active:
            try:
                # Process alerts (could send notifications, etc.)
                for alert in self.alerts:
                    if not alert.resolved:
                        # Here you could send notifications, update dashboards, etc.
                        pass
                
            except Exception as e:
                logger.error(f"Error checking alerts: {e}")
            
            await asyncio.sleep(60)  # Check every minute
    
    async def _cleanup_old_alerts(self):
        """Clean up old resolved alerts"""
        while self.monitoring_active:
            try:
                # Remove alerts older than 24 hours
                cutoff_time = datetime.now() - timedelta(hours=24)
                self.alerts = [
                    alert for alert in self.alerts 
                    if not alert.resolved or alert.timestamp > cutoff_time
                ]
                
            except Exception as e:
                logger.error(f"Error cleaning up alerts: {e}")
            
            await asyncio.sleep(3600)  # Clean up every hour
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        healthy_services = sum(1 for metrics in self.metrics.values() 
                             if metrics.status == ServiceStatus.HEALTHY)
        total_services = len(self.metrics)
        
        active_alerts = len([alert for alert in self.alerts if not alert.resolved])
        critical_alerts = len([alert for alert in self.alerts 
                             if not alert.resolved and alert.severity == "critical"])
        
        return {
            "overall_status": "healthy" if healthy_services == total_services and active_alerts == 0 else "degraded",
            "healthy_services": healthy_services,
            "total_services": total_services,
            "active_alerts": active_alerts,
            "critical_alerts": critical_alerts,
            "services": {name: metrics.__dict__ for name, metrics in self.metrics.items()},
            "alerts": [alert.__dict__ for alert in self.alerts[-10:]]  # Last 10 alerts
        }
    
    def get_service_metrics(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Get metrics for a specific service"""
        if service_name in self.metrics:
            return self.metrics[service_name].__dict__
        return None
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                logger.info(f"Alert {alert_id} resolved")
                return True
        return False

# Global monitor instance
system_monitor = SystemMonitor()

