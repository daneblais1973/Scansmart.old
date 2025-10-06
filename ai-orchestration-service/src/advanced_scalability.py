"""
Advanced Scalability System
==========================
Enterprise-grade auto-scaling and load balancing system.
"""

import asyncio
import logging
import time
import threading
import psutil
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)

class ScalingPolicy(Enum):
    """Scaling policies"""
    CPU_BASED = "cpu_based"
    MEMORY_BASED = "memory_based"
    REQUEST_BASED = "request_based"
    CUSTOM_METRIC = "custom_metric"
    HYBRID = "hybrid"

class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    IP_HASH = "ip_hash"
    LEAST_LOAD = "least_load"

class NodeStatus(Enum):
    """Node status"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"
    MAINTENANCE = "maintenance"
    STARTING = "starting"
    STOPPING = "stopping"

class ScalingAction(Enum):
    """Scaling actions"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_ACTION = "no_action"
    EMERGENCY_SCALE_UP = "emergency_scale_up"

@dataclass
class Node:
    """Represents a compute node"""
    node_id: str
    host: str
    port: int
    status: NodeStatus = NodeStatus.HEALTHY
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    active_connections: int = 0
    response_time: float = 0.0
    last_health_check: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)
    weight: int = 1
    tags: Dict[str, str] = field(default_factory=dict)
    
    def is_healthy(self) -> bool:
        """Check if node is healthy"""
        return (self.status == NodeStatus.HEALTHY and 
                self.cpu_usage < 90.0 and 
                self.memory_usage < 90.0 and
                self.response_time < 1000.0)  # 1 second

@dataclass
class ScalingMetrics:
    """Scaling metrics"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    request_rate: float
    response_time: float
    error_rate: float
    active_connections: int
    queue_length: int

@dataclass
class ScalingRule:
    """Scaling rule configuration"""
    rule_id: str
    name: str
    metric: str
    threshold: float
    comparison: str  # "gt", "lt", "eq"
    action: ScalingAction
    cooldown_period: int = 300  # seconds
    min_nodes: int = 1
    max_nodes: int = 10
    scale_factor: float = 1.0
    enabled: bool = True

class LoadBalancer:
    """Advanced load balancer"""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN):
        self.strategy = strategy
        self.nodes: Dict[str, Node] = {}
        self.current_index = 0
        self.health_check_interval = 30  # seconds
        self.health_check_timeout = 5  # seconds
        self.lock = threading.Lock()
        
        # Start health checking
        self.health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self.health_check_thread.start()
    
    def add_node(self, node: Node) -> bool:
        """Add a node to the load balancer"""
        try:
            with self.lock:
                self.nodes[node.node_id] = node
                logger.info(f"Added node {node.node_id} to load balancer")
                return True
        except Exception as e:
            logger.error(f"Error adding node: {e}")
            return False
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node from the load balancer"""
        try:
            with self.lock:
                if node_id in self.nodes:
                    del self.nodes[node_id]
                    logger.info(f"Removed node {node_id} from load balancer")
                    return True
                return False
        except Exception as e:
            logger.error(f"Error removing node: {e}")
            return False
    
    def get_next_node(self, client_ip: str = None) -> Optional[Node]:
        """Get the next node based on load balancing strategy"""
        try:
            with self.lock:
                healthy_nodes = [node for node in self.nodes.values() if node.is_healthy()]
                
                if not healthy_nodes:
                    return None
                
                if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                    node = healthy_nodes[self.current_index % len(healthy_nodes)]
                    self.current_index += 1
                    return node
                
                elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                    return min(healthy_nodes, key=lambda n: n.active_connections)
                
                elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
                    return min(healthy_nodes, key=lambda n: n.response_time)
                
                elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                    # Weighted round robin based on node weight
                    total_weight = sum(node.weight for node in healthy_nodes)
                    if total_weight == 0:
                        return healthy_nodes[0]
                    
                    # Simple weighted selection
                    target_weight = self.current_index % total_weight
                    current_weight = 0
                    for node in healthy_nodes:
                        current_weight += node.weight
                        if current_weight > target_weight:
                            self.current_index += 1
                            return node
                    return healthy_nodes[0]
                
                elif self.strategy == LoadBalancingStrategy.IP_HASH:
                    if client_ip:
                        hash_value = hash(client_ip) % len(healthy_nodes)
                        return healthy_nodes[hash_value]
                    else:
                        return healthy_nodes[0]
                
                elif self.strategy == LoadBalancingStrategy.LEAST_LOAD:
                    # Combine CPU, memory, and connections for load calculation
                    def calculate_load(node):
                        return (node.cpu_usage * 0.4 + 
                                node.memory_usage * 0.4 + 
                                (node.active_connections / 100) * 0.2)
                    
                    return min(healthy_nodes, key=calculate_load)
                
                return healthy_nodes[0]
        except Exception as e:
            logger.error(f"Error getting next node: {e}")
            return None
    
    def update_node_metrics(self, node_id: str, metrics: Dict[str, Any]):
        """Update node metrics"""
        try:
            with self.lock:
                if node_id in self.nodes:
                    node = self.nodes[node_id]
                    node.cpu_usage = metrics.get('cpu_usage', node.cpu_usage)
                    node.memory_usage = metrics.get('memory_usage', node.memory_usage)
                    node.active_connections = metrics.get('active_connections', node.active_connections)
                    node.response_time = metrics.get('response_time', node.response_time)
                    node.last_health_check = datetime.now()
        except Exception as e:
            logger.error(f"Error updating node metrics: {e}")
    
    def _health_check_loop(self):
        """Health check loop for nodes"""
        while True:
            try:
                time.sleep(self.health_check_interval)
                
                with self.lock:
                    for node in self.nodes.values():
                        # Simulate health check
                        is_healthy = (node.cpu_usage < 90.0 and 
                                    node.memory_usage < 90.0 and 
                                    node.response_time < 1000.0)
                        
                        if is_healthy:
                            node.status = NodeStatus.HEALTHY
                        else:
                            node.status = NodeStatus.UNHEALTHY
                            logger.warning(f"Node {node.node_id} marked as unhealthy")
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
    
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        try:
            with self.lock:
                healthy_nodes = [node for node in self.nodes.values() if node.is_healthy()]
                total_nodes = len(self.nodes)
                
                if healthy_nodes:
                    avg_cpu = statistics.mean(node.cpu_usage for node in healthy_nodes)
                    avg_memory = statistics.mean(node.memory_usage for node in healthy_nodes)
                    avg_response_time = statistics.mean(node.response_time for node in healthy_nodes)
                    total_connections = sum(node.active_connections for node in healthy_nodes)
                else:
                    avg_cpu = avg_memory = avg_response_time = total_connections = 0
                
                return {
                    'total_nodes': total_nodes,
                    'healthy_nodes': len(healthy_nodes),
                    'unhealthy_nodes': total_nodes - len(healthy_nodes),
                    'strategy': self.strategy.value,
                    'average_cpu': avg_cpu,
                    'average_memory': avg_memory,
                    'average_response_time': avg_response_time,
                    'total_connections': total_connections
                }
        except Exception as e:
            logger.error(f"Error getting load balancer stats: {e}")
            return {}

class AutoScaler:
    """Advanced auto-scaler"""
    
    def __init__(self):
        self.scaling_rules: Dict[str, ScalingRule] = {}
        self.metrics_history: deque = deque(maxlen=1000)
        self.scaling_actions: deque = deque(maxlen=100)
        self.current_nodes = 1
        self.min_nodes = 1
        self.max_nodes = 10
        self.scaling_cooldown = 300  # seconds
        self.last_scaling_action = None
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Default scaling rules
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default scaling rules"""
        default_rules = [
            ScalingRule(
                rule_id="cpu_scale_up",
                name="CPU Scale Up",
                metric="cpu_usage",
                threshold=80.0,
                comparison="gt",
                action=ScalingAction.SCALE_UP,
                cooldown_period=300
            ),
            ScalingRule(
                rule_id="cpu_scale_down",
                name="CPU Scale Down",
                metric="cpu_usage",
                threshold=30.0,
                comparison="lt",
                action=ScalingAction.SCALE_DOWN,
                cooldown_period=600
            ),
            ScalingRule(
                rule_id="memory_scale_up",
                name="Memory Scale Up",
                metric="memory_usage",
                threshold=85.0,
                comparison="gt",
                action=ScalingAction.SCALE_UP,
                cooldown_period=300
            ),
            ScalingRule(
                rule_id="response_time_scale_up",
                name="Response Time Scale Up",
                metric="response_time",
                threshold=500.0,
                comparison="gt",
                action=ScalingAction.SCALE_UP,
                cooldown_period=180
            )
        ]
        
        for rule in default_rules:
            self.scaling_rules[rule.rule_id] = rule
    
    def start_monitoring(self):
        """Start auto-scaling monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Auto-scaling monitoring started")
    
    def stop_monitoring(self):
        """Stop auto-scaling monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Auto-scaling monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect current metrics
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)
                
                # Evaluate scaling rules
                scaling_action = self._evaluate_scaling_rules(metrics)
                
                if scaling_action and scaling_action != ScalingAction.NO_ACTION:
                    self._execute_scaling_action(scaling_action)
                
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _collect_system_metrics(self) -> ScalingMetrics:
        """Collect system metrics"""
        try:
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Simulate other metrics
            request_rate = 100.0 + (cpu_usage * 2)  # Simulate request rate based on CPU
            response_time = 50.0 + (cpu_usage * 5)   # Simulate response time based on CPU
            error_rate = max(0, (cpu_usage - 80) * 0.5)  # Higher error rate under load
            active_connections = int(cpu_usage * 10)  # Simulate connections based on CPU
            queue_length = max(0, int((cpu_usage - 70) * 5))  # Queue builds up under load
            
            return ScalingMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                request_rate=request_rate,
                response_time=response_time,
                error_rate=error_rate,
                active_connections=active_connections,
                queue_length=queue_length
            )
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return ScalingMetrics(
                timestamp=datetime.now(),
                cpu_usage=0.0,
                memory_usage=0.0,
                request_rate=0.0,
                response_time=0.0,
                error_rate=0.0,
                active_connections=0,
                queue_length=0
            )
    
    def _evaluate_scaling_rules(self, metrics: ScalingMetrics) -> Optional[ScalingAction]:
        """Evaluate scaling rules against current metrics"""
        try:
            # Check cooldown period
            if (self.last_scaling_action and 
                datetime.now() - self.last_scaling_action < timedelta(seconds=self.scaling_cooldown)):
                return ScalingAction.NO_ACTION
            
            for rule in self.scaling_rules.values():
                if not rule.enabled:
                    continue
                
                # Get metric value
                metric_value = getattr(metrics, rule.metric, 0.0)
                
                # Check threshold
                should_trigger = False
                if rule.comparison == "gt" and metric_value > rule.threshold:
                    should_trigger = True
                elif rule.comparison == "lt" and metric_value < rule.threshold:
                    should_trigger = True
                elif rule.comparison == "eq" and metric_value == rule.threshold:
                    should_trigger = True
                
                if should_trigger:
                    # Check if scaling is within limits
                    if (rule.action == ScalingAction.SCALE_UP and 
                        self.current_nodes < rule.max_nodes):
                        logger.info(f"Scaling rule triggered: {rule.name}")
                        return rule.action
                    elif (rule.action == ScalingAction.SCALE_DOWN and 
                          self.current_nodes > rule.min_nodes):
                        logger.info(f"Scaling rule triggered: {rule.name}")
                        return rule.action
            
            return ScalingAction.NO_ACTION
        except Exception as e:
            logger.error(f"Error evaluating scaling rules: {e}")
            return ScalingAction.NO_ACTION
    
    def _execute_scaling_action(self, action: ScalingAction):
        """Execute scaling action"""
        try:
            if action == ScalingAction.SCALE_UP:
                self.current_nodes = min(self.current_nodes + 1, self.max_nodes)
                logger.info(f"Scaled up to {self.current_nodes} nodes")
            elif action == ScalingAction.SCALE_DOWN:
                self.current_nodes = max(self.current_nodes - 1, self.min_nodes)
                logger.info(f"Scaled down to {self.current_nodes} nodes")
            elif action == ScalingAction.EMERGENCY_SCALE_UP:
                self.current_nodes = min(self.current_nodes + 2, self.max_nodes)
                logger.warning(f"Emergency scale up to {self.current_nodes} nodes")
            
            # Record scaling action
            self.scaling_actions.append({
                'timestamp': datetime.now(),
                'action': action.value,
                'nodes': self.current_nodes
            })
            
            self.last_scaling_action = datetime.now()
        except Exception as e:
            logger.error(f"Error executing scaling action: {e}")
    
    def add_scaling_rule(self, rule: ScalingRule):
        """Add a scaling rule"""
        self.scaling_rules[rule.rule_id] = rule
        logger.info(f"Added scaling rule: {rule.name}")
    
    def remove_scaling_rule(self, rule_id: str):
        """Remove a scaling rule"""
        if rule_id in self.scaling_rules:
            del self.scaling_rules[rule_id]
            logger.info(f"Removed scaling rule: {rule_id}")
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get auto-scaling statistics"""
        try:
            recent_actions = list(self.scaling_actions)[-10:]  # Last 10 actions
            
            return {
                'current_nodes': self.current_nodes,
                'min_nodes': self.min_nodes,
                'max_nodes': self.max_nodes,
                'scaling_rules': len(self.scaling_rules),
                'enabled_rules': len([r for r in self.scaling_rules.values() if r.enabled]),
                'recent_actions': recent_actions,
                'monitoring_active': self.monitoring_active,
                'last_scaling_action': self.last_scaling_action.isoformat() if self.last_scaling_action else None
            }
        except Exception as e:
            logger.error(f"Error getting scaling stats: {e}")
            return {}

class AdvancedScalabilitySystem:
    """Main scalability system"""
    
    def __init__(self):
        self.load_balancer = LoadBalancer()
        self.auto_scaler = AutoScaler()
        self.nodes: Dict[str, Node] = {}
        self.metrics_collector = None
        self.alert_handlers: List[Callable[[str, Dict[str, Any]], None]] = []
        
        # Start auto-scaling
        self.auto_scaler.start_monitoring()
        
        logger.info("Advanced Scalability System initialized")
    
    def add_node(self, host: str, port: int, weight: int = 1, 
                tags: Dict[str, str] = None) -> str:
        """Add a new node to the system"""
        try:
            node_id = str(uuid.uuid4())
            node = Node(
                node_id=node_id,
                host=host,
                port=port,
                weight=weight,
                tags=tags or {}
            )
            
            self.nodes[node_id] = node
            self.load_balancer.add_node(node)
            
            logger.info(f"Added node {node_id} ({host}:{port})")
            return node_id
        except Exception as e:
            logger.error(f"Error adding node: {e}")
            return None
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node from the system"""
        try:
            if node_id in self.nodes:
                del self.nodes[node_id]
                self.load_balancer.remove_node(node_id)
                logger.info(f"Removed node {node_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error removing node: {e}")
            return False
    
    def get_next_node(self, client_ip: str = None) -> Optional[Node]:
        """Get the next available node"""
        return self.load_balancer.get_next_node(client_ip)
    
    def update_node_metrics(self, node_id: str, metrics: Dict[str, Any]):
        """Update node metrics"""
        self.load_balancer.update_node_metrics(node_id, metrics)
    
    def get_scalability_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive scalability dashboard"""
        try:
            load_balancer_stats = self.load_balancer.get_load_balancer_stats()
            scaling_stats = self.auto_scaler.get_scaling_stats()
            
            return {
                'load_balancer': load_balancer_stats,
                'auto_scaler': scaling_stats,
                'nodes': {
                    'total': len(self.nodes),
                    'healthy': len([n for n in self.nodes.values() if n.is_healthy()]),
                    'unhealthy': len([n for n in self.nodes.values() if not n.is_healthy()])
                },
                'system_metrics': self._get_system_metrics()
            }
        except Exception as e:
            logger.error(f"Error getting scalability dashboard: {e}")
            return {'error': str(e)}
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        try:
            cpu_usage = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            return {
                'cpu_usage': cpu_usage,
                'memory_usage': memory.percent,
                'memory_available': memory.available,
                'disk_usage': psutil.disk_usage('/').percent,
                'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
            }
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {}
    
    def add_scaling_rule(self, rule: ScalingRule):
        """Add a custom scaling rule"""
        self.auto_scaler.add_scaling_rule(rule)
    
    def remove_scaling_rule(self, rule_id: str):
        """Remove a scaling rule"""
        self.auto_scaler.remove_scaling_rule(rule_id)
    
    def shutdown(self):
        """Shutdown the scalability system"""
        self.auto_scaler.stop_monitoring()
        logger.info("Advanced Scalability System shutdown")

# Global instance
advanced_scalability = AdvancedScalabilitySystem()




