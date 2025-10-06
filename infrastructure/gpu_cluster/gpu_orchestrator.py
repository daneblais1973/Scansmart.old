"""
GPU Cluster Orchestrator
Advanced GPU cluster management with load balancing, failover, and resource optimization
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import multiprocessing as mp
from queue import Queue, PriorityQueue
import os
import sys

# Optional psutil import for system monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

# Optional subprocess import
try:
    import subprocess
    SUBPROCESS_AVAILABLE = True
except ImportError:
    SUBPROCESS_AVAILABLE = False
    subprocess = None


class ClusterStatus(Enum):
    """GPU cluster status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RESOURCE_BASED = "resource_based"
    PERFORMANCE_BASED = "performance_based"


class FailoverStrategy(Enum):
    """Failover strategies"""
    ACTIVE_PASSIVE = "active_passive"
    ACTIVE_ACTIVE = "active_active"
    N_PLUS_1 = "n_plus_1"
    N_PLUS_M = "n_plus_m"
    GEOGRAPHIC = "geographic"


@dataclass
class GPUNode:
    """GPU node configuration"""
    node_id: str
    hostname: str
    ip_address: str
    gpu_count: int
    gpu_types: List[str]
    total_memory: int  # in GB
    available_memory: int  # in GB
    cpu_cores: int
    cpu_usage: float
    memory_usage: float
    gpu_usage: List[float]
    temperature: List[float]
    power_usage: float
    status: ClusterStatus
    last_heartbeat: datetime
    priority: int = 0
    tags: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)


@dataclass
class Workload:
    """GPU workload definition"""
    workload_id: str
    name: str
    priority: int
    resource_requirements: Dict[str, Any]
    estimated_duration: float  # in seconds
    deadline: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Allocation:
    """GPU resource allocation"""
    allocation_id: str
    workload_id: str
    node_id: str
    gpu_indices: List[int]
    memory_allocation: int  # in GB
    cpu_allocation: int
    start_time: datetime
    estimated_end_time: datetime
    status: str = "active"


class GPUClusterOrchestrator:
    """Advanced GPU cluster orchestrator"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.nodes: Dict[str, GPUNode] = {}
        self.workloads: Dict[str, Workload] = {}
        self.allocations: Dict[str, Allocation] = {}
        self.workload_queue = PriorityQueue()
        self.load_balancing_strategy = LoadBalancingStrategy.LEAST_LOADED
        self.failover_strategy = FailoverStrategy.ACTIVE_ACTIVE
        self.heartbeat_interval = 30  # seconds
        self.cleanup_interval = 300  # seconds
        self.max_retries = 3
        self.retry_delay = 5  # seconds
        
        # Performance metrics
        self.metrics = {
            'total_allocations': 0,
            'successful_allocations': 0,
            'failed_allocations': 0,
            'average_allocation_time': 0.0,
            'average_workload_duration': 0.0,
            'cluster_utilization': 0.0,
            'throughput': 0.0
        }
        
        # Initialize monitoring
        self._initialize_monitoring()
        
        logger.info("GPU Cluster Orchestrator initialized")
    
    def _initialize_monitoring(self):
        """Initialize cluster monitoring"""
        try:
            # Start heartbeat monitoring
            self.heartbeat_task = asyncio.create_task(self._monitor_heartbeats())
            
            # Start cleanup task
            self.cleanup_task = asyncio.create_task(self._cleanup_expired_allocations())
            
            # Start metrics collection
            self.metrics_task = asyncio.create_task(self._collect_metrics())
            
            logger.info("Cluster monitoring initialized")
            
        except Exception as e:
            logger.error(f"Error initializing monitoring: {e}")
    
    async def register_node(self, node: GPUNode) -> bool:
        """Register a GPU node with the cluster"""
        try:
            # Validate node configuration
            if not self._validate_node(node):
                logger.error(f"Invalid node configuration: {node.node_id}")
                return False
            
            # Add node to cluster
            self.nodes[node.node_id] = node
            
            # Update cluster status
            await self._update_cluster_status()
            
            logger.info(f"Node registered: {node.node_id} ({node.hostname})")
            return True
            
        except Exception as e:
            logger.error(f"Error registering node {node.node_id}: {e}")
            return False
    
    async def unregister_node(self, node_id: str) -> bool:
        """Unregister a GPU node from the cluster"""
        try:
            if node_id not in self.nodes:
                logger.warning(f"Node not found: {node_id}")
                return False
            
            # Migrate workloads from this node
            await self._migrate_workloads_from_node(node_id)
            
            # Remove node
            del self.nodes[node_id]
            
            # Update cluster status
            await self._update_cluster_status()
            
            logger.info(f"Node unregistered: {node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error unregistering node {node_id}: {e}")
            return False
    
    async def submit_workload(self, workload: Workload) -> str:
        """Submit a workload to the cluster"""
        try:
            # Validate workload
            if not self._validate_workload(workload):
                logger.error(f"Invalid workload: {workload.workload_id}")
                return None
            
            # Add to workload queue
            self.workloads[workload.workload_id] = workload
            self.workload_queue.put((workload.priority, workload.workload_id))
            
            # Try to allocate resources
            allocation_id = await self._allocate_resources(workload)
            
            if allocation_id:
                logger.info(f"Workload allocated: {workload.workload_id} -> {allocation_id}")
                return allocation_id
            else:
                logger.info(f"Workload queued: {workload.workload_id}")
                return workload.workload_id
            
        except Exception as e:
            logger.error(f"Error submitting workload {workload.workload_id}: {e}")
            return None
    
    async def _allocate_resources(self, workload: Workload) -> Optional[str]:
        """Allocate resources for a workload"""
        try:
            # Find suitable nodes
            suitable_nodes = await self._find_suitable_nodes(workload)
            
            if not suitable_nodes:
                logger.warning(f"No suitable nodes for workload: {workload.workload_id}")
                return None
            
            # Select best node based on load balancing strategy
            selected_node = await self._select_node(suitable_nodes, workload)
            
            if not selected_node:
                logger.warning(f"No node selected for workload: {workload.workload_id}")
                return None
            
            # Create allocation
            allocation = await self._create_allocation(workload, selected_node)
            
            if allocation:
                self.allocations[allocation.allocation_id] = allocation
                self.metrics['total_allocations'] += 1
                self.metrics['successful_allocations'] += 1
                
                return allocation.allocation_id
            
            return None
            
        except Exception as e:
            logger.error(f"Error allocating resources for {workload.workload_id}: {e}")
            self.metrics['failed_allocations'] += 1
            return None
    
    async def _find_suitable_nodes(self, workload: Workload) -> List[GPUNode]:
        """Find nodes suitable for a workload"""
        suitable_nodes = []
        
        for node in self.nodes.values():
            if node.status != ClusterStatus.HEALTHY:
                continue
            
            # Check resource requirements
            if self._check_resource_requirements(node, workload):
                suitable_nodes.append(node)
        
        return suitable_nodes
    
    def _check_resource_requirements(self, node: GPUNode, workload: Workload) -> bool:
        """Check if node meets workload requirements"""
        try:
            requirements = workload.resource_requirements
            
            # Check GPU requirements
            if 'gpu_count' in requirements:
                if node.gpu_count < requirements['gpu_count']:
                    return False
            
            # Check memory requirements
            if 'memory' in requirements:
                if node.available_memory < requirements['memory']:
                    return False
            
            # Check CPU requirements
            if 'cpu_cores' in requirements:
                if node.cpu_cores < requirements['cpu_cores']:
                    return False
            
            # Check GPU type requirements
            if 'gpu_types' in requirements:
                if not any(gpu_type in node.gpu_types for gpu_type in requirements['gpu_types']):
                    return False
            
            # Check capabilities
            if 'capabilities' in requirements:
                if not all(cap in node.capabilities for cap in requirements['capabilities']):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking resource requirements: {e}")
            return False
    
    async def _select_node(self, nodes: List[GPUNode], workload: Workload) -> Optional[GPUNode]:
        """Select the best node based on load balancing strategy"""
        try:
            if not nodes:
                return None
            
            if self.load_balancing_strategy == LoadBalancingStrategy.ROUND_ROBIN:
                return nodes[0]  # Simplified round robin
            
            elif self.load_balancing_strategy == LoadBalancingStrategy.LEAST_LOADED:
                return min(nodes, key=lambda n: n.gpu_usage[0] if n.gpu_usage else 0)
            
            elif self.load_balancing_strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                # Weight by available resources
                weights = [n.available_memory * n.gpu_count for n in nodes]
                total_weight = sum(weights)
                if total_weight == 0:
                    return nodes[0]
                
                # Select based on weight
                import random
                rand = random.uniform(0, total_weight)
                cumulative = 0
                for i, weight in enumerate(weights):
                    cumulative += weight
                    if rand <= cumulative:
                        return nodes[i]
                
                return nodes[0]
            
            elif self.load_balancing_strategy == LoadBalancingStrategy.PERFORMANCE_BASED:
                # Select based on performance metrics
                performance_scores = []
                for node in nodes:
                    score = self._calculate_performance_score(node)
                    performance_scores.append((score, node))
                
                performance_scores.sort(reverse=True)
                return performance_scores[0][1]
            
            else:
                return nodes[0]
                
        except Exception as e:
            logger.error(f"Error selecting node: {e}")
            return nodes[0] if nodes else None
    
    def _calculate_performance_score(self, node: GPUNode) -> float:
        """Calculate performance score for a node"""
        try:
            # Base score from available resources
            memory_score = node.available_memory / node.total_memory
            gpu_score = 1.0 - (sum(node.gpu_usage) / len(node.gpu_usage)) if node.gpu_usage else 1.0
            cpu_score = 1.0 - node.cpu_usage
            
            # Temperature penalty
            temp_penalty = 0.0
            if node.temperature:
                avg_temp = sum(node.temperature) / len(node.temperature)
                if avg_temp > 80:
                    temp_penalty = (avg_temp - 80) / 20  # Penalty for high temperature
            
            # Power efficiency bonus
            power_bonus = 0.0
            if node.power_usage < 200:  # Low power usage
                power_bonus = 0.1
            
            # Calculate final score
            score = (memory_score * 0.4 + gpu_score * 0.4 + cpu_score * 0.2) - temp_penalty + power_bonus
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating performance score: {e}")
            return 0.5
    
    async def _create_allocation(self, workload: Workload, node: GPUNode) -> Optional[Allocation]:
        """Create resource allocation"""
        try:
            allocation_id = f"alloc_{workload.workload_id}_{int(time.time())}"
            
            # Calculate resource allocation
            gpu_indices = list(range(min(workload.resource_requirements.get('gpu_count', 1), node.gpu_count)))
            memory_allocation = workload.resource_requirements.get('memory', 1024)  # Default 1GB
            cpu_allocation = workload.resource_requirements.get('cpu_cores', 1)
            
            # Estimate end time
            estimated_duration = workload.estimated_duration
            estimated_end_time = datetime.now() + timedelta(seconds=estimated_duration)
            
            allocation = Allocation(
                allocation_id=allocation_id,
                workload_id=workload.workload_id,
                node_id=node.node_id,
                gpu_indices=gpu_indices,
                memory_allocation=memory_allocation,
                cpu_allocation=cpu_allocation,
                start_time=datetime.now(),
                estimated_end_time=estimated_end_time
            )
            
            return allocation
            
        except Exception as e:
            logger.error(f"Error creating allocation: {e}")
            return None
    
    async def _monitor_heartbeats(self):
        """Monitor node heartbeats"""
        while True:
            try:
                current_time = datetime.now()
                offline_nodes = []
                
                for node_id, node in self.nodes.items():
                    if (current_time - node.last_heartbeat).total_seconds() > self.heartbeat_interval * 3:
                        offline_nodes.append(node_id)
                        node.status = ClusterStatus.OFFLINE
                
                # Handle offline nodes
                for node_id in offline_nodes:
                    await self._handle_node_failure(node_id)
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error monitoring heartbeats: {e}")
                await asyncio.sleep(self.heartbeat_interval)
    
    async def _handle_node_failure(self, node_id: str):
        """Handle node failure"""
        try:
            logger.warning(f"Node failure detected: {node_id}")
            
            # Migrate workloads from failed node
            await self._migrate_workloads_from_node(node_id)
            
            # Update cluster status
            await self._update_cluster_status()
            
        except Exception as e:
            logger.error(f"Error handling node failure {node_id}: {e}")
    
    async def _migrate_workloads_from_node(self, node_id: str):
        """Migrate workloads from a failed node"""
        try:
            # Find allocations on this node
            allocations_to_migrate = [
                alloc for alloc in self.allocations.values()
                if alloc.node_id == node_id and alloc.status == "active"
            ]
            
            for allocation in allocations_to_migrate:
                # Find alternative node
                workload = self.workloads.get(allocation.workload_id)
                if workload:
                    suitable_nodes = await self._find_suitable_nodes(workload)
                    if suitable_nodes:
                        # Migrate to new node
                        new_allocation = await self._create_allocation(workload, suitable_nodes[0])
                        if new_allocation:
                            # Mark old allocation as failed
                            allocation.status = "failed"
                            
                            # Add new allocation
                            self.allocations[new_allocation.allocation_id] = new_allocation
                            
                            logger.info(f"Workload migrated: {allocation.workload_id}")
            
        except Exception as e:
            logger.error(f"Error migrating workloads from node {node_id}: {e}")
    
    async def _cleanup_expired_allocations(self):
        """Clean up expired allocations"""
        while True:
            try:
                current_time = datetime.now()
                expired_allocations = []
                
                for allocation_id, allocation in self.allocations.items():
                    if allocation.estimated_end_time < current_time:
                        expired_allocations.append(allocation_id)
                
                # Remove expired allocations
                for allocation_id in expired_allocations:
                    del self.allocations[allocation_id]
                
                if expired_allocations:
                    logger.info(f"Cleaned up {len(expired_allocations)} expired allocations")
                
                await asyncio.sleep(self.cleanup_interval)
                
            except Exception as e:
                logger.error(f"Error cleaning up allocations: {e}")
                await asyncio.sleep(self.cleanup_interval)
    
    async def _collect_metrics(self):
        """Collect cluster metrics"""
        while True:
            try:
                # Calculate cluster utilization
                total_gpus = sum(node.gpu_count for node in self.nodes.values())
                used_gpus = sum(len(alloc.gpu_indices) for alloc in self.allocations.values() if alloc.status == "active")
                
                if total_gpus > 0:
                    self.metrics['cluster_utilization'] = used_gpus / total_gpus
                
                # Calculate throughput
                active_workloads = len([w for w in self.workloads.values() if w.workload_id in [a.workload_id for a in self.allocations.values() if a.status == "active"]])
                self.metrics['throughput'] = active_workloads
                
                await asyncio.sleep(60)  # Update metrics every minute
                
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(60)
    
    async def _update_cluster_status(self):
        """Update overall cluster status"""
        try:
            if not self.nodes:
                return
            
            healthy_nodes = sum(1 for node in self.nodes.values() if node.status == ClusterStatus.HEALTHY)
            total_nodes = len(self.nodes)
            
            if healthy_nodes == total_nodes:
                cluster_status = ClusterStatus.HEALTHY
            elif healthy_nodes >= total_nodes * 0.7:
                cluster_status = ClusterStatus.DEGRADED
            else:
                cluster_status = ClusterStatus.CRITICAL
            
            logger.info(f"Cluster status: {cluster_status.value} ({healthy_nodes}/{total_nodes} nodes healthy)")
            
        except Exception as e:
            logger.error(f"Error updating cluster status: {e}")
    
    def _validate_node(self, node: GPUNode) -> bool:
        """Validate node configuration"""
        try:
            if not node.node_id or not node.hostname:
                return False
            
            if node.gpu_count <= 0:
                return False
            
            if node.total_memory <= 0:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating node: {e}")
            return False
    
    def _validate_workload(self, workload: Workload) -> bool:
        """Validate workload configuration"""
        try:
            if not workload.workload_id or not workload.name:
                return False
            
            if workload.estimated_duration <= 0:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating workload: {e}")
            return False
    
    async def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status"""
        try:
            status = {
                'cluster_health': self._get_cluster_health(),
                'node_count': len(self.nodes),
                'active_workloads': len([w for w in self.workloads.values() if w.workload_id in [a.workload_id for a in self.allocations.values() if a.status == "active"]]),
                'total_allocations': len(self.allocations),
                'metrics': self.metrics.copy(),
                'nodes': {node_id: {
                    'hostname': node.hostname,
                    'status': node.status.value,
                    'gpu_count': node.gpu_count,
                    'available_memory': node.available_memory,
                    'cpu_usage': node.cpu_usage,
                    'gpu_usage': node.gpu_usage,
                    'temperature': node.temperature,
                    'last_heartbeat': node.last_heartbeat.isoformat()
                } for node_id, node in self.nodes.items()}
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting cluster status: {e}")
            return {}
    
    def _get_cluster_health(self) -> str:
        """Get overall cluster health"""
        try:
            if not self.nodes:
                return "empty"
            
            healthy_nodes = sum(1 for node in self.nodes.values() if node.status == ClusterStatus.HEALTHY)
            total_nodes = len(self.nodes)
            
            if healthy_nodes == total_nodes:
                return "healthy"
            elif healthy_nodes >= total_nodes * 0.7:
                return "degraded"
            else:
                return "critical"
                
        except Exception as e:
            logger.error(f"Error getting cluster health: {e}")
            return "unknown"
    
    async def shutdown(self):
        """Shutdown the orchestrator"""
        try:
            # Cancel monitoring tasks
            if hasattr(self, 'heartbeat_task'):
                self.heartbeat_task.cancel()
            if hasattr(self, 'cleanup_task'):
                self.cleanup_task.cancel()
            if hasattr(self, 'metrics_task'):
                self.metrics_task.cancel()
            
            logger.info("GPU Cluster Orchestrator shutdown")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Configure logging
logger = logging.getLogger(__name__)
