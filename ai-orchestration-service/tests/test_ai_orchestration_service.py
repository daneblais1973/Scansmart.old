"""
AI Orchestration Service Tests
==============================
Comprehensive test suite for the AI Orchestration Service
"""

import asyncio
import pytest
import sys
import os
from datetime import datetime
from typing import Dict, List, Any

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import service components
from quantum_orchestrator import quantum_orchestrator, OrchestrationStatus, TaskPriority, QuantumAlgorithm
from meta_learning_hub import meta_learning_hub, MetaLearningType, TaskDistribution
from model_ensemble import model_ensemble, EnsembleType, ModelType, EnsembleStatus
from continual_learner import continual_learner, ContinualLearningType, LearningStatus
from performance_optimizer import performance_optimizer, OptimizationType, OptimizationTarget, OptimizationStatus

class TestQuantumOrchestrator:
    """Test Quantum Orchestrator functionality"""
    
    @pytest.mark.asyncio
    async def test_quantum_orchestrator_initialization(self):
        """Test quantum orchestrator initialization"""
        assert quantum_orchestrator.status == OrchestrationStatus.IDLE
        assert len(quantum_orchestrator.tasks) == 0
        assert len(quantum_orchestrator.active_tasks) == 0
        assert len(quantum_orchestrator.completed_tasks) == 0
    
    @pytest.mark.asyncio
    async def test_start_stop_orchestration(self):
        """Test starting and stopping orchestration service"""
        # Start orchestration
        await quantum_orchestrator.start_orchestration()
        assert quantum_orchestrator.status == OrchestrationStatus.RUNNING
        
        # Stop orchestration
        await quantum_orchestrator.stop_orchestration()
        assert quantum_orchestrator.status == OrchestrationStatus.IDLE
    
    @pytest.mark.asyncio
    async def test_submit_task(self):
        """Test task submission"""
        task_id = await quantum_orchestrator.submit_task(
            task_type="quantum_optimization",
            parameters={"algorithm": "qaoa", "problem_data": {}},
            priority=TaskPriority.HIGH
        )
        
        assert task_id is not None
        assert task_id in quantum_orchestrator.tasks
        assert quantum_orchestrator.metrics.total_tasks == 1
    
    @pytest.mark.asyncio
    async def test_get_task_status(self):
        """Test getting task status"""
        task_id = await quantum_orchestrator.submit_task(
            task_type="test_task",
            parameters={},
            priority=TaskPriority.MEDIUM
        )
        
        status = await quantum_orchestrator.get_task_status(task_id)
        assert status is not None
        assert status['task_id'] == task_id
        assert status['status'] == "queued"
    
    @pytest.mark.asyncio
    async def test_get_orchestration_status(self):
        """Test getting orchestration status"""
        status = quantum_orchestrator.get_orchestration_status()
        
        assert 'status' in status
        assert 'total_tasks' in status
        assert 'active_tasks' in status
        assert 'completed_tasks' in status
        assert 'failed_tasks' in status
        assert 'metrics' in status
        assert 'timestamp' in status

class TestMetaLearningHub:
    """Test Meta-Learning Hub functionality"""
    
    @pytest.mark.asyncio
    async def test_meta_learning_hub_initialization(self):
        """Test meta-learning hub initialization"""
        assert meta_learning_hub.status == "initialized"
        assert len(meta_learning_hub.meta_tasks) == 0
        assert len(meta_learning_hub.meta_results) == 0
        assert len(meta_learning_hub.algorithms) == 6
    
    @pytest.mark.asyncio
    async def test_start_stop_meta_learning_hub(self):
        """Test starting and stopping meta-learning hub"""
        # Start hub
        await meta_learning_hub.start_meta_learning_hub()
        assert meta_learning_hub.status == "running"
        
        # Stop hub
        await meta_learning_hub.stop_meta_learning_hub()
        assert meta_learning_hub.status == "stopped"
    
    @pytest.mark.asyncio
    async def test_create_meta_task(self):
        """Test creating meta-learning task"""
        support_set = [{"features": [1, 2, 3], "label": 0}]
        query_set = [{"features": [4, 5, 6], "label": 1}]
        
        task_id = await meta_learning_hub.create_meta_task(
            task_type="classification",
            support_set=support_set,
            query_set=query_set,
            num_classes=2,
            num_shots=1,
            difficulty=0.5
        )
        
        assert task_id is not None
        assert task_id in meta_learning_hub.meta_tasks
        assert meta_learning_hub.metrics.total_tasks == 1
    
    @pytest.mark.asyncio
    async def test_execute_meta_learning(self):
        """Test executing meta-learning"""
        # Create task first
        support_set = [{"features": [1, 2, 3], "label": 0}]
        query_set = [{"features": [4, 5, 6], "label": 1}]
        
        task_id = await meta_learning_hub.create_meta_task(
            task_type="classification",
            support_set=support_set,
            query_set=query_set,
            num_classes=2,
            num_shots=1,
            difficulty=0.5
        )
        
        # Execute meta-learning
        result = await meta_learning_hub.execute_meta_learning(task_id, MetaLearningType.MAML)
        
        assert result is not None
        assert result.task_id == task_id
        assert result.algorithm == "maml"
        assert 0.0 <= result.accuracy <= 1.0
        assert result.convergence_time > 0.0
    
    @pytest.mark.asyncio
    async def test_get_meta_learning_status(self):
        """Test getting meta-learning status"""
        status = await meta_learning_hub.get_meta_learning_status()
        
        assert 'status' in status
        assert 'total_tasks' in status
        assert 'successful_tasks' in status
        assert 'failed_tasks' in status
        assert 'average_accuracy' in status
        assert 'available_algorithms' in status
        assert 'timestamp' in status

class TestModelEnsemble:
    """Test Model Ensemble functionality"""
    
    @pytest.mark.asyncio
    async def test_model_ensemble_initialization(self):
        """Test model ensemble initialization"""
        assert model_ensemble.status == EnsembleStatus.READY
        assert len(model_ensemble.models) > 0
        assert model_ensemble.ensemble_type == EnsembleType.WEIGHTED_AVERAGING
        assert len(model_ensemble.predictions) == 0
    
    @pytest.mark.asyncio
    async def test_start_stop_ensemble(self):
        """Test starting and stopping ensemble"""
        # Start ensemble
        await model_ensemble.start_ensemble()
        assert model_ensemble.status == EnsembleStatus.READY
        
        # Stop ensemble
        await model_ensemble.stop_ensemble()
        assert model_ensemble.status == EnsembleStatus.INITIALIZING
    
    @pytest.mark.asyncio
    async def test_ensemble_prediction(self):
        """Test ensemble prediction"""
        input_data = {"features": [1, 2, 3, 4, 5]}
        
        prediction = await model_ensemble.predict(input_data)
        
        assert prediction is not None
        assert prediction.ensemble_id == model_ensemble.ensemble_id
        assert prediction.confidence >= 0.0
        assert prediction.uncertainty >= 0.0
        assert prediction.prediction_time > 0.0
        assert len(prediction.individual_predictions) > 0
    
    @pytest.mark.asyncio
    async def test_add_remove_model(self):
        """Test adding and removing models"""
        initial_count = len(model_ensemble.models)
        
        # Add model
        model_id = await model_ensemble.add_model(
            model_type=ModelType.NEURAL_NETWORK,
            name="test_model",
            weight=0.1
        )
        
        assert model_id is not None
        assert len(model_ensemble.models) == initial_count + 1
        
        # Remove model
        success = await model_ensemble.remove_model(model_id)
        assert success
        assert len(model_ensemble.models) == initial_count
    
    @pytest.mark.asyncio
    async def test_get_ensemble_status(self):
        """Test getting ensemble status"""
        status = await model_ensemble.get_ensemble_status()
        
        assert 'ensemble_id' in status
        assert 'status' in status
        assert 'ensemble_type' in status
        assert 'num_models' in status
        assert 'models' in status
        assert 'metrics' in status
        assert 'config' in status
        assert 'timestamp' in status

class TestContinualLearner:
    """Test Continual Learner functionality"""
    
    @pytest.mark.asyncio
    async def test_continual_learner_initialization(self):
        """Test continual learner initialization"""
        assert continual_learner.status == LearningStatus.IDLE
        assert len(continual_learner.learning_tasks) == 0
        assert len(continual_learner.learning_results) == 0
        assert len(continual_learner.algorithms) == 6
    
    @pytest.mark.asyncio
    async def test_start_stop_continual_learning(self):
        """Test starting and stopping continual learning"""
        # Start continual learning
        await continual_learner.start_continual_learning()
        assert continual_learner.status == LearningStatus.IDLE
        
        # Stop continual learning
        await continual_learner.stop_continual_learning()
        assert continual_learner.status == LearningStatus.IDLE
    
    @pytest.mark.asyncio
    async def test_add_learning_task(self):
        """Test adding learning task"""
        data = [{"features": [1, 2, 3], "label": 0}]
        labels = [0]
        
        task_id = await continual_learner.add_learning_task(
            task_name="test_task",
            task_type="classification",
            data=data,
            labels=labels
        )
        
        assert task_id is not None
        assert task_id in continual_learner.learning_tasks
        assert continual_learner.metrics.total_tasks == 1
    
    @pytest.mark.asyncio
    async def test_learn_task(self):
        """Test learning a task"""
        # Add task first
        data = [{"features": [1, 2, 3], "label": 0}]
        labels = [0]
        
        task_id = await continual_learner.add_learning_task(
            task_name="test_task",
            task_type="classification",
            data=data,
            labels=labels
        )
        
        # Learn task
        result = await continual_learner.learn_task(task_id, ContinualLearningType.EWC)
        
        assert result is not None
        assert result.task_id == task_id
        assert result.algorithm == "ewc"
        assert 0.0 <= result.accuracy <= 1.0
        assert result.adaptation_time > 0.0
        assert 0.0 <= result.forgetting_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_get_continual_learning_status(self):
        """Test getting continual learning status"""
        status = await continual_learner.get_continual_learning_status()
        
        assert 'status' in status
        assert 'total_tasks' in status
        assert 'completed_tasks' in status
        assert 'average_accuracy' in status
        assert 'average_forgetting_score' in status
        assert 'available_algorithms' in status
        assert 'memory_bank_size' in status
        assert 'timestamp' in status

class TestPerformanceOptimizer:
    """Test Performance Optimizer functionality"""
    
    @pytest.mark.asyncio
    async def test_performance_optimizer_initialization(self):
        """Test performance optimizer initialization"""
        assert performance_optimizer.status == OptimizationStatus.IDLE
        assert len(performance_optimizer.optimization_tasks) == 0
        assert len(performance_optimizer.optimization_results) == 0
        assert len(performance_optimizer.optimization_algorithms) == 8
    
    @pytest.mark.asyncio
    async def test_start_stop_optimization_service(self):
        """Test starting and stopping optimization service"""
        # Start service
        await performance_optimizer.start_optimization_service()
        assert performance_optimizer.status == OptimizationStatus.IDLE
        
        # Stop service
        await performance_optimizer.stop_optimization_service()
        assert performance_optimizer.status == OptimizationStatus.IDLE
    
    @pytest.mark.asyncio
    async def test_create_optimization_task(self):
        """Test creating optimization task"""
        baseline_metrics = {
            "latency": 100.0,
            "memory_usage": 1000.0,
            "accuracy": 0.9
        }
        target_metrics = {
            "latency": 50.0,
            "memory_usage": 500.0,
            "accuracy": 0.85
        }
        
        task_id = await performance_optimizer.create_optimization_task(
            optimization_type=OptimizationType.QUANTIZATION,
            target=OptimizationTarget.LATENCY,
            model_id="test_model",
            baseline_metrics=baseline_metrics,
            target_metrics=target_metrics
        )
        
        assert task_id is not None
        assert task_id in performance_optimizer.optimization_tasks
        assert performance_optimizer.metrics.total_optimizations == 1
    
    @pytest.mark.asyncio
    async def test_execute_optimization(self):
        """Test executing optimization"""
        # Create task first
        baseline_metrics = {
            "latency": 100.0,
            "memory_usage": 1000.0,
            "accuracy": 0.9
        }
        target_metrics = {
            "latency": 50.0,
            "memory_usage": 500.0,
            "accuracy": 0.85
        }
        
        task_id = await performance_optimizer.create_optimization_task(
            optimization_type=OptimizationType.QUANTIZATION,
            target=OptimizationTarget.LATENCY,
            model_id="test_model",
            baseline_metrics=baseline_metrics,
            target_metrics=target_metrics
        )
        
        # Execute optimization
        result = await performance_optimizer.execute_optimization(task_id)
        
        assert result is not None
        assert result.task_id == task_id
        assert result.optimization_type == "quantization"
        assert result.optimization_time > 0.0
        assert result.performance_gain >= 0.0
        assert result.memory_savings >= 0.0
        assert result.speed_improvement >= 0.0
    
    @pytest.mark.asyncio
    async def test_get_optimization_status(self):
        """Test getting optimization status"""
        status = await performance_optimizer.get_optimization_status()
        
        assert 'status' in status
        assert 'total_optimizations' in status
        assert 'successful_optimizations' in status
        assert 'failed_optimizations' in status
        assert 'average_performance_gain' in status
        assert 'available_optimization_types' in status
        assert 'timestamp' in status

class TestServiceIntegration:
    """Test service integration and end-to-end functionality"""
    
    @pytest.mark.asyncio
    async def test_service_startup_sequence(self):
        """Test starting all services in sequence"""
        # Start quantum orchestrator
        await quantum_orchestrator.start_orchestration()
        assert quantum_orchestrator.status == OrchestrationStatus.RUNNING
        
        # Start meta-learning hub
        await meta_learning_hub.start_meta_learning_hub()
        assert meta_learning_hub.status == "running"
        
        # Start model ensemble
        await model_ensemble.start_ensemble()
        assert model_ensemble.status == EnsembleStatus.READY
        
        # Start continual learner
        await continual_learner.start_continual_learning()
        assert continual_learner.status == LearningStatus.IDLE
        
        # Start performance optimizer
        await performance_optimizer.start_optimization_service()
        assert performance_optimizer.status == OptimizationStatus.IDLE
    
    @pytest.mark.asyncio
    async def test_service_shutdown_sequence(self):
        """Test stopping all services in sequence"""
        # Stop quantum orchestrator
        await quantum_orchestrator.stop_orchestration()
        assert quantum_orchestrator.status == OrchestrationStatus.IDLE
        
        # Stop meta-learning hub
        await meta_learning_hub.stop_meta_learning_hub()
        assert meta_learning_hub.status == "stopped"
        
        # Stop model ensemble
        await model_ensemble.stop_ensemble()
        assert model_ensemble.status == EnsembleStatus.INITIALIZING
        
        # Stop continual learner
        await continual_learner.stop_continual_learning()
        assert continual_learner.status == LearningStatus.IDLE
        
        # Stop performance optimizer
        await performance_optimizer.stop_optimization_service()
        assert performance_optimizer.status == OptimizationStatus.IDLE
    
    @pytest.mark.asyncio
    async def test_cross_service_communication(self):
        """Test communication between services"""
        # Create a meta-learning task
        support_set = [{"features": [1, 2, 3], "label": 0}]
        query_set = [{"features": [4, 5, 6], "label": 1}]
        
        task_id = await meta_learning_hub.create_meta_task(
            task_type="classification",
            support_set=support_set,
            query_set=query_set,
            num_classes=2,
            num_shots=1,
            difficulty=0.5
        )
        
        # Execute meta-learning
        meta_result = await meta_learning_hub.execute_meta_learning(task_id, MetaLearningType.MAML)
        
        # Use result in ensemble prediction
        input_data = {"features": [1, 2, 3, 4, 5]}
        ensemble_prediction = await model_ensemble.predict(input_data)
        
        # Optimize the ensemble
        baseline_metrics = {
            "latency": 100.0,
            "memory_usage": 1000.0,
            "accuracy": ensemble_prediction.confidence
        }
        target_metrics = {
            "latency": 50.0,
            "memory_usage": 500.0,
            "accuracy": ensemble_prediction.confidence * 0.95
        }
        
        opt_task_id = await performance_optimizer.create_optimization_task(
            optimization_type=OptimizationType.INFERENCE_OPTIMIZATION,
            target=OptimizationTarget.LATENCY,
            model_id="ensemble_model",
            baseline_metrics=baseline_metrics,
            target_metrics=target_metrics
        )
        
        opt_result = await performance_optimizer.execute_optimization(opt_task_id)
        
        # Verify all results
        assert meta_result is not None
        assert ensemble_prediction is not None
        assert opt_result is not None
        assert meta_result.accuracy > 0.0
        assert ensemble_prediction.confidence > 0.0
        assert opt_result.performance_gain >= 0.0

# Test runner
if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])




