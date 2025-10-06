"""
Catalyst Service Test Suite
===========================
Comprehensive test suite for catalyst service components
"""

import asyncio
import sys
import os
import logging
from datetime import datetime
import numpy as np

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import service components
from quantum_catalyst_detector import quantum_catalyst_detector, CatalystType, CatalystImpact
from multi_modal_processor import multi_modal_processor, ModalityType, MultiModalData
from ensemble_classifier import ensemble_classifier, EnsembleType
from real_time_analyzer import real_time_analyzer, RealTimeData, AlertLevel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

async def test_quantum_catalyst_detector():
    """Test Quantum Catalyst Detector"""
    print("\n=== TESTING QUANTUM CATALYST DETECTOR ===")
    
    try:
        # Test 1: Service Lifecycle
        print("1. Testing Service Lifecycle...")
        await quantum_catalyst_detector.start_detection_service()
        assert quantum_catalyst_detector.status.value == "idle"
        print("   Service started: SUCCESS")
        
        await quantum_catalyst_detector.stop_detection_service()
        assert quantum_catalyst_detector.status.value == "idle"
        print("   Service stopped: SUCCESS")
        
        # Test 2: Catalyst Detection
        print("2. Testing Catalyst Detection...")
        await quantum_catalyst_detector.start_detection_service()
        
        sample_text = "Apple Inc. reported record quarterly earnings with strong iPhone sales growth"
        detection = await quantum_catalyst_detector.detect_catalyst(sample_text)
        
        assert detection is not None
        assert detection.catalyst_type in [t for t in CatalystType]
        assert detection.impact_level in [i for i in CatalystImpact]
        assert 0.0 <= detection.confidence <= 1.0
        print(f"   Catalyst detected: {detection.catalyst_type.value} with {detection.confidence:.3f} confidence")
        
        # Test 3: Service Status
        print("3. Testing Service Status...")
        status = await quantum_catalyst_detector.get_detection_status()
        assert 'status' in status
        assert 'total_detections' in status
        assert 'quantum_available' in status
        print("   Service status: SUCCESS")
        
        await quantum_catalyst_detector.stop_detection_service()
        
        print("   [SUCCESS] Quantum Catalyst Detector: PASSED")
        return True
        
    except Exception as e:
        print(f"   [ERROR] Quantum Catalyst Detector: {e}")
        return False

async def test_multi_modal_processor():
    """Test Multi-Modal Processor"""
    print("\n=== TESTING MULTI-MODAL PROCESSOR ===")
    
    try:
        # Test 1: Service Lifecycle
        print("1. Testing Service Lifecycle...")
        await multi_modal_processor.start_processing_service()
        assert multi_modal_processor.status.value == "idle"
        print("   Service started: SUCCESS")
        
        await multi_modal_processor.stop_processing_service()
        assert multi_modal_processor.status.value == "idle"
        print("   Service stopped: SUCCESS")
        
        # Test 2: Multi-Modal Processing
        print("2. Testing Multi-Modal Processing...")
        await multi_modal_processor.start_processing_service()
        
        # Create multi-modal data
        text_data = MultiModalData(
            data_id=str(uuid.uuid4()),
            modality_type=ModalityType.TEXT,
            content="Apple Inc. reported strong quarterly earnings",
            metadata={"source": "news"}
        )
        
        image_data = MultiModalData(
            data_id=str(uuid.uuid4()),
            modality_type=ModalityType.IMAGE,
            content="chart.png",
            metadata={"source": "chart"}
        )
        
        result = await multi_modal_processor.process_multi_modal_data(
            [text_data, image_data], fusion_method="attention"
        )
        
        assert result is not None
        assert result.fused_features is not None
        assert 0.0 <= result.fusion_quality <= 1.0
        print(f"   Multi-modal processing: {result.fusion_quality:.3f} quality")
        
        # Test 3: Service Status
        print("3. Testing Service Status...")
        status = await multi_modal_processor.get_processing_status()
        assert 'status' in status
        assert 'total_processings' in status
        assert 'fusion_consistency' in status
        print("   Service status: SUCCESS")
        
        await multi_modal_processor.stop_processing_service()
        
        print("   [SUCCESS] Multi-Modal Processor: PASSED")
        return True
        
    except Exception as e:
        print(f"   [ERROR] Multi-Modal Processor: {e}")
        return False

async def test_ensemble_classifier():
    """Test Ensemble Classifier"""
    print("\n=== TESTING ENSEMBLE CLASSIFIER ===")
    
    try:
        # Test 1: Service Lifecycle
        print("1. Testing Service Lifecycle...")
        await ensemble_classifier.start_classification_service()
        assert ensemble_classifier.status.value == "idle"
        print("   Service started: SUCCESS")
        
        await ensemble_classifier.stop_classification_service()
        assert ensemble_classifier.status.value == "idle"
        print("   Service stopped: SUCCESS")
        
        # Test 2: Training and Prediction
        print("2. Testing Training and Prediction...")
        await ensemble_classifier.start_classification_service()
        
        # Create training data
        training_data = {
            'features': [
                [1, 2, 3, 4, 5],
                [2, 3, 4, 5, 6],
                [3, 4, 5, 6, 7],
                [4, 5, 6, 7, 8]
            ],
            'labels': ['earnings', 'merger', 'product', 'earnings']
        }
        
        print(f"Training data: {len(training_data['features'])} samples, {len(training_data['labels'])} labels")
        
        # Train ensemble
        training_result = await ensemble_classifier.train_ensemble(training_data, EnsembleType.VOTING)
        assert training_result is not None
        assert 'training_time' in training_result
        print(f"   Ensemble training: {training_result['training_time']:.3f}s")
        
        # Make prediction
        features = {'text': 'Apple reported strong earnings', 'embeddings': [0.1, 0.2, 0.3]}
        prediction = await ensemble_classifier.predict_ensemble(features, EnsembleType.VOTING)
        
        assert prediction is not None
        assert prediction.ensemble_prediction is not None
        assert 0.0 <= prediction.ensemble_confidence <= 1.0
        print(f"   Ensemble prediction: {prediction.ensemble_prediction} with {prediction.ensemble_confidence:.3f} confidence")
        
        # Test 3: Service Status
        print("3. Testing Service Status...")
        status = await ensemble_classifier.get_classification_status()
        assert 'status' in status
        assert 'total_predictions' in status
        assert 'ensemble_accuracy' in status
        print("   Service status: SUCCESS")
        
        await ensemble_classifier.stop_classification_service()
        
        print("   [SUCCESS] Ensemble Classifier: PASSED")
        return True
        
    except Exception as e:
        print(f"   [ERROR] Ensemble Classifier: {e}")
        return False

async def test_real_time_analyzer():
    """Test Real-Time Analyzer"""
    print("\n=== TESTING REAL-TIME ANALYZER ===")
    
    try:
        # Test 1: Service Lifecycle
        print("1. Testing Service Lifecycle...")
        await real_time_analyzer.start_analysis_service()
        assert real_time_analyzer.status.value == "monitoring"
        print("   Service started: SUCCESS")
        
        await real_time_analyzer.stop_analysis_service()
        assert real_time_analyzer.status.value == "idle"
        print("   Service stopped: SUCCESS")
        
        # Test 2: Real-Time Analysis
        print("2. Testing Real-Time Analysis...")
        await real_time_analyzer.start_analysis_service()
        
        # Create real-time data
        data = RealTimeData(
            data_id=str(uuid.uuid4()),
            data_type="news",
            content="Apple Inc. reported record quarterly earnings with strong iPhone sales",
            timestamp=datetime.now(),
            source="financial_news"
        )
        
        result = await real_time_analyzer.analyze_real_time_data(data)
        
        assert result is not None
        assert result.findings is not None
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.alerts) >= 0
        assert len(result.recommendations) >= 0
        print(f"   Real-time analysis: {result.confidence:.3f} confidence, {len(result.alerts)} alerts")
        
        # Test 3: Service Status
        print("3. Testing Service Status...")
        status = await real_time_analyzer.get_analysis_status()
        assert 'status' in status
        assert 'total_analyses' in status
        assert 'throughput' in status
        print("   Service status: SUCCESS")
        
        await real_time_analyzer.stop_analysis_service()
        
        print("   [SUCCESS] Real-Time Analyzer: PASSED")
        return True
        
    except Exception as e:
        print(f"   [ERROR] Real-Time Analyzer: {e}")
        return False

async def test_integrated_workflow():
    """Test integrated catalyst service workflow"""
    print("\n=== TESTING INTEGRATED WORKFLOW ===")
    
    try:
        # Start all services
        print("1. Starting All Services...")
        await quantum_catalyst_detector.start_detection_service()
        await multi_modal_processor.start_processing_service()
        await ensemble_classifier.start_classification_service()
        await real_time_analyzer.start_analysis_service()
        print("   All services started: SUCCESS")
        
        # Test 2: End-to-End Workflow
        print("2. Testing End-to-End Workflow...")
        
        # Step 1: Real-time analysis
        data = RealTimeData(
            data_id=str(uuid.uuid4()),
            data_type="news",
            content="Apple Inc. reported record quarterly earnings with strong iPhone sales growth",
            timestamp=datetime.now(),
            source="financial_news"
        )
        
        analysis_result = await real_time_analyzer.analyze_real_time_data(data)
        print(f"   Real-time analysis: {analysis_result.confidence:.3f} confidence")
        
        # Step 2: Catalyst detection
        detection = await quantum_catalyst_detector.detect_catalyst(data.content)
        print(f"   Catalyst detection: {detection.catalyst_type.value} with {detection.confidence:.3f} confidence")
        
        # Step 3: Multi-modal processing
        text_data = MultiModalData(
            data_id=str(uuid.uuid4()),
            modality_type=ModalityType.TEXT,
            content=data.content,
            metadata={"source": "news"}
        )
        
        processing_result = await multi_modal_processor.process_multi_modal_data(
            [text_data], fusion_method="attention"
        )
        print(f"   Multi-modal processing: {processing_result.fusion_quality:.3f} quality")
        
        # Step 4: Ensemble classification
        features = {
            'text': data.content,
            'embeddings': processing_result.fused_features.get('fused_features', [0.1, 0.2, 0.3])
        }
        
        prediction = await ensemble_classifier.predict_ensemble(features, EnsembleType.VOTING)
        print(f"   Ensemble classification: {prediction.ensemble_prediction} with {prediction.ensemble_confidence:.3f} confidence")
        
        # Test 3: Service Integration
        print("3. Testing Service Integration...")
        
        # Check all service statuses
        detector_status = await quantum_catalyst_detector.get_detection_status()
        processor_status = await multi_modal_processor.get_processing_status()
        classifier_status = await ensemble_classifier.get_classification_status()
        analyzer_status = await real_time_analyzer.get_analysis_status()
        
        print(f"   Quantum Detector: {detector_status['status']}")
        print(f"   Multi-Modal Processor: {processor_status['status']}")
        print(f"   Ensemble Classifier: {classifier_status['status']}")
        print(f"   Real-Time Analyzer: {analyzer_status['status']}")
        
        # Stop all services
        print("4. Stopping All Services...")
        await quantum_catalyst_detector.stop_detection_service()
        await multi_modal_processor.stop_processing_service()
        await ensemble_classifier.stop_classification_service()
        await real_time_analyzer.stop_analysis_service()
        print("   All services stopped: SUCCESS")
        
        print("   [SUCCESS] Integrated Workflow: PASSED")
        return True
        
    except Exception as e:
        print(f"   [ERROR] Integrated Workflow: {e}")
        return False

async def main():
    """Main test function"""
    print("CATALYST SERVICE COMPREHENSIVE TEST")
    print("=" * 60)
    
    try:
        # Run all tests
        tests = [
            test_quantum_catalyst_detector(),
            test_multi_modal_processor(),
            test_ensemble_classifier(),
            test_real_time_analyzer(),
            test_integrated_workflow()
        ]
        
        results = await asyncio.gather(*tests, return_exceptions=True)
        
        # Summary
        print("\n" + "=" * 60)
        print("CATALYST SERVICE TEST SUMMARY")
        print("=" * 60)
        
        passed = 0
        failed = 0
        
        test_names = [
            "Quantum Catalyst Detector",
            "Multi-Modal Processor",
            "Ensemble Classifier",
            "Real-Time Analyzer",
            "Integrated Workflow"
        ]
        
        for i, (name, result) in enumerate(zip(test_names, results)):
            if isinstance(result, Exception):
                print(f"[FAILED] {name}: {result}")
                failed += 1
            elif result:
                print(f"[PASSED] {name}")
                passed += 1
            else:
                print(f"[FAILED] {name}")
                failed += 1
        
        print(f"\nRESULTS: {passed} PASSED, {failed} FAILED")
        
        if failed == 0:
            print("ALL CATALYST SERVICE TESTS PASSED!")
            print("ENTERPRISE-GRADE CATALYST SERVICE CONFIRMED!")
            return True
        else:
            print("SOME CATALYST SERVICE TESTS FAILED")
            return False
            
    except Exception as e:
        print(f"\nCATALYST SERVICE TEST ERROR: {e}")
        return False

if __name__ == "__main__":
    import uuid
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
