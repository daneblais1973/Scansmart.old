"""
Test Suite for Data Service
===========================
Comprehensive test suite for the Enhanced Data Service
"""

import asyncio
import logging
import sys
import os
import numpy as np
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import data service components
from quantum_data_processor import QuantumDataProcessor, ProcessingType, ProcessingStatus, DataPoint, ProcessingResult
from real_time_feeder import RealTimeFeeder, DataSource, FeedStatus, DataStream, StreamData, FeedResult
from feature_engineer import FeatureEngineer, FeatureType, EngineeringMethod, EngineeringStatus, Feature, FeatureSet, EngineeringResult
from data_quality_ai import DataQualityAI, QualityDimension, QualityLevel, AssessmentStatus, QualityMetric, QualityAssessment

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataServiceTester:
    """Comprehensive tester for the Data Service"""
    
    def __init__(self):
        self.quantum_data_processor = QuantumDataProcessor()
        self.real_time_feeder = RealTimeFeeder()
        self.feature_engineer = FeatureEngineer()
        self.data_quality_ai = DataQualityAI()
        
        self.test_results = {
            'quantum_data_processor': {'passed': 0, 'failed': 0, 'total': 0},
            'real_time_feeder': {'passed': 0, 'failed': 0, 'total': 0},
            'feature_engineer': {'passed': 0, 'failed': 0, 'total': 0},
            'data_quality_ai': {'passed': 0, 'failed': 0, 'total': 0}
        }
        
        logger.info("Data Service Tester initialized")
    
    async def run_all_tests(self):
        """Run all data service tests"""
        try:
            logger.info("Starting comprehensive data service tests...")
            
            # Test quantum data processor
            await self._test_quantum_data_processor()
            
            # Test real-time feeder
            await self._test_real_time_feeder()
            
            # Test feature engineer
            await self._test_feature_engineer()
            
            # Test data quality AI
            await self._test_data_quality_ai()
            
            # Test integration
            await self._test_integration()
            
            # Print results
            self._print_test_results()
            
        except Exception as e:
            logger.error(f"Error running tests: {e}")
            raise
    
    async def _test_quantum_data_processor(self):
        """Test quantum data processor functionality"""
        try:
            logger.info("Testing Quantum Data Processor...")
            
            # Start service
            await self.quantum_data_processor.start_processing_service()
            self._assert_test('quantum_data_processor', 'start_service', 
                            self.quantum_data_processor.status == ProcessingStatus.IDLE)
            
            # Test processing pipelines
            self._assert_test('quantum_data_processor', 'processing_pipelines', 
                            len(self.quantum_data_processor.processing_types) > 0)
            
            # Test data processing
            data_points = [
                DataPoint(
                    point_id=str(uuid.uuid4()),
                    timestamp=datetime.now(),
                    data={'values': [1, 2, 3, 4, 5], 'price': 150.0, 'volume': 1000000},
                    quality_score=0.8
                ),
                DataPoint(
                    point_id=str(uuid.uuid4()),
                    timestamp=datetime.now(),
                    data={'values': [2, 3, 4, 5, 6], 'price': 151.0, 'volume': 1100000},
                    quality_score=0.9
                )
            ]
            
            processing_result = await self.quantum_data_processor.process_data(
                data_points, ProcessingType.PREPROCESSING
            )
            self._assert_test('quantum_data_processor', 'data_processing', 
                            isinstance(processing_result, ProcessingResult))
            
            # Test status
            status = await self.quantum_data_processor.get_processing_status()
            self._assert_test('quantum_data_processor', 'get_status', 
                            'status' in status and 'total_processings' in status)
            
            # Stop service
            await self.quantum_data_processor.stop_processing_service()
            self._assert_test('quantum_data_processor', 'stop_service', 
                            self.quantum_data_processor.status == ProcessingStatus.IDLE)
            
            logger.info("Quantum Data Processor tests completed")
            
        except Exception as e:
            logger.error(f"Error testing quantum data processor: {e}")
            self.test_results['quantum_data_processor']['failed'] += 1
    
    async def _test_real_time_feeder(self):
        """Test real-time feeder functionality"""
        try:
            logger.info("Testing Real-Time Feeder...")
            
            # Start service
            await self.real_time_feeder.start_feed_service()
            self._assert_test('real_time_feeder', 'start_service', 
                            self.real_time_feeder.status == FeedStatus.IDLE)
            
            # Test feed components
            self._assert_test('real_time_feeder', 'feed_components', 
                            len(self.real_time_feeder.feed_components) > 0)
            
            # Test stream creation
            stream_id = await self.real_time_feeder.create_stream(
                DataSource.MARKET_DATA, 'price_data', frequency=1.0, buffer_size=100
            )
            self._assert_test('real_time_feeder', 'create_stream', 
                            stream_id is not None)
            
            # Test stream start
            start_success = await self.real_time_feeder.start_stream(stream_id)
            self._assert_test('real_time_feeder', 'start_stream', 
                            start_success)
            
            # Wait for some data
            await asyncio.sleep(2)
            
            # Test stream data
            stream_data = await self.real_time_feeder.get_stream_data(stream_id, limit=10)
            self._assert_test('real_time_feeder', 'get_stream_data', 
                            len(stream_data) > 0)
            
            # Test stream stop
            stop_success = await self.real_time_feeder.stop_stream(stream_id)
            self._assert_test('real_time_feeder', 'stop_stream', 
                            stop_success)
            
            # Test status
            status = await self.real_time_feeder.get_feed_status()
            self._assert_test('real_time_feeder', 'get_status', 
                            'status' in status and 'total_streams' in status)
            
            # Stop service
            await self.real_time_feeder.stop_feed_service()
            self._assert_test('real_time_feeder', 'stop_service', 
                            self.real_time_feeder.status == FeedStatus.IDLE)
            
            logger.info("Real-Time Feeder tests completed")
            
        except Exception as e:
            logger.error(f"Error testing real-time feeder: {e}")
            self.test_results['real_time_feeder']['failed'] += 1
    
    async def _test_feature_engineer(self):
        """Test feature engineer functionality"""
        try:
            logger.info("Testing Feature Engineer...")
            
            # Start service
            await self.feature_engineer.start_engineering_service()
            self._assert_test('feature_engineer', 'start_service', 
                            self.feature_engineer.status == EngineeringStatus.IDLE)
            
            # Test engineering methods
            self._assert_test('feature_engineer', 'engineering_methods', 
                            len(self.feature_engineer.engineering_methods) > 0)
            
            # Test feature engineering
            data = {
                'price': 150.0,
                'volume': 1000000,
                'earnings': 5.0,
                'shares': 10000000,
                'news_sentiment': 0.7,
                'social_sentiment': 0.6,
                'gdp': 2.5,
                'interest_rate': 0.05,
                'revenue': 1000000000,
                'user_activity': 0.8,
                'timestamp': datetime.now().isoformat(),
                'sector': 'Technology'
            }
            
            feature_types = [FeatureType.TECHNICAL, FeatureType.FUNDAMENTAL, FeatureType.SENTIMENT]
            engineering_result = await self.feature_engineer.engineer_features(
                data, feature_types, EngineeringMethod.AUTOMATED
            )
            self._assert_test('feature_engineer', 'feature_engineering', 
                            isinstance(engineering_result, EngineeringResult))
            
            # Test status
            status = await self.feature_engineer.get_engineering_status()
            self._assert_test('feature_engineer', 'get_status', 
                            'status' in status and 'total_engineerings' in status)
            
            # Stop service
            await self.feature_engineer.stop_engineering_service()
            self._assert_test('feature_engineer', 'stop_service', 
                            self.feature_engineer.status == EngineeringStatus.IDLE)
            
            logger.info("Feature Engineer tests completed")
            
        except Exception as e:
            logger.error(f"Error testing feature engineer: {e}")
            self.test_results['feature_engineer']['failed'] += 1
    
    async def _test_data_quality_ai(self):
        """Test data quality AI functionality"""
        try:
            logger.info("Testing Data Quality AI...")
            
            # Start service
            await self.data_quality_ai.start_quality_service()
            self._assert_test('data_quality_ai', 'start_service', 
                            self.data_quality_ai.status == AssessmentStatus.IDLE)
            
            # Test quality dimensions
            self._assert_test('data_quality_ai', 'quality_dimensions', 
                            len(self.data_quality_ai.quality_dimensions) > 0)
            
            # Test quality assessment
            data = {
                'price': 150.0,
                'volume': 1000000,
                'timestamp': datetime.now().isoformat(),
                'source': 'market_data',
                'quality_indicators': {
                    'completeness': 0.95,
                    'accuracy': 0.90,
                    'consistency': 0.85
                }
            }
            
            quality_dimensions = [QualityDimension.COMPLETENESS, QualityDimension.ACCURACY, QualityDimension.CONSISTENCY]
            quality_assessment = await self.data_quality_ai.assess_data_quality(data, quality_dimensions)
            self._assert_test('data_quality_ai', 'quality_assessment', 
                            isinstance(quality_assessment, QualityAssessment))
            
            # Test status
            status = await self.data_quality_ai.get_quality_status()
            self._assert_test('data_quality_ai', 'get_status', 
                            'status' in status and 'total_assessments' in status)
            
            # Stop service
            await self.data_quality_ai.stop_quality_service()
            self._assert_test('data_quality_ai', 'stop_service', 
                            self.data_quality_ai.status == AssessmentStatus.IDLE)
            
            logger.info("Data Quality AI tests completed")
            
        except Exception as e:
            logger.error(f"Error testing data quality AI: {e}")
            self.test_results['data_quality_ai']['failed'] += 1
    
    async def _test_integration(self):
        """Test integration between components"""
        try:
            logger.info("Testing component integration...")
            
            # Start all services
            await self.quantum_data_processor.start_processing_service()
            await self.real_time_feeder.start_feed_service()
            await self.feature_engineer.start_engineering_service()
            await self.data_quality_ai.start_quality_service()
            
            # Test end-to-end workflow
            # Step 1: Create data stream
            stream_id = await self.real_time_feeder.create_stream(
                DataSource.MARKET_DATA, 'price_data', frequency=1.0, buffer_size=50
            )
            self._assert_test('integration', 'create_stream', 
                            stream_id is not None)
            
            # Step 2: Start stream and collect data
            await self.real_time_feeder.start_stream(stream_id)
            await asyncio.sleep(3)  # Collect some data
            
            # Step 3: Get stream data
            stream_data = await self.real_time_feeder.get_stream_data(stream_id, limit=5)
            self._assert_test('integration', 'get_stream_data', 
                            len(stream_data) > 0)
            
            # Step 4: Process data
            data_points = []
            for data in stream_data:
                data_point = DataPoint(
                    point_id=data.data_id,
                    timestamp=data.timestamp,
                    data=data.data,
                    quality_score=data.quality_score
                )
                data_points.append(data_point)
            
            processing_result = await self.quantum_data_processor.process_data(
                data_points, ProcessingType.FEATURE_EXTRACTION
            )
            self._assert_test('integration', 'data_processing', 
                            isinstance(processing_result, ProcessingResult))
            
            # Step 5: Engineer features
            if processing_result.processed_data:
                sample_data = processing_result.processed_data[0].data
                feature_result = await self.feature_engineer.engineer_features(
                    sample_data, [FeatureType.TECHNICAL, FeatureType.FUNDAMENTAL]
                )
                self._assert_test('integration', 'feature_engineering', 
                                isinstance(feature_result, EngineeringResult))
            
            # Step 6: Assess quality
            if processing_result.processed_data:
                sample_data = processing_result.processed_data[0].data
                quality_result = await self.data_quality_ai.assess_data_quality(
                    sample_data, [QualityDimension.COMPLETENESS, QualityDimension.ACCURACY]
                )
                self._assert_test('integration', 'quality_assessment', 
                                isinstance(quality_result, QualityAssessment))
            
            # Stop stream
            await self.real_time_feeder.stop_stream(stream_id)
            
            # Stop all services
            await self.quantum_data_processor.stop_processing_service()
            await self.real_time_feeder.stop_feed_service()
            await self.feature_engineer.stop_engineering_service()
            await self.data_quality_ai.stop_quality_service()
            
            logger.info("Integration tests completed")
            
        except Exception as e:
            logger.error(f"Error testing integration: {e}")
            self.test_results['integration'] = {'passed': 0, 'failed': 1, 'total': 1}
    
    def _assert_test(self, component: str, test_name: str, condition: bool):
        """Assert test condition and update results"""
        try:
            if component not in self.test_results:
                self.test_results[component] = {'passed': 0, 'failed': 0, 'total': 0}
            
            self.test_results[component]['total'] += 1
            
            if condition:
                self.test_results[component]['passed'] += 1
                logger.info(f"✓ {component}.{test_name} PASSED")
            else:
                self.test_results[component]['failed'] += 1
                logger.error(f"✗ {component}.{test_name} FAILED")
                
        except Exception as e:
            logger.error(f"Error asserting test {component}.{test_name}: {e}")
            if component in self.test_results:
                self.test_results[component]['failed'] += 1
    
    def _print_test_results(self):
        """Print comprehensive test results"""
        try:
            print("\n" + "="*80)
            print("DATA SERVICE TEST RESULTS")
            print("="*80)
            
            total_passed = 0
            total_failed = 0
            total_tests = 0
            
            for component, results in self.test_results.items():
                passed = results['passed']
                failed = results['failed']
                total = results['total']
                
                total_passed += passed
                total_failed += failed
                total_tests += total
                
                success_rate = (passed / total * 100) if total > 0 else 0
                
                print(f"\n{component.upper()}:")
                print(f"  Passed: {passed}")
                print(f"  Failed: {failed}")
                print(f"  Total:  {total}")
                print(f"  Success Rate: {success_rate:.1f}%")
            
            overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
            
            print(f"\nOVERALL RESULTS:")
            print(f"  Total Passed: {total_passed}")
            print(f"  Total Failed: {total_failed}")
            print(f"  Total Tests:  {total_tests}")
            print(f"  Overall Success Rate: {overall_success_rate:.1f}%")
            
            if overall_success_rate >= 95:
                print(f"\nEXCELLENT! Data Service is working at {overall_success_rate:.1f}% functionality!")
            elif overall_success_rate >= 90:
                print(f"\nGOOD! Data Service is working at {overall_success_rate:.1f}% functionality!")
            elif overall_success_rate >= 80:
                print(f"\nFAIR! Data Service is working at {overall_success_rate:.1f}% functionality!")
            else:
                print(f"\nPOOR! Data Service is working at {overall_success_rate:.1f}% functionality!")
            
            print("="*80)
            
        except Exception as e:
            logger.error(f"Error printing test results: {e}")

async def main():
    """Main test function"""
    try:
        tester = DataServiceTester()
        await tester.run_all_tests()
        
    except Exception as e:
        logger.error(f"Error in main test function: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
