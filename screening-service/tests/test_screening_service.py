"""
Test Suite for Screening Service
================================
Comprehensive test suite for the AI-Enhanced Screening Service
"""

import asyncio
import logging
import sys
import os
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import screening service components
from quantum_screener import QuantumScreener, ScreeningCriteria, ScreeningStatus, StockData, ScreeningResult
from multi_model_ranker import MultiModelRanker, RankingMethod, RankingStatus, RankingResult
from portfolio_optimizer import PortfolioOptimizer, OptimizationMethod, OptimizationStatus, PortfolioAsset, OptimizationResult
from risk_analyzer import RiskAnalyzer, RiskType, RiskLevel, AnalysisStatus, RiskAnalysisResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScreeningServiceTester:
    """Comprehensive tester for the Screening Service"""
    
    def __init__(self):
        self.quantum_screener = QuantumScreener()
        self.multi_model_ranker = MultiModelRanker()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.risk_analyzer = RiskAnalyzer()
        
        self.test_results = {
            'quantum_screener': {'passed': 0, 'failed': 0, 'total': 0},
            'multi_model_ranker': {'passed': 0, 'failed': 0, 'total': 0},
            'portfolio_optimizer': {'passed': 0, 'failed': 0, 'total': 0},
            'risk_analyzer': {'passed': 0, 'failed': 0, 'total': 0}
        }
        
        logger.info("Screening Service Tester initialized")
    
    async def run_all_tests(self):
        """Run all screening service tests"""
        try:
            logger.info("Starting comprehensive screening service tests...")
            
            # Test quantum screener
            await self._test_quantum_screener()
            
            # Test multi-model ranker
            await self._test_multi_model_ranker()
            
            # Test portfolio optimizer
            await self._test_portfolio_optimizer()
            
            # Test risk analyzer
            await self._test_risk_analyzer()
            
            # Test integration
            await self._test_integration()
            
            # Print results
            self._print_test_results()
            
        except Exception as e:
            logger.error(f"Error running tests: {e}")
            raise
    
    async def _test_quantum_screener(self):
        """Test quantum screener functionality"""
        try:
            logger.info("Testing Quantum Screener...")
            
            # Start service
            await self.quantum_screener.start_screening_service()
            self._assert_test('quantum_screener', 'start_service', 
                            self.quantum_screener.status == ScreeningStatus.IDLE)
            
            # Test screening criteria
            criteria = [ScreeningCriteria.TECHNICAL, ScreeningCriteria.FUNDAMENTAL]
            self._assert_test('quantum_screener', 'screening_criteria', 
                            len(self.quantum_screener.screening_criteria) > 0)
            
            # Test stock screening
            stock_data = StockData(
                symbol='AAPL',
                name='Apple Inc.',
                price=150.0,
                market_cap=2500000000000,
                volume=50000000,
                sector='Technology',
                industry='Consumer Electronics',
                pe_ratio=25.0,
                pb_ratio=5.0,
                debt_to_equity=0.3,
                roe=0.2,
                revenue_growth=0.1,
                earnings_growth=0.15,
                beta=1.2,
                rsi=60.0,
                macd=0.5,
                sma_50=145.0,
                sma_200=140.0
            )
            
            screening_result = await self.quantum_screener.screen_stock(stock_data, criteria)
            self._assert_test('quantum_screener', 'stock_screening', 
                            isinstance(screening_result, ScreeningResult))
            
            # Test status
            status = await self.quantum_screener.get_screening_status()
            self._assert_test('quantum_screener', 'get_status', 
                            'status' in status and 'total_screenings' in status)
            
            # Stop service
            await self.quantum_screener.stop_screening_service()
            self._assert_test('quantum_screener', 'stop_service', 
                            self.quantum_screener.status == ScreeningStatus.IDLE)
            
            logger.info("Quantum Screener tests completed")
            
        except Exception as e:
            logger.error(f"Error testing quantum screener: {e}")
            self.test_results['quantum_screener']['failed'] += 1
    
    async def _test_multi_model_ranker(self):
        """Test multi-model ranker functionality"""
        try:
            logger.info("Testing Multi-Model Ranker...")
            
            # Start service
            await self.multi_model_ranker.start_ranking_service()
            self._assert_test('multi_model_ranker', 'start_service', 
                            self.multi_model_ranker.status == RankingStatus.IDLE)
            
            # Test ranking models
            self._assert_test('multi_model_ranker', 'ranking_models', 
                            len(self.multi_model_ranker.ranking_models) > 0)
            
            # Test stock ranking
            stock_data_list = [
                {
                    'symbol': 'AAPL',
                    'name': 'Apple Inc.',
                    'price': 150.0,
                    'volume': 50000000,
                    'market_cap': 2500000000000,
                    'pe_ratio': 25.0,
                    'pb_ratio': 5.0,
                    'roe': 0.2,
                    'quantum_score': 0.8,
                    'classical_score': 0.7,
                    'combined_score': 0.75
                },
                {
                    'symbol': 'MSFT',
                    'name': 'Microsoft Corporation',
                    'price': 300.0,
                    'volume': 30000000,
                    'market_cap': 2200000000000,
                    'pe_ratio': 30.0,
                    'pb_ratio': 6.0,
                    'roe': 0.25,
                    'quantum_score': 0.85,
                    'classical_score': 0.8,
                    'combined_score': 0.82
                },
                {
                    'symbol': 'GOOGL',
                    'name': 'Alphabet Inc.',
                    'price': 2500.0,
                    'volume': 20000000,
                    'market_cap': 1800000000000,
                    'pe_ratio': 20.0,
                    'pb_ratio': 4.0,
                    'roe': 0.18,
                    'quantum_score': 0.75,
                    'classical_score': 0.72,
                    'combined_score': 0.73
                }
            ]
            
            ranking_result = await self.multi_model_ranker.rank_stocks(
                stock_data_list, RankingMethod.ENSEMBLE
            )
            self._assert_test('multi_model_ranker', 'stock_ranking', 
                            isinstance(ranking_result, RankingResult))
            
            # Test status
            status = await self.multi_model_ranker.get_ranking_status()
            self._assert_test('multi_model_ranker', 'get_status', 
                            'status' in status and 'total_rankings' in status)
            
            # Stop service
            await self.multi_model_ranker.stop_ranking_service()
            self._assert_test('multi_model_ranker', 'stop_service', 
                            self.multi_model_ranker.status == RankingStatus.IDLE)
            
            logger.info("Multi-Model Ranker tests completed")
            
        except Exception as e:
            logger.error(f"Error testing multi-model ranker: {e}")
            self.test_results['multi_model_ranker']['failed'] += 1
    
    async def _test_portfolio_optimizer(self):
        """Test portfolio optimizer functionality"""
        try:
            logger.info("Testing Portfolio Optimizer...")
            
            # Start service
            await self.portfolio_optimizer.start_optimization_service()
            self._assert_test('portfolio_optimizer', 'start_service', 
                            self.portfolio_optimizer.status == OptimizationStatus.IDLE)
            
            # Test optimization methods
            self._assert_test('portfolio_optimizer', 'optimization_methods', 
                            len(self.portfolio_optimizer.optimization_methods) > 0)
            
            # Test portfolio optimization
            assets = [
                PortfolioAsset(
                    symbol='AAPL',
                    name='Apple Inc.',
                    weight=0.0,  # Will be optimized
                    expected_return=0.12,
                    volatility=0.25,
                    beta=1.2,
                    sector='Technology',
                    industry='Consumer Electronics',
                    market_cap=2500000000000,
                    price=150.0
                ),
                PortfolioAsset(
                    symbol='MSFT',
                    name='Microsoft Corporation',
                    weight=0.0,  # Will be optimized
                    expected_return=0.15,
                    volatility=0.22,
                    beta=1.1,
                    sector='Technology',
                    industry='Software',
                    market_cap=2200000000000,
                    price=300.0
                ),
                PortfolioAsset(
                    symbol='GOOGL',
                    name='Alphabet Inc.',
                    weight=0.0,  # Will be optimized
                    expected_return=0.18,
                    volatility=0.28,
                    beta=1.3,
                    sector='Technology',
                    industry='Internet',
                    market_cap=1800000000000,
                    price=2500.0
                )
            ]
            
            optimization_result = await self.portfolio_optimizer.optimize_portfolio(
                assets, OptimizationMethod.MAX_SHARPE
            )
            self._assert_test('portfolio_optimizer', 'portfolio_optimization', 
                            isinstance(optimization_result, OptimizationResult))
            
            # Test status
            status = await self.portfolio_optimizer.get_optimization_status()
            self._assert_test('portfolio_optimizer', 'get_status', 
                            'status' in status and 'total_optimizations' in status)
            
            # Stop service
            await self.portfolio_optimizer.stop_optimization_service()
            self._assert_test('portfolio_optimizer', 'stop_service', 
                            self.portfolio_optimizer.status == OptimizationStatus.IDLE)
            
            logger.info("Portfolio Optimizer tests completed")
            
        except Exception as e:
            logger.error(f"Error testing portfolio optimizer: {e}")
            self.test_results['portfolio_optimizer']['failed'] += 1
    
    async def _test_risk_analyzer(self):
        """Test risk analyzer functionality"""
        try:
            logger.info("Testing Risk Analyzer...")
            
            # Start service
            await self.risk_analyzer.start_analysis_service()
            self._assert_test('risk_analyzer', 'start_service', 
                            self.risk_analyzer.status == AnalysisStatus.IDLE)
            
            # Test risk analyzers
            self._assert_test('risk_analyzer', 'risk_analyzers', 
                            len(self.risk_analyzer.risk_analyzers) > 0)
            
            # Test risk analysis
            portfolio_data = {
                'assets': [
                    {'symbol': 'AAPL', 'weight': 0.4, 'sector': 'Technology'},
                    {'symbol': 'MSFT', 'weight': 0.3, 'sector': 'Technology'},
                    {'symbol': 'GOOGL', 'weight': 0.3, 'sector': 'Technology'}
                ],
                'total_value': 1000000,
                'cash': 100000
            }
            
            risk_types = [RiskType.MARKET, RiskType.CONCENTRATION, RiskType.CORRELATION]
            risk_result = await self.risk_analyzer.analyze_risk(portfolio_data, risk_types)
            self._assert_test('risk_analyzer', 'risk_analysis', 
                            isinstance(risk_result, RiskAnalysisResult))
            
            # Test status
            status = await self.risk_analyzer.get_analysis_status()
            self._assert_test('risk_analyzer', 'get_status', 
                            'status' in status and 'total_analyses' in status)
            
            # Stop service
            await self.risk_analyzer.stop_analysis_service()
            self._assert_test('risk_analyzer', 'stop_service', 
                            self.risk_analyzer.status == AnalysisStatus.IDLE)
            
            logger.info("Risk Analyzer tests completed")
            
        except Exception as e:
            logger.error(f"Error testing risk analyzer: {e}")
            self.test_results['risk_analyzer']['failed'] += 1
    
    async def _test_integration(self):
        """Test integration between components"""
        try:
            logger.info("Testing component integration...")
            
            # Start all services
            await self.quantum_screener.start_screening_service()
            await self.multi_model_ranker.start_ranking_service()
            await self.portfolio_optimizer.start_optimization_service()
            await self.risk_analyzer.start_analysis_service()
            
            # Test end-to-end workflow
            stock_data = StockData(
                symbol='AAPL',
                name='Apple Inc.',
                price=150.0,
                market_cap=2500000000000,
                volume=50000000,
                sector='Technology',
                industry='Consumer Electronics',
                pe_ratio=25.0,
                pb_ratio=5.0,
                debt_to_equity=0.3,
                roe=0.2,
                revenue_growth=0.1,
                earnings_growth=0.15,
                beta=1.2,
                rsi=60.0,
                macd=0.5,
                sma_50=145.0,
                sma_200=140.0
            )
            
            # Step 1: Screen stock
            screening_result = await self.quantum_screener.screen_stock(
                stock_data, [ScreeningCriteria.TECHNICAL, ScreeningCriteria.FUNDAMENTAL]
            )
            self._assert_test('integration', 'screening', 
                            isinstance(screening_result, ScreeningResult))
            
            # Step 2: Rank stocks
            stock_data_list = [
                {
                    'symbol': 'AAPL',
                    'name': 'Apple Inc.',
                    'price': 150.0,
                    'volume': 50000000,
                    'market_cap': 2500000000000,
                    'pe_ratio': 25.0,
                    'pb_ratio': 5.0,
                    'roe': 0.2,
                    'quantum_score': screening_result.quantum_score,
                    'classical_score': screening_result.classical_score,
                    'combined_score': screening_result.combined_score
                }
            ]
            
            ranking_result = await self.multi_model_ranker.rank_stocks(
                stock_data_list, RankingMethod.ENSEMBLE
            )
            self._assert_test('integration', 'ranking', 
                            isinstance(ranking_result, RankingResult))
            
            # Step 3: Optimize portfolio
            assets = [
                PortfolioAsset(
                    symbol='AAPL',
                    name='Apple Inc.',
                    weight=0.0,
                    expected_return=0.12,
                    volatility=0.25,
                    beta=1.2,
                    sector='Technology',
                    industry='Consumer Electronics',
                    market_cap=2500000000000,
                    price=150.0
                )
            ]
            
            optimization_result = await self.portfolio_optimizer.optimize_portfolio(
                assets, OptimizationMethod.MAX_SHARPE
            )
            self._assert_test('integration', 'optimization', 
                            isinstance(optimization_result, OptimizationResult))
            
            # Step 4: Analyze risk
            portfolio_data = {
                'assets': [
                    {'symbol': 'AAPL', 'weight': optimization_result.optimized_weights.get('AAPL', 0.0), 'sector': 'Technology'}
                ],
                'total_value': 1000000,
                'cash': 100000
            }
            
            risk_result = await self.risk_analyzer.analyze_risk(
                portfolio_data, [RiskType.MARKET, RiskType.CONCENTRATION]
            )
            self._assert_test('integration', 'risk_analysis', 
                            isinstance(risk_result, RiskAnalysisResult))
            
            # Stop all services
            await self.quantum_screener.stop_screening_service()
            await self.multi_model_ranker.stop_ranking_service()
            await self.portfolio_optimizer.stop_optimization_service()
            await self.risk_analyzer.stop_analysis_service()
            
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
            print("SCREENING SERVICE TEST RESULTS")
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
                print(f"\nEXCELLENT! Screening Service is working at {overall_success_rate:.1f}% functionality!")
            elif overall_success_rate >= 90:
                print(f"\nGOOD! Screening Service is working at {overall_success_rate:.1f}% functionality!")
            elif overall_success_rate >= 80:
                print(f"\nFAIR! Screening Service is working at {overall_success_rate:.1f}% functionality!")
            else:
                print(f"\nPOOR! Screening Service is working at {overall_success_rate:.1f}% functionality!")
            
            print("="*80)
            
        except Exception as e:
            logger.error(f"Error printing test results: {e}")

async def main():
    """Main test function"""
    try:
        tester = ScreeningServiceTester()
        await tester.run_all_tests()
        
    except Exception as e:
        logger.error(f"Error in main test function: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
