"""
Real-Time Feeder
================
Enterprise-grade real-time data feeding service for AI-enhanced data streaming
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Data streaming imports with graceful fallback
try:
    import aiohttp
    import websockets
    import redis
    import kafka
    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False
    aiohttp = None
    websockets = None
    redis = None
    kafka = None

# Import advanced connection pooling
try:
    from .advanced_connection_pool import (
        AdvancedConnectionPool, ConnectionConfig, ConnectionType, 
        PoolStrategy, BatchOperation
    )
    CONNECTION_POOL_AVAILABLE = True
except ImportError:
    CONNECTION_POOL_AVAILABLE = False
    AdvancedConnectionPool = None
    ConnectionConfig = None
    ConnectionType = None
    PoolStrategy = None
    BatchOperation = None

# AI/ML imports with graceful fallback
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    torch = None

logger = logging.getLogger(__name__)

class DataSource(Enum):
    """Data source types"""
    MARKET_DATA = "market_data"
    NEWS_FEED = "news_feed"
    SOCIAL_MEDIA = "social_media"
    ECONOMIC_INDICATORS = "economic_indicators"
    COMPANY_FILINGS = "company_filings"
    ANALYST_REPORTS = "analyst_reports"
    SENSOR_DATA = "sensor_data"
    USER_BEHAVIOR = "user_behavior"

class FeedStatus(Enum):
    """Feed status levels"""
    IDLE = "idle"
    CONNECTING = "connecting"
    STREAMING = "streaming"
    PROCESSING = "processing"
    ERROR = "error"
    DISCONNECTED = "disconnected"

@dataclass
class DataStream:
    """Data stream container"""
    stream_id: str
    source: DataSource
    data_type: str
    frequency: float  # Hz
    buffer_size: int
    subscribers: List[str] = field(default_factory=list)
    is_active: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StreamData:
    """Stream data container"""
    data_id: str
    stream_id: str
    timestamp: datetime
    data: Dict[str, Any]
    quality_score: float
    is_real_time: bool
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FeedResult:
    """Feed result container"""
    result_id: str
    stream_id: str
    data_count: int
    processing_time: float
    quality_score: float
    latency: float
    throughput: float
    error_count: int
    success_rate: float
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class RealTimeFeedMetrics:
    """Real-time feed metrics"""
    total_streams: int
    active_streams: int
    total_data_points: int
    average_latency: float
    average_throughput: float
    data_quality_score: float
    error_rate: float
    uptime: float
    processing_efficiency: float

class RealTimeFeeder:
    """Enterprise-grade real-time data feeding service"""
    
    def __init__(self):
        self.status = FeedStatus.IDLE
        self.data_streams = {}
        self.feed_results = {}
        self.subscribers = {}
        self.data_buffer = {}
        
        # Real-time components
        self.feed_components = {
            DataSource.MARKET_DATA: self._create_market_data_feeder(),
            DataSource.NEWS_FEED: self._create_news_feed_feeder(),
            DataSource.SOCIAL_MEDIA: self._create_social_media_feeder(),
            DataSource.ECONOMIC_INDICATORS: self._create_economic_indicators_feeder(),
            DataSource.COMPANY_FILINGS: self._create_company_filings_feeder(),
            DataSource.ANALYST_REPORTS: self._create_analyst_reports_feeder(),
            DataSource.SENSOR_DATA: self._create_sensor_data_feeder(),
            DataSource.USER_BEHAVIOR: self._create_user_behavior_feeder()
        }
        
        # Performance tracking
        self.metrics = RealTimeFeedMetrics(
            total_streams=0, active_streams=0, total_data_points=0,
            average_latency=0.0, average_throughput=0.0, data_quality_score=0.0,
            error_rate=0.0, uptime=0.0, processing_efficiency=0.0
        )
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=12)
        
        # Advanced connection pooling
        if CONNECTION_POOL_AVAILABLE:
            self.connection_pools = {}
            self.batch_operations = []
            self._initialize_connection_pools()
            logger.info("Advanced connection pooling initialized")
        else:
            logger.warning("Advanced connection pooling not available")
        
        # Initialize feed components
        self._initialize_feed_components()
        
        logger.info("Real-Time Feeder with advanced connection pooling initialized")
    
    def _initialize_connection_pools(self):
        """Initialize advanced connection pools for different data sources"""
        try:
            if not CONNECTION_POOL_AVAILABLE:
                return
            
            # HTTP connection pool for API calls
            http_config = ConnectionConfig(
                connection_type=ConnectionType.HTTP,
                host="api.example.com",
                port=443,
                max_connections=20,
                min_connections=5,
                pool_strategy=PoolStrategy.LEAST_CONNECTIONS
            )
            self.connection_pools['http'] = AdvancedConnectionPool(http_config)
            
            # Database connection pool
            db_config = ConnectionConfig(
                connection_type=ConnectionType.DATABASE,
                host="localhost",
                port=5432,
                database="scansmart",
                max_connections=15,
                min_connections=3,
                pool_strategy=PoolStrategy.ROUND_ROBIN
            )
            self.connection_pools['database'] = AdvancedConnectionPool(db_config)
            
            # Redis connection pool for caching
            redis_config = ConnectionConfig(
                connection_type=ConnectionType.REDIS,
                host="localhost",
                port=6379,
                max_connections=10,
                min_connections=2,
                pool_strategy=PoolStrategy.LEAST_RESPONSE_TIME
            )
            self.connection_pools['redis'] = AdvancedConnectionPool(redis_config)
            
            logger.info(f"Initialized {len(self.connection_pools)} connection pools")
        except Exception as e:
            logger.error(f"Error initializing connection pools: {e}")
    
    def _initialize_feed_components(self):
        """Initialize feed components"""
        try:
            # Initialize data quality components
            self.quality_components = {
                'scaler': StandardScaler() if AI_AVAILABLE else None,
                'anomaly_detector': IsolationForest(contamination=0.1) if AI_AVAILABLE else None
            }
            
            # Initialize data processing components
            self.processing_components = {
                'data_validator': self._create_data_validator(),
                'data_cleaner': self._create_data_cleaner(),
                'data_enhancer': self._create_data_enhancer(),
                'data_aggregator': self._create_data_aggregator()
            }
            
            logger.info("Feed components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing feed components: {e}")
    
    def _create_data_validator(self) -> Dict[str, Any]:
        """Create data validator"""
        return {
            'type': 'validator',
            'rules': ['schema_validation', 'range_validation', 'type_validation'],
            'description': 'Data validation component'
        }
    
    def _create_data_cleaner(self) -> Dict[str, Any]:
        """Create data cleaner"""
        return {
            'type': 'cleaner',
            'methods': ['outlier_removal', 'missing_value_handling', 'duplicate_removal'],
            'description': 'Data cleaning component'
        }
    
    def _create_data_enhancer(self) -> Dict[str, Any]:
        """Create data enhancer"""
        return {
            'type': 'enhancer',
            'methods': ['feature_engineering', 'data_enrichment', 'quality_improvement'],
            'description': 'Data enhancement component'
        }
    
    def _create_data_aggregator(self) -> Dict[str, Any]:
        """Create data aggregator"""
        return {
            'type': 'aggregator',
            'methods': ['time_aggregation', 'spatial_aggregation', 'statistical_aggregation'],
            'description': 'Data aggregation component'
        }
    
    def _create_market_data_feeder(self) -> Dict[str, Any]:
        """Create market data feeder"""
        return {
            'type': 'market_data',
            'sources': ['yahoo_finance', 'alpha_vantage', 'quandl', 'polygon'],
            'data_types': ['price', 'volume', 'ohlc', 'indicators'],
            'frequency': 1.0,  # 1 Hz
            'description': 'Market data feeder'
        }
    
    def _create_news_feed_feeder(self) -> Dict[str, Any]:
        """Create news feed feeder"""
        return {
            'type': 'news_feed',
            'sources': ['rss', 'api', 'web_scraping'],
            'data_types': ['headlines', 'articles', 'sentiment'],
            'frequency': 0.1,  # 0.1 Hz
            'description': 'News feed feeder'
        }
    
    def _create_social_media_feeder(self) -> Dict[str, Any]:
        """Create social media feeder"""
        return {
            'type': 'social_media',
            'sources': ['twitter', 'reddit', 'stocktwits'],
            'data_types': ['posts', 'sentiment', 'mentions'],
            'frequency': 0.5,  # 0.5 Hz
            'description': 'Social media feeder'
        }
    
    def _create_economic_indicators_feeder(self) -> Dict[str, Any]:
        """Create economic indicators feeder"""
        return {
            'type': 'economic_indicators',
            'sources': ['fred', 'bls', 'census'],
            'data_types': ['gdp', 'inflation', 'employment'],
            'frequency': 0.01,  # 0.01 Hz
            'description': 'Economic indicators feeder'
        }
    
    def _create_company_filings_feeder(self) -> Dict[str, Any]:
        """Create company filings feeder"""
        return {
            'type': 'company_filings',
            'sources': ['sec_edgar', 'sec_api'],
            'data_types': ['10k', '10q', '8k', 'proxy'],
            'frequency': 0.05,  # 0.05 Hz
            'description': 'Company filings feeder'
        }
    
    def _create_analyst_reports_feeder(self) -> Dict[str, Any]:
        """Create analyst reports feeder"""
        return {
            'type': 'analyst_reports',
            'sources': ['bloomberg', 'reuters', 'factset'],
            'data_types': ['ratings', 'targets', 'estimates'],
            'frequency': 0.1,  # 0.1 Hz
            'description': 'Analyst reports feeder'
        }
    
    def _create_sensor_data_feeder(self) -> Dict[str, Any]:
        """Create sensor data feeder"""
        return {
            'type': 'sensor_data',
            'sources': ['iot_sensors', 'satellite_data', 'weather_stations'],
            'data_types': ['temperature', 'humidity', 'pressure', 'location'],
            'frequency': 10.0,  # 10 Hz
            'description': 'Sensor data feeder'
        }
    
    def _create_user_behavior_feeder(self) -> Dict[str, Any]:
        """Create user behavior feeder"""
        return {
            'type': 'user_behavior',
            'sources': ['web_analytics', 'mobile_apps', 'user_interactions'],
            'data_types': ['clicks', 'views', 'searches', 'purchases'],
            'frequency': 1.0,  # 1 Hz
            'description': 'User behavior feeder'
        }
    
    async def start_feed_service(self):
        """Start the real-time feed service"""
        try:
            logger.info("Starting Real-Time Feed Service...")
            
            self.status = FeedStatus.IDLE
            
            # Start background tasks
            asyncio.create_task(self._feed_monitoring_loop())
            asyncio.create_task(self._data_processing_loop())
            
            logger.info("Real-Time Feed Service started successfully")
            
        except Exception as e:
            logger.error(f"Error starting feed service: {e}")
            self.status = FeedStatus.ERROR
            raise
    
    async def stop_feed_service(self):
        """Stop the real-time feed service"""
        try:
            logger.info("Stopping Real-Time Feed Service...")
            
            self.status = FeedStatus.IDLE
            
            # Stop all active streams
            for stream in self.data_streams.values():
                await self._stop_stream(stream)
            
            # Shutdown thread pool
            self.thread_pool.shutdown(wait=True)
            
            logger.info("Real-Time Feed Service stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping feed service: {e}")
            raise
    
    async def create_stream(self, source: Union[DataSource, str], data_type: str, 
                          frequency: float = 1.0, buffer_size: int = 1000) -> str:
        """Create a new data stream"""
        try:
            stream_id = str(uuid.uuid4())
            
            # Handle both enum and string inputs
            if isinstance(source, str):
                # Convert string to enum
                try:
                    source_enum = DataSource(source)
                except ValueError:
                    # Default to MARKET_DATA if string doesn't match
                    source_enum = DataSource.MARKET_DATA
            else:
                source_enum = source
            
            # Create data stream
            stream = DataStream(
                stream_id=stream_id,
                source=source_enum,
                data_type=data_type,
                frequency=frequency,
                buffer_size=buffer_size,
                is_active=False
            )
            
            # Store stream
            self.data_streams[stream_id] = stream
            
            # Initialize data buffer
            self.data_buffer[stream_id] = []
            
            self.metrics.total_streams += 1
            
            logger.info(f"Data stream created: {stream_id}")
            return stream_id
            
        except Exception as e:
            logger.error(f"Error creating stream: {e}")
            raise
    
    async def start_stream(self, stream_id: str) -> bool:
        """Start a data stream"""
        try:
            if stream_id not in self.data_streams:
                raise ValueError(f"Stream {stream_id} not found")
            
            stream = self.data_streams[stream_id]
            
            # Start streaming
            stream.is_active = True
            self.metrics.active_streams += 1
            
            # Start background streaming task
            asyncio.create_task(self._stream_data(stream))
            
            logger.info(f"Data stream started: {stream_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting stream: {e}")
            return False
    
    async def stop_stream(self, stream_id: str) -> bool:
        """Stop a data stream"""
        try:
            if stream_id not in self.data_streams:
                raise ValueError(f"Stream {stream_id} not found")
            
            stream = self.data_streams[stream_id]
            await self._stop_stream(stream)
            
            logger.info(f"Data stream stopped: {stream_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping stream: {e}")
            return False
    
    async def _stop_stream(self, stream: DataStream):
        """Stop a data stream"""
        try:
            stream.is_active = False
            if self.metrics.active_streams > 0:
                self.metrics.active_streams -= 1
                
        except Exception as e:
            logger.error(f"Error stopping stream: {e}")
    
    async def _stream_data(self, stream: DataStream):
        """Stream data for a specific stream"""
        try:
            while stream.is_active:
                # Generate data based on source
                data = await self._generate_data(stream)
                
                # Process data
                processed_data = await self._process_stream_data(data, stream)
                
                # Store in buffer
                self.data_buffer[stream.stream_id].append(processed_data)
                
                # Keep buffer size
                if len(self.data_buffer[stream.stream_id]) > stream.buffer_size:
                    self.data_buffer[stream.stream_id] = self.data_buffer[stream.stream_id][-stream.buffer_size:]
                
                # Update metrics
                self.metrics.total_data_points += 1
                
                # Wait for next data point
                await asyncio.sleep(1.0 / stream.frequency)
                
        except Exception as e:
            logger.error(f"Error streaming data for {stream.stream_id}: {e}")
            stream.is_active = False
    
    async def _generate_data(self, stream: DataStream) -> StreamData:
        """Generate data for a stream"""
        try:
            # Create stream data
            source_value = stream.source.value if hasattr(stream.source, 'value') else str(stream.source)
            data = StreamData(
                data_id=str(uuid.uuid4()),
                stream_id=stream.stream_id,
                timestamp=datetime.now(),
                data={},
                quality_score=1.0,
                is_real_time=True,
                source=source_value,
                metadata={'data_type': stream.data_type}
            )
            
            # Generate data based on source
            source_value = stream.source.value if hasattr(stream.source, 'value') else str(stream.source)
            if stream.source == DataSource.MARKET_DATA or source_value == 'market_data':
                data.data = await self._generate_market_data()
            elif stream.source == DataSource.NEWS_FEED or source_value == 'news_feed':
                data.data = await self._generate_news_data()
            elif stream.source == DataSource.SOCIAL_MEDIA or source_value == 'social_media':
                data.data = await self._generate_social_media_data()
            elif stream.source == DataSource.ECONOMIC_INDICATORS or source_value == 'economic_indicators':
                data.data = await self._generate_economic_data()
            elif stream.source == DataSource.COMPANY_FILINGS or source_value == 'company_filings':
                data.data = await self._generate_filings_data()
            elif stream.source == DataSource.ANALYST_REPORTS or source_value == 'analyst_reports':
                data.data = await self._generate_analyst_data()
            elif stream.source == DataSource.SENSOR_DATA or source_value == 'sensor_data':
                data.data = await self._generate_sensor_data()
            elif stream.source == DataSource.USER_BEHAVIOR or source_value == 'user_behavior':
                data.data = await self._generate_user_behavior_data()
            else:
                # Default to market data
                data.data = await self._generate_market_data()
            
            return data
            
        except Exception as e:
            logger.error(f"Error generating data: {e}")
            return StreamData(
                data_id=str(uuid.uuid4()),
                stream_id=stream.stream_id,
                timestamp=datetime.now(),
                data={},
                quality_score=0.0,
                is_real_time=True,
                source=stream.source.value
            )
    
    async def _generate_market_data(self) -> Dict[str, Any]:
        """Generate market data"""
        try:
            # Simulate market data
            await asyncio.sleep(0.001)
            
            return {
                'symbol': 'AAPL',
                'price': 150.0 + np.random.normal(0, 2.0),
                'volume': int(1000000 + np.random.normal(0, 100000)),
                'open': 150.0,
                'high': 152.0,
                'low': 148.0,
                'close': 151.0,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating market data: {e}")
            return {}
    
    async def _generate_news_data(self) -> Dict[str, Any]:
        """Generate news data"""
        try:
            # Simulate news data
            await asyncio.sleep(0.01)
            
            headlines = [
                "Apple reports strong Q4 earnings",
                "Market volatility increases",
                "Federal Reserve announces rate decision",
                "Tech stocks rally on positive outlook"
            ]
            
            return {
                'headline': np.random.choice(headlines),
                'source': 'Reuters',
                'sentiment': np.random.uniform(-1, 1),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating news data: {e}")
            return {}
    
    async def _generate_social_media_data(self) -> Dict[str, Any]:
        """Generate social media data"""
        try:
            # Simulate social media data
            await asyncio.sleep(0.01)
            
            return {
                'platform': 'Twitter',
                'post': 'Great earnings report from $AAPL!',
                'sentiment': np.random.uniform(-1, 1),
                'engagement': int(np.random.uniform(0, 1000)),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating social media data: {e}")
            return {}
    
    async def _generate_economic_data(self) -> Dict[str, Any]:
        """Generate economic data"""
        try:
            # Simulate economic data
            await asyncio.sleep(0.01)
            
            return {
                'indicator': 'GDP',
                'value': 2.5 + np.random.normal(0, 0.5),
                'unit': 'percent',
                'period': 'Q3 2024',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating economic data: {e}")
            return {}
    
    async def _generate_filings_data(self) -> Dict[str, Any]:
        """Generate filings data"""
        try:
            # Simulate filings data
            await asyncio.sleep(0.01)
            
            return {
                'company': 'Apple Inc.',
                'filing_type': '10-K',
                'filing_date': datetime.now().isoformat(),
                'url': 'https://sec.gov/edgar/...',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating filings data: {e}")
            return {}
    
    async def _generate_analyst_data(self) -> Dict[str, Any]:
        """Generate analyst data"""
        try:
            # Simulate analyst data
            await asyncio.sleep(0.01)
            
            return {
                'analyst': 'Goldman Sachs',
                'rating': 'Buy',
                'target_price': 160.0 + np.random.normal(0, 10.0),
                'company': 'Apple Inc.',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating analyst data: {e}")
            return {}
    
    async def _generate_sensor_data(self) -> Dict[str, Any]:
        """Generate sensor data"""
        try:
            # Simulate sensor data
            await asyncio.sleep(0.001)
            
            return {
                'sensor_id': 'TEMP_001',
                'temperature': 22.0 + np.random.normal(0, 2.0),
                'humidity': 50.0 + np.random.normal(0, 5.0),
                'pressure': 1013.25 + np.random.normal(0, 10.0),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating sensor data: {e}")
            return {}
    
    async def _generate_user_behavior_data(self) -> Dict[str, Any]:
        """Generate user behavior data"""
        try:
            # Simulate user behavior data
            await asyncio.sleep(0.01)
            
            return {
                'user_id': str(uuid.uuid4()),
                'action': 'view_stock',
                'symbol': 'AAPL',
                'duration': np.random.uniform(1, 60),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating user behavior data: {e}")
            return {}
    
    async def _process_stream_data(self, data: StreamData, stream: DataStream) -> StreamData:
        """Process stream data"""
        try:
            # Validate data
            data = await self._validate_data(data)
            
            # Clean data
            data = await self._clean_data(data)
            
            # Enhance data
            data = await self._enhance_data(data)
            
            # Calculate quality score
            data.quality_score = self._calculate_quality_score(data)
            
            return data
            
        except Exception as e:
            logger.error(f"Error processing stream data: {e}")
            return data
    
    async def _validate_data(self, data: StreamData) -> StreamData:
        """Validate stream data"""
        try:
            # Simulate data validation
            await asyncio.sleep(0.001)
            
            # Check data completeness
            if not data.data:
                data.quality_score = 0.0
            else:
                data.quality_score = min(1.0, data.quality_score + 0.1)
            
            return data
            
        except Exception as e:
            logger.error(f"Error validating data: {e}")
            return data
    
    async def _clean_data(self, data: StreamData) -> StreamData:
        """Clean stream data"""
        try:
            # Simulate data cleaning
            await asyncio.sleep(0.001)
            
            # Remove outliers (simplified)
            if 'price' in data.data:
                price = data.data['price']
                if price < 0 or price > 10000:  # Simple outlier detection
                    data.data['price'] = 150.0  # Default price
                    data.quality_score = max(0.0, data.quality_score - 0.1)
            
            return data
            
        except Exception as e:
            logger.error(f"Error cleaning data: {e}")
            return data
    
    async def _enhance_data(self, data: StreamData) -> StreamData:
        """Enhance stream data"""
        try:
            # Simulate data enhancement
            await asyncio.sleep(0.001)
            
            # Add derived features
            if 'price' in data.data and 'volume' in data.data:
                price = data.data['price']
                volume = data.data['volume']
                data.data['market_cap'] = price * volume
                data.data['price_volume_ratio'] = price / volume if volume > 0 else 0
            
            # Improve quality score
            data.quality_score = min(1.0, data.quality_score + 0.05)
            
            return data
            
        except Exception as e:
            logger.error(f"Error enhancing data: {e}")
            return data
    
    def _calculate_quality_score(self, data: StreamData) -> float:
        """Calculate data quality score"""
        try:
            score = 0.0
            
            # Check data completeness
            if data.data:
                score += 0.3
            
            # Check timestamp
            if data.timestamp:
                score += 0.2
            
            # Check real-time flag
            if data.is_real_time:
                score += 0.2
            
            # Check data validity
            if 'price' in data.data and data.data['price'] > 0:
                score += 0.3
            
            return min(1.0, score)
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 0.0
    
    async def subscribe_to_stream(self, stream_id: str, callback: Callable) -> str:
        """Subscribe to a data stream"""
        try:
            if stream_id not in self.data_streams:
                raise ValueError(f"Stream {stream_id} not found")
            
            subscriber_id = str(uuid.uuid4())
            
            # Store subscriber
            if stream_id not in self.subscribers:
                self.subscribers[stream_id] = {}
            
            self.subscribers[stream_id][subscriber_id] = callback
            
            # Add to stream subscribers
            self.data_streams[stream_id].subscribers.append(subscriber_id)
            
            logger.info(f"Subscribed to stream {stream_id}: {subscriber_id}")
            return subscriber_id
            
        except Exception as e:
            logger.error(f"Error subscribing to stream: {e}")
            raise
    
    async def unsubscribe_from_stream(self, stream_id: str, subscriber_id: str) -> bool:
        """Unsubscribe from a data stream"""
        try:
            if stream_id in self.subscribers and subscriber_id in self.subscribers[stream_id]:
                del self.subscribers[stream_id][subscriber_id]
                
                # Remove from stream subscribers
                if stream_id in self.data_streams:
                    if subscriber_id in self.data_streams[stream_id].subscribers:
                        self.data_streams[stream_id].subscribers.remove(subscriber_id)
                
                logger.info(f"Unsubscribed from stream {stream_id}: {subscriber_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error unsubscribing from stream: {e}")
            return False
    
    async def get_stream_data(self, stream_id: str, limit: int = 100) -> List[StreamData]:
        """Get data from a stream"""
        try:
            if stream_id not in self.data_buffer:
                return []
            
            # Get latest data
            data = self.data_buffer[stream_id][-limit:]
            return data
            
        except Exception as e:
            logger.error(f"Error getting stream data: {e}")
            return []
    
    async def _feed_monitoring_loop(self):
        """Monitor feed performance"""
        try:
            while self.status in [FeedStatus.IDLE, FeedStatus.STREAMING]:
                await asyncio.sleep(30)
                
                # Update performance metrics
                self._update_performance_metrics()
                
        except Exception as e:
            logger.error(f"Error in feed monitoring loop: {e}")
    
    async def _data_processing_loop(self):
        """Process incoming data"""
        try:
            while self.status in [FeedStatus.IDLE, FeedStatus.STREAMING]:
                await asyncio.sleep(1)
                
                # Process data for all active streams
                for stream_id, stream in self.data_streams.items():
                    if stream.is_active and stream_id in self.data_buffer:
                        # Get latest data
                        if self.data_buffer[stream_id]:
                            latest_data = self.data_buffer[stream_id][-1]
                            
                            # Notify subscribers
                            if stream_id in self.subscribers:
                                for subscriber_id, callback in self.subscribers[stream_id].items():
                                    try:
                                        await callback(latest_data)
                                    except Exception as e:
                                        logger.error(f"Error in subscriber callback: {e}")
                
        except Exception as e:
            logger.error(f"Error in data processing loop: {e}")
    
    def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            # Calculate data quality score
            if self.data_buffer:
                quality_scores = []
                for stream_data in self.data_buffer.values():
                    if stream_data:
                        scores = [data.quality_score for data in stream_data]
                        quality_scores.extend(scores)
                
                if quality_scores:
                    self.metrics.data_quality_score = np.mean(quality_scores)
            
            # Calculate throughput
            self.metrics.average_throughput = self.metrics.total_data_points / 60  # Per minute
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    async def get_feed_status(self) -> Dict[str, Any]:
        """Get feed service status"""
        return {
            'status': self.status.value,
            'total_streams': self.metrics.total_streams,
            'active_streams': self.metrics.active_streams,
            'total_data_points': self.metrics.total_data_points,
            'average_latency': self.metrics.average_latency,
            'average_throughput': self.metrics.average_throughput,
            'data_quality_score': self.metrics.data_quality_score,
            'error_rate': self.metrics.error_rate,
            'uptime': self.metrics.uptime,
            'processing_efficiency': self.metrics.processing_efficiency,
            'available_sources': list(self.feed_components.keys()),
            'streaming_available': STREAMING_AVAILABLE,
            'ai_available': AI_AVAILABLE,
            'timestamp': datetime.now().isoformat()
        }
    
    async def get_feed_results(self, result_id: str) -> Optional[FeedResult]:
        """Get feed result by ID"""
        return self.feed_results.get(result_id)
    
    async def add_batch_operation(self, operation_type: str, data: Any, 
                                 priority: int = 0, callback: Optional[Callable] = None) -> str:
        """Add operation to batch processing queue"""
        try:
            if not CONNECTION_POOL_AVAILABLE:
                logger.warning("Connection pooling not available for batch operations")
                return None
            
            operation_id = str(uuid.uuid4())
            
            # Create batch operation
            batch_op = BatchOperation(
                operation_id=operation_id,
                operation_type=operation_type,
                data=data,
                callback=callback,
                priority=priority
            )
            
            # Add to batch queue
            self.batch_operations.append(batch_op)
            
            # Add to appropriate connection pool
            if operation_type in ['http', 'api']:
                pool = self.connection_pools.get('http')
            elif operation_type in ['database', 'db', 'sql']:
                pool = self.connection_pools.get('database')
            elif operation_type in ['cache', 'redis']:
                pool = self.connection_pools.get('redis')
            else:
                pool = self.connection_pools.get('http')  # Default to HTTP
            
            if pool:
                pool.add_batch_operation(batch_op)
            
            logger.debug(f"Batch operation added: {operation_id}")
            return operation_id
            
        except Exception as e:
            logger.error(f"Error adding batch operation: {e}")
            return None
    
    async def process_batch_operations(self, batch_size: int = 100) -> Dict[str, Any]:
        """Process batch operations using connection pools"""
        try:
            if not CONNECTION_POOL_AVAILABLE:
                return {'error': 'Connection pooling not available'}
            
            results = {
                'processed_operations': 0,
                'successful_operations': 0,
                'failed_operations': 0,
                'pool_metrics': {}
            }
            
            # Process operations from each pool
            for pool_name, pool in self.connection_pools.items():
                try:
                    # Get pool metrics
                    pool_metrics = pool.get_metrics()
                    results['pool_metrics'][pool_name] = {
                        'total_connections': pool_metrics.total_connections,
                        'active_connections': pool_metrics.active_connections,
                        'idle_connections': pool_metrics.idle_connections,
                        'pool_utilization': pool_metrics.pool_utilization,
                        'average_response_time': pool_metrics.average_response_time,
                        'successful_requests': pool_metrics.successful_requests,
                        'failed_requests': pool_metrics.failed_requests
                    }
                    
                    # Process batch operations
                    batch_ops = self.batch_operations[:batch_size]
                    if batch_ops:
                        # Remove processed operations
                        self.batch_operations = self.batch_operations[batch_size:]
                        
                        # Execute operations
                        for op in batch_ops:
                            try:
                                result = await pool.execute_operation(
                                    operation=op.operation_type,
                                    data=op.data
                                )
                                
                                if result and result.get('status') == 'success':
                                    results['successful_operations'] += 1
                                else:
                                    results['failed_operations'] += 1
                                
                                # Call callback if provided
                                if op.callback:
                                    try:
                                        op.callback(result)
                                    except Exception as e:
                                        logger.warning(f"Callback failed: {e}")
                                
                            except Exception as e:
                                logger.error(f"Error executing batch operation: {e}")
                                results['failed_operations'] += 1
                        
                        results['processed_operations'] += len(batch_ops)
                        
                except Exception as e:
                    logger.error(f"Error processing batch for pool {pool_name}: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing batch operations: {e}")
            return {'error': str(e)}
    
    async def get_connection_pool_metrics(self) -> Dict[str, Any]:
        """Get comprehensive connection pool metrics"""
        try:
            if not CONNECTION_POOL_AVAILABLE:
                return {'error': 'Connection pooling not available'}
            
            metrics = {
                'total_pools': len(self.connection_pools),
                'pools': {},
                'overall_metrics': {
                    'total_connections': 0,
                    'active_connections': 0,
                    'idle_connections': 0,
                    'total_requests': 0,
                    'successful_requests': 0,
                    'failed_requests': 0,
                    'average_response_time': 0.0
                }
            }
            
            # Get metrics from each pool
            for pool_name, pool in self.connection_pools.items():
                pool_metrics = pool.get_metrics()
                metrics['pools'][pool_name] = {
                    'total_connections': pool_metrics.total_connections,
                    'active_connections': pool_metrics.active_connections,
                    'idle_connections': pool_metrics.idle_connections,
                    'failed_connections': pool_metrics.failed_connections,
                    'total_requests': pool_metrics.total_requests,
                    'successful_requests': pool_metrics.successful_requests,
                    'failed_requests': pool_metrics.failed_requests,
                    'average_response_time': pool_metrics.average_response_time,
                    'pool_utilization': pool_metrics.pool_utilization,
                    'batch_operations_processed': pool_metrics.batch_operations_processed,
                    'batch_throughput': pool_metrics.batch_throughput
                }
                
                # Aggregate overall metrics
                metrics['overall_metrics']['total_connections'] += pool_metrics.total_connections
                metrics['overall_metrics']['active_connections'] += pool_metrics.active_connections
                metrics['overall_metrics']['idle_connections'] += pool_metrics.idle_connections
                metrics['overall_metrics']['total_requests'] += pool_metrics.total_requests
                metrics['overall_metrics']['successful_requests'] += pool_metrics.successful_requests
                metrics['overall_metrics']['failed_requests'] += pool_metrics.failed_requests
            
            # Calculate average response time
            if metrics['overall_metrics']['total_requests'] > 0:
                total_response_time = sum(
                    pool.get_metrics().average_response_time * pool.get_metrics().total_requests
                    for pool in self.connection_pools.values()
                )
                metrics['overall_metrics']['average_response_time'] = (
                    total_response_time / metrics['overall_metrics']['total_requests']
                )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting connection pool metrics: {e}")
            return {'error': str(e)}

# Global instance
real_time_feeder = RealTimeFeeder()
