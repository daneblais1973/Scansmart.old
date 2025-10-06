"""
Production Configuration for Multi-Modal Processor
================================================
Enterprise-grade configuration management for multi-modal catalyst processing.
"""

import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from datetime import timedelta

class TextProcessingConfig(BaseModel):
    """Text processing configuration"""
    enabled: bool = Field(True, description="Enable text processing")
    model_name: str = Field("all-MiniLM-L6-v2", description="Sentence transformer model name")
    max_length: int = Field(512, description="Maximum text length", ge=1, le=2048)
    batch_size: int = Field(32, description="Processing batch size", ge=1, le=256)
    enable_sentiment: bool = Field(True, description="Enable sentiment analysis")
    enable_entities: bool = Field(True, description="Enable entity extraction")
    enable_keywords: bool = Field(True, description="Enable keyword extraction")

class ImageProcessingConfig(BaseModel):
    """Image processing configuration"""
    enabled: bool = Field(True, description="Enable image processing")
    max_size: tuple = Field((224, 224), description="Maximum image size")
    channels: int = Field(3, description="Number of color channels", ge=1, le=4)
    enable_ocr: bool = Field(True, description="Enable OCR text extraction")
    enable_objects: bool = Field(True, description="Enable object detection")
    enable_texture: bool = Field(True, description="Enable texture analysis")

class AudioProcessingConfig(BaseModel):
    """Audio processing configuration"""
    enabled: bool = Field(True, description="Enable audio processing")
    sample_rate: int = Field(22050, description="Audio sample rate", ge=8000, le=48000)
    max_duration: float = Field(300.0, description="Maximum audio duration in seconds", ge=1.0, le=3600.0)
    enable_speech: bool = Field(True, description="Enable speech-to-text")
    enable_emotion: bool = Field(True, description="Enable emotion analysis")
    mfcc_coefficients: int = Field(13, description="Number of MFCC coefficients", ge=5, le=40)

class VideoProcessingConfig(BaseModel):
    """Video processing configuration"""
    enabled: bool = Field(True, description="Enable video processing")
    max_duration: float = Field(600.0, description="Maximum video duration in seconds", ge=1.0, le=7200.0)
    frame_rate: float = Field(30.0, description="Video frame rate", ge=1.0, le=120.0)
    max_resolution: tuple = Field((1920, 1080), description="Maximum video resolution")
    enable_motion: bool = Field(True, description="Enable motion analysis")
    enable_audio: bool = Field(True, description="Enable audio extraction from video")

class StructuredDataConfig(BaseModel):
    """Structured data processing configuration"""
    enabled: bool = Field(True, description="Enable structured data processing")
    max_features: int = Field(100, description="Maximum number of features", ge=1, le=1000)
    enable_statistics: bool = Field(True, description="Enable statistical analysis")
    enable_correlation: bool = Field(True, description="Enable correlation analysis")
    categorical_encoding: str = Field("one_hot", description="Categorical encoding method", pattern=r'^(one_hot|label|target)$')

class TimeSeriesConfig(BaseModel):
    """Time series processing configuration"""
    enabled: bool = Field(True, description="Enable time series processing")
    max_length: int = Field(1000, description="Maximum time series length", ge=10, le=10000)
    enable_trend: bool = Field(True, description="Enable trend analysis")
    enable_seasonality: bool = Field(True, description="Enable seasonality analysis")
    enable_autocorrelation: bool = Field(True, description="Enable autocorrelation analysis")

class FusionConfig(BaseModel):
    """Multi-modal fusion configuration"""
    default_method: str = Field("attention", description="Default fusion method", pattern=r'^(early|late|attention|cross_modal)$')
    enable_early_fusion: bool = Field(True, description="Enable early fusion")
    enable_late_fusion: bool = Field(True, description="Enable late fusion")
    enable_attention_fusion: bool = Field(True, description="Enable attention fusion")
    enable_cross_modal_fusion: bool = Field(True, description="Enable cross-modal fusion")
    attention_heads: int = Field(8, description="Number of attention heads", ge=1, le=16)
    hidden_size: int = Field(256, description="Hidden layer size", ge=64, le=1024)

class PerformanceConfig(BaseModel):
    """Performance monitoring configuration"""
    metrics_enabled: bool = Field(True, description="Enable performance metrics")
    processing_timeout: float = Field(60.0, description="Processing timeout in seconds", ge=1.0, le=600.0)
    memory_limit_mb: int = Field(4096, description="Memory limit in MB", ge=512, le=32768)
    cpu_limit_percent: float = Field(80.0, description="CPU usage limit", ge=10.0, le=100.0)
    enable_caching: bool = Field(True, description="Enable feature caching")
    cache_size_mb: int = Field(1024, description="Cache size in MB", ge=64, le=8192)

class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: str = Field("INFO", description="Logging level", pattern=r'^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$')
    structured: bool = Field(True, description="Enable structured logging")
    include_metrics: bool = Field(True, description="Include metrics in logs")
    log_processing: bool = Field(False, description="Log processing details (privacy sensitive)")

class MultiModalProcessorConfig(BaseModel):
    """Main multi-modal processor configuration"""
    text: TextProcessingConfig = Field(default_factory=TextProcessingConfig)
    image: ImageProcessingConfig = Field(default_factory=ImageProcessingConfig)
    audio: AudioProcessingConfig = Field(default_factory=AudioProcessingConfig)
    video: VideoProcessingConfig = Field(default_factory=VideoProcessingConfig)
    structured: StructuredDataConfig = Field(default_factory=StructuredDataConfig)
    time_series: TimeSeriesConfig = Field(default_factory=TimeSeriesConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    @validator('text')
    def validate_text_config(cls, v):
        """Validate text processing configuration"""
        if v.max_length <= 0:
            raise ValueError("Maximum text length must be positive")
        if v.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        return v
    
    @validator('image')
    def validate_image_config(cls, v):
        """Validate image processing configuration"""
        if v.max_size[0] <= 0 or v.max_size[1] <= 0:
            raise ValueError("Maximum image size must be positive")
        if v.channels <= 0:
            raise ValueError("Number of channels must be positive")
        return v
    
    @validator('audio')
    def validate_audio_config(cls, v):
        """Validate audio processing configuration"""
        if v.sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        if v.max_duration <= 0:
            raise ValueError("Maximum duration must be positive")
        return v
    
    @validator('video')
    def validate_video_config(cls, v):
        """Validate video processing configuration"""
        if v.max_duration <= 0:
            raise ValueError("Maximum duration must be positive")
        if v.frame_rate <= 0:
            raise ValueError("Frame rate must be positive")
        return v
    
    @validator('fusion')
    def validate_fusion_config(cls, v):
        """Validate fusion configuration"""
        if v.attention_heads <= 0:
            raise ValueError("Number of attention heads must be positive")
        if v.hidden_size <= 0:
            raise ValueError("Hidden size must be positive")
        return v

def load_multi_modal_processor_config() -> MultiModalProcessorConfig:
    """Load multi-modal processor configuration from environment variables"""
    return MultiModalProcessorConfig(
        text=TextProcessingConfig(
            enabled=os.getenv("TEXT_PROCESSING_ENABLED", "true").lower() == "true",
            model_name=os.getenv("TEXT_MODEL_NAME", "all-MiniLM-L6-v2"),
            max_length=int(os.getenv("TEXT_MAX_LENGTH", "512")),
            batch_size=int(os.getenv("TEXT_BATCH_SIZE", "32")),
            enable_sentiment=os.getenv("TEXT_ENABLE_SENTIMENT", "true").lower() == "true",
            enable_entities=os.getenv("TEXT_ENABLE_ENTITIES", "true").lower() == "true",
            enable_keywords=os.getenv("TEXT_ENABLE_KEYWORDS", "true").lower() == "true"
        ),
        image=ImageProcessingConfig(
            enabled=os.getenv("IMAGE_PROCESSING_ENABLED", "true").lower() == "true",
            max_size=tuple(map(int, os.getenv("IMAGE_MAX_SIZE", "224,224").split(','))),
            channels=int(os.getenv("IMAGE_CHANNELS", "3")),
            enable_ocr=os.getenv("IMAGE_ENABLE_OCR", "true").lower() == "true",
            enable_objects=os.getenv("IMAGE_ENABLE_OBJECTS", "true").lower() == "true",
            enable_texture=os.getenv("IMAGE_ENABLE_TEXTURE", "true").lower() == "true"
        ),
        audio=AudioProcessingConfig(
            enabled=os.getenv("AUDIO_PROCESSING_ENABLED", "true").lower() == "true",
            sample_rate=int(os.getenv("AUDIO_SAMPLE_RATE", "22050")),
            max_duration=float(os.getenv("AUDIO_MAX_DURATION", "300.0")),
            enable_speech=os.getenv("AUDIO_ENABLE_SPEECH", "true").lower() == "true",
            enable_emotion=os.getenv("AUDIO_ENABLE_EMOTION", "true").lower() == "true",
            mfcc_coefficients=int(os.getenv("AUDIO_MFCC_COEFFICIENTS", "13"))
        ),
        video=VideoProcessingConfig(
            enabled=os.getenv("VIDEO_PROCESSING_ENABLED", "true").lower() == "true",
            max_duration=float(os.getenv("VIDEO_MAX_DURATION", "600.0")),
            frame_rate=float(os.getenv("VIDEO_FRAME_RATE", "30.0")),
            max_resolution=tuple(map(int, os.getenv("VIDEO_MAX_RESOLUTION", "1920,1080").split(','))),
            enable_motion=os.getenv("VIDEO_ENABLE_MOTION", "true").lower() == "true",
            enable_audio=os.getenv("VIDEO_ENABLE_AUDIO", "true").lower() == "true"
        ),
        structured=StructuredDataConfig(
            enabled=os.getenv("STRUCTURED_PROCESSING_ENABLED", "true").lower() == "true",
            max_features=int(os.getenv("STRUCTURED_MAX_FEATURES", "100")),
            enable_statistics=os.getenv("STRUCTURED_ENABLE_STATISTICS", "true").lower() == "true",
            enable_correlation=os.getenv("STRUCTURED_ENABLE_CORRELATION", "true").lower() == "true",
            categorical_encoding=os.getenv("STRUCTURED_CATEGORICAL_ENCODING", "one_hot")
        ),
        time_series=TimeSeriesConfig(
            enabled=os.getenv("TIME_SERIES_PROCESSING_ENABLED", "true").lower() == "true",
            max_length=int(os.getenv("TIME_SERIES_MAX_LENGTH", "1000")),
            enable_trend=os.getenv("TIME_SERIES_ENABLE_TREND", "true").lower() == "true",
            enable_seasonality=os.getenv("TIME_SERIES_ENABLE_SEASONALITY", "true").lower() == "true",
            enable_autocorrelation=os.getenv("TIME_SERIES_ENABLE_AUTOCORRELATION", "true").lower() == "true"
        ),
        fusion=FusionConfig(
            default_method=os.getenv("FUSION_DEFAULT_METHOD", "attention"),
            enable_early_fusion=os.getenv("FUSION_ENABLE_EARLY", "true").lower() == "true",
            enable_late_fusion=os.getenv("FUSION_ENABLE_LATE", "true").lower() == "true",
            enable_attention_fusion=os.getenv("FUSION_ENABLE_ATTENTION", "true").lower() == "true",
            enable_cross_modal_fusion=os.getenv("FUSION_ENABLE_CROSS_MODAL", "true").lower() == "true",
            attention_heads=int(os.getenv("FUSION_ATTENTION_HEADS", "8")),
            hidden_size=int(os.getenv("FUSION_HIDDEN_SIZE", "256"))
        ),
        performance=PerformanceConfig(
            metrics_enabled=os.getenv("PERFORMANCE_METRICS_ENABLED", "true").lower() == "true",
            processing_timeout=float(os.getenv("PERFORMANCE_PROCESSING_TIMEOUT", "60.0")),
            memory_limit_mb=int(os.getenv("PERFORMANCE_MEMORY_LIMIT_MB", "4096")),
            cpu_limit_percent=float(os.getenv("PERFORMANCE_CPU_LIMIT_PERCENT", "80.0")),
            enable_caching=os.getenv("PERFORMANCE_ENABLE_CACHING", "true").lower() == "true",
            cache_size_mb=int(os.getenv("PERFORMANCE_CACHE_SIZE_MB", "1024"))
        ),
        logging=LoggingConfig(
            level=os.getenv("MULTI_MODAL_LOG_LEVEL", "INFO"),
            structured=os.getenv("MULTI_MODAL_STRUCTURED_LOGGING", "true").lower() == "true",
            include_metrics=os.getenv("MULTI_MODAL_INCLUDE_METRICS", "true").lower() == "true",
            log_processing=os.getenv("MULTI_MODAL_LOG_PROCESSING", "false").lower() == "true"
        )
    )

def validate_multi_modal_processor_config(config: MultiModalProcessorConfig) -> bool:
    """Validate multi-modal processor configuration settings"""
    try:
        # Validate text processing
        if config.text.max_length <= 0:
            print("Error: Text max length must be positive")
            return False
        
        # Validate image processing
        if config.image.max_size[0] <= 0 or config.image.max_size[1] <= 0:
            print("Error: Image max size must be positive")
            return False
        
        # Validate audio processing
        if config.audio.sample_rate <= 0:
            print("Error: Audio sample rate must be positive")
            return False
        
        # Validate video processing
        if config.video.max_duration <= 0:
            print("Error: Video max duration must be positive")
            return False
        
        # Validate fusion configuration
        if config.fusion.attention_heads <= 0:
            print("Error: Attention heads must be positive")
            return False
        
        # Validate performance configuration
        if config.performance.processing_timeout <= 0:
            print("Error: Processing timeout must be positive")
            return False
        
        if config.performance.memory_limit_mb <= 0:
            print("Error: Memory limit must be positive")
            return False
        
        # Validate logging configuration
        if config.logging.log_processing:
            print("Warning: Processing logging is enabled - may contain sensitive data")
        
        return True
        
    except Exception as e:
        print(f"Multi-modal processor configuration validation error: {e}")
        return False

# Global configuration instance
multi_modal_processor_config = load_multi_modal_processor_config()

def get_multi_modal_processor_config() -> MultiModalProcessorConfig:
    """Get the global multi-modal processor configuration instance"""
    return multi_modal_processor_config

def reload_multi_modal_processor_config() -> MultiModalProcessorConfig:
    """Reload multi-modal processor configuration from environment"""
    global multi_modal_processor_config
    multi_modal_processor_config = load_multi_modal_processor_config()
    return multi_modal_processor_config


