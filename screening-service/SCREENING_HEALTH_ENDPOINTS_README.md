# Production-Ready Health Endpoints for Screening Service

## Overview
Enterprise-grade health monitoring for the screening service with real service integration. **NO MOCK DATA** - All health checks use real service components and system metrics with professional error handling and monitoring.

## Features
- ✅ **Real Screening Service Integration** - Actual health checks of screening service components
- ✅ **System Resource Monitoring** - CPU, memory, disk usage monitoring
- ✅ **Dependency Health Checks** - External dependency status monitoring
- ✅ **Kubernetes-Style Probes** - Readiness and liveness checks
- ✅ **Professional Error Handling** - Comprehensive error management
- ✅ **Performance Monitoring** - Health check response time tracking
- ✅ **Smart Caching** - Configurable health check caching
- ✅ **Structured Logging** - Enterprise-grade logging with metrics

## Health Endpoints

### Main Health Check
```
GET /health/
```
Comprehensive health check for the screening service with system metrics.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-10-03T10:00:00Z",
  "service": "screening-service",
  "version": "1.0.0",
  "uptime": 3600.5,
  "details": {
    "cpu_usage": 45.2,
    "memory_usage": 67.8,
    "memory_available": 2147483648,
    "disk_usage": 23.4,
    "disk_free": 500000000000,
    "process_count": 156,
    "load_average": [1.2, 1.5, 1.8]
  },
  "dependencies": {
    "python": "3.12.0",
    "psutil": "available",
    "fastapi": "available"
  }
}
```

### Component Health Checks

#### Multi-Model Ranker Health
```
GET /health/multi-model-ranker
```
Health check for multi-model ranker component.

**Response:**
```json
{
  "status": "healthy",
  "component": "multi-model-ranker",
  "details": {
    "ranking_status": "idle",
    "active_rankings": 0,
    "completed_rankings": 200,
    "failed_rankings": 5,
    "average_ranking_time": 1.5,
    "ranking_efficiency": 0.95,
    "ai_available": true,
    "available_models": ["model_1", "model_2", "model_3"],
    "uptime": 3600.5
  },
  "dependencies": {
    "torch": "available",
    "scikit-learn": "available",
    "transformers": "available",
    "numpy": "available",
    "pandas": "available"
  },
  "last_check": "2025-10-03T10:00:00Z",
  "response_time": 0.12
}
```

### Detailed Health Check
```
GET /health/detailed
```
Comprehensive health check with all components and system metrics.

**Response:**
```json
{
  "overall_status": "healthy",
  "timestamp": "2025-10-03T10:00:00Z",
  "components": {
    "multi_model_ranker": {
      "status": "healthy",
      "component": "multi-model-ranker",
      "details": {...},
      "dependencies": {...},
      "last_check": "2025-10-03T10:00:00Z",
      "response_time": 0.12
    }
  },
  "system_metrics": {
    "cpu_usage": 45.2,
    "memory_usage": 67.8,
    "memory_available": 2147483648,
    "disk_usage": 23.4,
    "disk_free": 500000000000,
    "process_count": 156,
    "load_average": [1.2, 1.5, 1.8]
  },
  "service_info": {
    "service": "screening-service",
    "version": "1.0.0",
    "uptime": 3600.5,
    "python_version": "3.12.0",
    "platform": "linux"
  }
}
```

### Kubernetes-Style Probes

#### Readiness Check
```
GET /health/ready
```
Kubernetes-style readiness probe for service dependencies.

**Response (Ready):**
```json
{
  "status": "ready",
  "timestamp": "2025-10-03T10:00:00Z",
  "uptime": 3600.5
}
```

**Response (Not Ready):**
```json
{
  "detail": "Service not ready - one or more components are unhealthy"
}
```

#### Liveness Check
```
GET /health/live
```
Kubernetes-style liveness probe for service availability.

**Response:**
```json
{
  "status": "alive",
  "timestamp": "2025-10-03T10:00:00Z",
  "uptime": 3600.5
}
```

### Health Metrics
```
GET /health/metrics
```
Get health check performance metrics.

**Response:**
```json
{
  "service": "screening-service",
  "timestamp": "2025-10-03T10:00:00Z",
  "uptime": 3600.5,
  "system_metrics": {
    "cpu_usage": 45.2,
    "memory_usage": 67.8,
    "memory_available": 2147483648,
    "disk_usage": 23.4,
    "disk_free": 500000000000,
    "process_count": 156,
    "load_average": [1.2, 1.5, 1.8]
  },
  "cache_stats": {
    "cached_items": 1,
    "cache_duration": 30,
    "cache_keys": ["multi_model_ranker_health"]
  },
  "health_checker": {
    "start_time": "2025-10-03T09:00:00Z",
    "cache_duration": 30
  }
}
```

## Configuration

### Environment Variables

#### Health Check Configuration
```bash
SCREENING_HEALTH_CACHE_DURATION=30
SCREENING_HEALTH_TIMEOUT=10
SCREENING_HEALTH_RETRY_COUNT=3
SCREENING_HEALTH_RETRY_DELAY=1.0
```

#### System Metrics Configuration
```bash
SCREENING_SYSTEM_METRICS_ENABLED=true
SCREENING_CPU_THRESHOLD=90.0
SCREENING_MEMORY_THRESHOLD=90.0
SCREENING_DISK_THRESHOLD=90.0
SCREENING_METRICS_COLLECTION_INTERVAL=5
```

#### Component Configuration
```bash
MULTI_MODEL_RANKER_HEALTH=true
SCREENING_PARALLEL_HEALTH_CHECKS=true
```

#### Logging Configuration
```bash
SCREENING_HEALTH_LOG_LEVEL=INFO
SCREENING_HEALTH_STRUCTURED_LOGGING=true
SCREENING_HEALTH_INCLUDE_METRICS=true
SCREENING_HEALTH_LOG_SENSITIVE_DATA=false
```

#### Alerting Configuration
```bash
SCREENING_HEALTH_ALERTING_ENABLED=false
SCREENING_HEALTH_WEBHOOK_URL=https://hooks.slack.com/services/...
SCREENING_HEALTH_ALERT_THRESHOLD=3
SCREENING_HEALTH_ALERT_COOLDOWN=300
```

## Health Status Levels

### Healthy
- All screening components operational
- System resources within normal limits
- Dependencies available
- Response times acceptable

### Degraded
- Some screening components experiencing issues
- System resources approaching limits
- Some dependencies unavailable
- Response times elevated

### Unhealthy
- Critical screening components failing
- System resources critically low
- Key dependencies unavailable
- Service unable to function

## System Metrics Thresholds

### CPU Usage
- **Normal**: < 70%
- **Warning**: 70-90%
- **Critical**: > 90%

### Memory Usage
- **Normal**: < 70%
- **Warning**: 70-90%
- **Critical**: > 90%

### Disk Usage
- **Normal**: < 70%
- **Warning**: 70-90%
- **Critical**: > 90%

## Caching Strategy

### Health Check Caching
- **Cache Duration**: 30 seconds (configurable)
- **Cache Keys**: Component-specific health checks
- **Cache Invalidation**: Automatic expiry
- **Cache Benefits**: Reduced load, faster responses

### Cache Configuration
```python
# Cache duration in seconds
SCREENING_HEALTH_CACHE_DURATION=30

# Cache keys
multi_model_ranker_health
```

## Error Handling

### Error Types
- **503 Service Unavailable**: Component health check failed
- **500 Internal Server Error**: Unexpected health check error
- **Timeout**: Health check exceeded timeout threshold
- **Dependency Error**: External dependency unavailable

### Error Response Format
```json
{
  "detail": "Health check failed: Component not available",
  "status_code": 503
}
```

## Performance

### Optimization Features
- **Parallel Health Checks** - Concurrent component checking
- **Smart Caching** - Configurable response caching
- **Timeout Management** - Configurable timeouts
- **Retry Logic** - Automatic retry for failed checks

### Benchmarks
- **Main Health Check**: < 200ms average response
- **Component Checks**: < 100ms average response
- **Detailed Health**: < 250ms average response
- **System Metrics**: < 50ms average response

## Monitoring

### Health Check Metrics
- Request count per endpoint
- Response time averages
- Success/failure rates
- Component availability

### System Metrics
- CPU usage percentage
- Memory usage percentage
- Disk usage percentage
- Process count
- Load average

### Logging
- **Structured Logging** - JSON-formatted logs
- **Health Check Logging** - Complete health check tracking
- **Error Tracking** - Comprehensive error logging
- **Performance Metrics** - Built-in performance monitoring

## Security

### Health Check Security
- No sensitive data exposure
- Configurable sensitive data logging
- Secure error messages
- Access control ready

### Input Validation
- Pydantic model validation
- Parameter bounds checking
- Type safety enforcement
- Error boundary protection

## Troubleshooting

### Common Issues

1. **"Component not available"**
   - Check if screening service components are properly imported
   - Verify component initialization
   - Check component dependencies

2. **"Health check timeout"**
   - Increase SCREENING_HEALTH_TIMEOUT value
   - Check component performance
   - Verify system resources

3. **"System metrics unavailable"**
   - Check psutil installation
   - Verify system permissions
   - Check system resource access

4. **"Cache not working"**
   - Verify cache configuration
   - Check cache duration settings
   - Monitor cache statistics

### Debug Mode
```bash
export SCREENING_HEALTH_LOG_LEVEL=DEBUG
export SCREENING_HEALTH_STRUCTURED_LOGGING=true
```

### Health Check Testing
```bash
# Test main health check
curl http://localhost:8005/health/

# Test component health
curl http://localhost:8005/health/multi-model-ranker

# Test readiness
curl http://localhost:8005/health/ready

# Test liveness
curl http://localhost:8005/health/live
```

## Migration from Mock Data

### Before (Mock Data)
```python
# OLD - Mock service fallback
class MockService:
    def get_uptime(self): return 0
    def get_ranking_status(self): 
        return type('Status', (), {'status': type('Status', (), {'value': 'idle'})(), 'active_rankings': [], 'completed_rankings': 0, 'failed_rankings': 0, 'average_ranking_time': 0, 'ranking_efficiency': 1.0, 'ai_available': True, 'available_models': []})()
```

### After (Real Service Integration)
```python
# NEW - Real service integration
try:
    from multi_model_ranker import multi_model_ranker
    ranking_status = multi_model_ranker.get_ranking_status()
    # Real health check logic with actual service status
except ImportError:
    # Service not available - return unhealthy status
    return ComponentHealthResponse(status="unhealthy", ...)
```

### Benefits
- ✅ **Real screening health monitoring** - Actual screening service component status
- ✅ **Production ready** - No mock data dependencies
- ✅ **Accurate status** - True screening service health representation
- ✅ **Professional error handling** - Comprehensive error management

## Support

For issues or questions:
1. Check logs: `tail -f logs/screening_service.log`
2. Verify configuration: `python -c "from health_config import validate_screening_service_health_config; print(validate_screening_service_health_config())"`
3. Test health endpoints: `curl http://localhost:8005/health/`
4. Monitor metrics: `curl http://localhost:8005/health/metrics`

---

**Status**: ✅ Production Ready - No Mock Data
**Last Updated**: 2025-10-03
**Version**: 1.0.0


