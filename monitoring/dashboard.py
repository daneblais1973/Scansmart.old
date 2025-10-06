"""
Monitoring Dashboard
===================
Web-based monitoring dashboard for ScanSmart services
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, List, Any
import json
import logging

from system_monitor import system_monitor, ServiceStatus

logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="ScanSmart Monitoring Dashboard",
    description="Real-time monitoring dashboard for all services",
    version="1.0.0"
)

# Templates
templates = Jinja2Templates(directory="monitoring/templates")

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main monitoring dashboard"""
    try:
        # Get system status
        system_status = system_monitor.get_system_status()
        
        # Get service health data
        services_data = []
        for service_name, metrics in system_status["services"].items():
            services_data.append({
                "name": service_name,
                "status": metrics["status"],
                "response_time": metrics["response_time"],
                "cpu_usage": metrics["cpu_usage"],
                "memory_usage": metrics["memory_usage"],
                "uptime": metrics["uptime"],
                "last_check": metrics["last_check"]
            })
        
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "system_status": system_status,
            "services": services_data,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error rendering dashboard: {e}")
        return HTMLResponse(f"<h1>Error loading dashboard: {str(e)}</h1>")

@app.get("/api/status")
async def api_status():
    """API endpoint for system status"""
    try:
        return system_monitor.get_system_status()
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return {"error": str(e)}

@app.get("/api/services")
async def api_services():
    """API endpoint for services status"""
    try:
        system_status = system_monitor.get_system_status()
        return {
            "services": system_status["services"],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting services status: {e}")
        return {"error": str(e)}

@app.get("/api/alerts")
async def api_alerts():
    """API endpoint for alerts"""
    try:
        system_status = system_monitor.get_system_status()
        return {
            "alerts": system_status["alerts"],
            "active_alerts": system_status["active_alerts"],
            "critical_alerts": system_status["critical_alerts"],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        return {"error": str(e)}

@app.post("/api/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str):
    """Resolve an alert"""
    try:
        success = system_monitor.resolve_alert(alert_id)
        return {"success": success, "alert_id": alert_id}
    except Exception as e:
        logger.error(f"Error resolving alert: {e}")
        return {"error": str(e)}

@app.get("/api/metrics")
async def api_metrics():
    """API endpoint for Prometheus metrics"""
    try:
        # This would typically return Prometheus-formatted metrics
        # For now, return JSON format
        system_status = system_monitor.get_system_status()
        return {
            "metrics": {
                "system_cpu_usage": system_status.get("system_cpu_usage", 0),
                "system_memory_usage": system_status.get("system_memory_usage", 0),
                "system_disk_usage": system_status.get("system_disk_usage", 0),
                "healthy_services": system_status["healthy_services"],
                "total_services": system_status["total_services"],
                "active_alerts": system_status["active_alerts"]
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    """Health check for the monitoring dashboard"""
    try:
        return {
            "status": "healthy",
            "service": "monitoring-dashboard",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

