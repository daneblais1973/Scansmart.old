"""
Configuration Validation Script
==============================
Validate configuration files and environment setup
"""

import sys
import os
from pathlib import Path
import yaml
import json
from typing import List, Dict, Any
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from shared.configuration.config_manager import ConfigManager, Environment

logger = logging.getLogger(__name__)

class ConfigValidator:
    """Configuration validator"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate_all(self) -> bool:
        """Validate all configuration files"""
        logger.info("Starting configuration validation...")
        
        # Validate base configuration
        self._validate_base_config()
        
        # Validate environment configurations
        for env in Environment:
            self._validate_environment_config(env.value)
        
        # Validate service configurations
        self._validate_service_configs()
        
        # Validate environment variables
        self._validate_environment_variables()
        
        # Print results
        self._print_results()
        
        return len(self.errors) == 0
    
    def _validate_base_config(self):
        """Validate base configuration file"""
        base_config_file = self.config_dir / "base.yaml"
        
        if not base_config_file.exists():
            self.errors.append("Base configuration file not found: base.yaml")
            return
        
        try:
            with open(base_config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Validate required fields
            required_fields = [
                "environment",
                "database",
                "redis",
                "monitoring",
                "security",
                "ai"
            ]
            
            for field in required_fields:
                if field not in config:
                    self.errors.append(f"Required field missing in base config: {field}")
            
            # Validate database configuration
            if "database" in config:
                self._validate_database_config(config["database"])
            
            # Validate redis configuration
            if "redis" in config:
                self._validate_redis_config(config["redis"])
            
            # Validate monitoring configuration
            if "monitoring" in config:
                self._validate_monitoring_config(config["monitoring"])
            
            # Validate security configuration
            if "security" in config:
                self._validate_security_config(config["security"])
            
            # Validate AI configuration
            if "ai" in config:
                self._validate_ai_config(config["ai"])
            
        except Exception as e:
            self.errors.append(f"Error reading base configuration: {e}")
    
    def _validate_environment_config(self, environment: str):
        """Validate environment-specific configuration"""
        env_config_file = self.config_dir / f"{environment}.yaml"
        
        if not env_config_file.exists():
            self.warnings.append(f"Environment configuration file not found: {environment}.yaml")
            return
        
        try:
            with open(env_config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Validate environment-specific settings
            if environment == "production":
                self._validate_production_config(config)
            elif environment == "development":
                self._validate_development_config(config)
            
        except Exception as e:
            self.errors.append(f"Error reading {environment} configuration: {e}")
    
    def _validate_service_configs(self):
        """Validate service-specific configurations"""
        services_dir = self.config_dir / "services"
        
        if not services_dir.exists():
            self.warnings.append("Services configuration directory not found")
            return
        
        for service_file in services_dir.glob("*.yaml"):
            service_name = service_file.stem
            
            try:
                with open(service_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Validate service configuration
                self._validate_service_config(service_name, config)
                
            except Exception as e:
                self.errors.append(f"Error reading service configuration {service_name}: {e}")
    
    def _validate_database_config(self, config: Dict[str, Any]):
        """Validate database configuration"""
        required_fields = ["type", "host", "port", "name"]
        
        for field in required_fields:
            if field not in config:
                self.errors.append(f"Database configuration missing required field: {field}")
        
        # Validate port
        if "port" in config and not isinstance(config["port"], int):
            self.errors.append("Database port must be an integer")
        
        # Validate pool size
        if "pool_size" in config and not isinstance(config["pool_size"], int):
            self.errors.append("Database pool_size must be an integer")
        
        # Validate database type
        if "type" in config and config["type"] not in ["sqlite", "postgresql", "mysql"]:
            self.warnings.append(f"Unsupported database type: {config['type']}")
    
    def _validate_redis_config(self, config: Dict[str, Any]):
        """Validate Redis configuration"""
        required_fields = ["host", "port"]
        
        for field in required_fields:
            if field not in config:
                self.errors.append(f"Redis configuration missing required field: {field}")
        
        # Validate port
        if "port" in config and not isinstance(config["port"], int):
            self.errors.append("Redis port must be an integer")
        
        # Validate database number
        if "db" in config and not isinstance(config["db"], int):
            self.errors.append("Redis db must be an integer")
    
    def _validate_monitoring_config(self, config: Dict[str, Any]):
        """Validate monitoring configuration"""
        required_fields = ["enabled", "metrics_interval", "retention_days"]
        
        for field in required_fields:
            if field not in config:
                self.errors.append(f"Monitoring configuration missing required field: {field}")
        
        # Validate metrics interval
        if "metrics_interval" in config and not isinstance(config["metrics_interval"], int):
            self.errors.append("Monitoring metrics_interval must be an integer")
        
        # Validate retention days
        if "retention_days" in config and not isinstance(config["retention_days"], int):
            self.errors.append("Monitoring retention_days must be an integer")
    
    def _validate_security_config(self, config: Dict[str, Any]):
        """Validate security configuration"""
        required_fields = ["authentication_enabled", "authorization_enabled"]
        
        for field in required_fields:
            if field not in config:
                self.errors.append(f"Security configuration missing required field: {field}")
        
        # Validate JWT secret
        if config.get("authentication_enabled", False) and not config.get("jwt_secret"):
            self.errors.append("JWT secret required when authentication is enabled")
        
        # Validate JWT expiry
        if "jwt_expiry" in config and not isinstance(config["jwt_expiry"], int):
            self.errors.append("JWT expiry must be an integer")
    
    def _validate_ai_config(self, config: Dict[str, Any]):
        """Validate AI configuration"""
        required_fields = ["quantum_enabled", "ai_enabled"]
        
        for field in required_fields:
            if field not in config:
                self.errors.append(f"AI configuration missing required field: {field}")
        
        # Validate model timeout
        if "model_timeout" in config and not isinstance(config["model_timeout"], int):
            self.errors.append("AI model_timeout must be an integer")
        
        # Validate max models
        if "max_models" in config and not isinstance(config["max_models"], int):
            self.errors.append("AI max_models must be an integer")
    
    def _validate_production_config(self, config: Dict[str, Any]):
        """Validate production-specific configuration"""
        # Check for debug mode
        if config.get("debug", False):
            self.warnings.append("Debug mode enabled in production configuration")
        
        # Check for authentication
        if not config.get("security", {}).get("authentication_enabled", False):
            self.warnings.append("Authentication not enabled in production")
        
        # Check for rate limiting
        if not config.get("security", {}).get("rate_limiting_enabled", False):
            self.warnings.append("Rate limiting not enabled in production")
        
        # Check for secure JWT secret
        jwt_secret = config.get("security", {}).get("jwt_secret", "")
        if jwt_secret in ["", "your-secret-key-here", "dev-secret-key"]:
            self.errors.append("Production JWT secret is not secure")
    
    def _validate_development_config(self, config: Dict[str, Any]):
        """Validate development-specific configuration"""
        # Check for debug mode
        if not config.get("debug", False):
            self.warnings.append("Debug mode not enabled in development configuration")
        
        # Check for authentication
        if config.get("security", {}).get("authentication_enabled", False):
            self.warnings.append("Authentication enabled in development (may be intentional)")
    
    def _validate_service_config(self, service_name: str, config: Dict[str, Any]):
        """Validate service-specific configuration"""
        required_fields = ["host", "port"]
        
        for field in required_fields:
            if field not in config:
                self.errors.append(f"Service {service_name} missing required field: {field}")
        
        # Validate port
        if "port" in config and not isinstance(config["port"], int):
            self.errors.append(f"Service {service_name} port must be an integer")
        
        # Validate timeout
        if "timeout" in config and not isinstance(config["timeout"], int):
            self.errors.append(f"Service {service_name} timeout must be an integer")
    
    def _validate_environment_variables(self):
        """Validate environment variables"""
        # Check for required environment variables
        required_env_vars = [
            "DATABASE_HOST",
            "DATABASE_PORT",
            "REDIS_HOST",
            "REDIS_PORT"
        ]
        
        for var in required_env_vars:
            if not os.getenv(var):
                self.warnings.append(f"Environment variable not set: {var}")
    
    def _print_results(self):
        """Print validation results"""
        print("\n" + "="*60)
        print("CONFIGURATION VALIDATION RESULTS")
        print("="*60)
        
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"  - {error}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        if not self.errors and not self.warnings:
            print("\n✅ All configuration files are valid!")
        elif not self.errors:
            print(f"\n✅ Configuration is valid with {len(self.warnings)} warnings")
        else:
            print(f"\n❌ Configuration has {len(self.errors)} errors and {len(self.warnings)} warnings")
        
        print("="*60)

def main():
    """Main validation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate ScanSmart configuration")
    parser.add_argument("--config-dir", default="config", help="Configuration directory")
    parser.add_argument("--environment", choices=["development", "staging", "production"], 
                       help="Environment to validate")
    
    args = parser.parse_args()
    
    validator = ConfigValidator(args.config_dir)
    
    if args.environment:
        # Validate specific environment
        validator._validate_environment_config(args.environment)
    else:
        # Validate all configurations
        validator.validate_all()
    
    sys.exit(0 if validator.validate_all() else 1)

if __name__ == "__main__":
    main()



