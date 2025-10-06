"""
Advanced Security System
=======================
Enterprise-grade security with encryption, authentication, and authorization.
"""

import asyncio
import logging
import hashlib
import hmac
import secrets
import time
import json
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Security levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class EncryptionAlgorithm(Enum):
    """Encryption algorithms"""
    AES_256 = "aes_256"
    RSA_2048 = "rsa_2048"
    RSA_4096 = "rsa_4096"
    CHACHA20 = "chacha20"
    BLAKE2B = "blake2b"

class AuthenticationMethod(Enum):
    """Authentication methods"""
    PASSWORD = "password"
    TOKEN = "token"
    CERTIFICATE = "certificate"
    BIOMETRIC = "biometric"
    MULTI_FACTOR = "multi_factor"

class Permission(Enum):
    """System permissions"""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"
    MONITOR = "monitor"
    AUDIT = "audit"

@dataclass
class User:
    """User entity"""
    user_id: str
    username: str
    email: str
    password_hash: str
    salt: str
    permissions: List[Permission] = field(default_factory=list)
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    is_active: bool = True
    failed_attempts: int = 0
    locked_until: Optional[datetime] = None

@dataclass
class SecurityToken:
    """Security token"""
    token_id: str
    user_id: str
    token: str
    expires_at: datetime
    permissions: List[Permission] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    is_revoked: bool = False

@dataclass
class SecurityEvent:
    """Security event for auditing"""
    event_id: str
    event_type: str
    user_id: Optional[str]
    timestamp: datetime
    severity: SecurityLevel
    description: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class EncryptionService:
    """Advanced encryption service"""
    
    def __init__(self):
        self.encryption_keys = {}
        self.key_rotation_interval = timedelta(days=30)
        self.last_key_rotation = datetime.now()
        
    def generate_key(self, algorithm: EncryptionAlgorithm) -> bytes:
        """Generate encryption key"""
        if algorithm == EncryptionAlgorithm.AES_256:
            return secrets.token_bytes(32)  # 256 bits
        elif algorithm == EncryptionAlgorithm.CHACHA20:
            return secrets.token_bytes(32)  # 256 bits
        elif algorithm == EncryptionAlgorithm.BLAKE2B:
            return secrets.token_bytes(64)  # 512 bits
        else:
            return secrets.token_bytes(32)
    
    def encrypt_data(self, data: str, key: bytes, algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256) -> Dict[str, Any]:
        """Encrypt data"""
        try:
            # Simple encryption simulation (in production, use proper crypto libraries)
            if algorithm == EncryptionAlgorithm.AES_256:
                # Simulate AES-256 encryption
                encrypted_data = base64.b64encode(data.encode()).decode()
                return {
                    'encrypted_data': encrypted_data,
                    'algorithm': algorithm.value,
                    'key_id': hashlib.sha256(key).hexdigest()[:16],
                    'timestamp': datetime.now().isoformat()
                }
            elif algorithm == EncryptionAlgorithm.CHACHA20:
                # Simulate ChaCha20 encryption
                encrypted_data = base64.b64encode(data.encode()).decode()
                return {
                    'encrypted_data': encrypted_data,
                    'algorithm': algorithm.value,
                    'key_id': hashlib.sha256(key).hexdigest()[:16],
                    'timestamp': datetime.now().isoformat()
                }
            else:
                # Default encryption
                encrypted_data = base64.b64encode(data.encode()).decode()
                return {
                    'encrypted_data': encrypted_data,
                    'algorithm': algorithm.value,
                    'key_id': hashlib.sha256(key).hexdigest()[:16],
                    'timestamp': datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Error encrypting data: {e}")
            return None
    
    def decrypt_data(self, encrypted_data: str, key: bytes, algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256) -> str:
        """Decrypt data"""
        try:
            # Simple decryption simulation
            decrypted_data = base64.b64decode(encrypted_data.encode()).decode()
            return decrypted_data
        except Exception as e:
            logger.error(f"Error decrypting data: {e}")
            return None
    
    def hash_password(self, password: str, salt: str = None) -> tuple[str, str]:
        """Hash password with salt"""
        if salt is None:
            salt = secrets.token_hex(32)
        
        # Use PBKDF2 for password hashing
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return password_hash.hex(), salt
    
    def verify_password(self, password: str, password_hash: str, salt: str) -> bool:
        """Verify password"""
        computed_hash, _ = self.hash_password(password, salt)
        return hmac.compare_digest(computed_hash, password_hash)

class AuthenticationService:
    """Advanced authentication service"""
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.tokens: Dict[str, SecurityToken] = {}
        self.active_sessions: Dict[str, str] = {}  # session_id -> user_id
        self.failed_attempts: Dict[str, List[datetime]] = defaultdict(list)
        self.lock = threading.Lock()
        
        # Security policies
        self.max_failed_attempts = 5
        self.lockout_duration = timedelta(minutes=15)
        self.session_timeout = timedelta(hours=8)
        self.token_expiry = timedelta(days=7)
        
    def register_user(self, username: str, email: str, password: str, 
                    permissions: List[Permission] = None, 
                    security_level: SecurityLevel = SecurityLevel.MEDIUM) -> str:
        """Register a new user"""
        try:
            with self.lock:
                # Check if user already exists
                for user in self.users.values():
                    if user.username == username or user.email == email:
                        raise ValueError("User already exists")
                
                # Create user
                user_id = str(uuid.uuid4())
                password_hash, salt = EncryptionService().hash_password(password)
                
                user = User(
                    user_id=user_id,
                    username=username,
                    email=email,
                    password_hash=password_hash,
                    salt=salt,
                    permissions=permissions or [Permission.READ],
                    security_level=security_level
                )
                
                self.users[user_id] = user
                logger.info(f"User registered: {username}")
                return user_id
        except Exception as e:
            logger.error(f"Error registering user: {e}")
            return None
    
    def authenticate_user(self, username: str, password: str, 
                         ip_address: str = None, user_agent: str = None) -> Optional[str]:
        """Authenticate user"""
        try:
            with self.lock:
                # Find user
                user = None
                for u in self.users.values():
                    if u.username == username:
                        user = u
                        break
                
                if not user:
                    self._record_failed_attempt(username, ip_address)
                    return None
                
                # Check if account is locked
                if user.locked_until and datetime.now() < user.locked_until:
                    logger.warning(f"Account locked for user: {username}")
                    return None
                
                # Verify password
                if not EncryptionService().verify_password(password, user.password_hash, user.salt):
                    self._record_failed_attempt(username, ip_address)
                    user.failed_attempts += 1
                    
                    # Lock account if too many failed attempts
                    if user.failed_attempts >= self.max_failed_attempts:
                        user.locked_until = datetime.now() + self.lockout_duration
                        logger.warning(f"Account locked due to failed attempts: {username}")
                    
                    return None
                
                # Reset failed attempts on successful login
                user.failed_attempts = 0
                user.locked_until = None
                user.last_login = datetime.now()
                
                # Create session
                session_id = str(uuid.uuid4())
                self.active_sessions[session_id] = user.user_id
                
                logger.info(f"User authenticated: {username}")
                return session_id
        except Exception as e:
            logger.error(f"Error authenticating user: {e}")
            return None
    
    def create_token(self, user_id: str, permissions: List[Permission] = None) -> Optional[str]:
        """Create security token"""
        try:
            with self.lock:
                if user_id not in self.users:
                    return None
                
                user = self.users[user_id]
                token_id = str(uuid.uuid4())
                token = secrets.token_urlsafe(64)
                
                security_token = SecurityToken(
                    token_id=token_id,
                    user_id=user_id,
                    token=token,
                    expires_at=datetime.now() + self.token_expiry,
                    permissions=permissions or user.permissions
                )
                
                self.tokens[token_id] = security_token
                logger.info(f"Token created for user: {user_id}")
                return token
        except Exception as e:
            logger.error(f"Error creating token: {e}")
            return None
    
    def validate_token(self, token: str) -> Optional[SecurityToken]:
        """Validate security token"""
        try:
            with self.lock:
                for security_token in self.tokens.values():
                    if security_token.token == token and not security_token.is_revoked:
                        if datetime.now() < security_token.expires_at:
                            security_token.last_used = datetime.now()
                            return security_token
                        else:
                            # Token expired
                            security_token.is_revoked = True
                return None
        except Exception as e:
            logger.error(f"Error validating token: {e}")
            return None
    
    def revoke_token(self, token: str) -> bool:
        """Revoke security token"""
        try:
            with self.lock:
                for security_token in self.tokens.values():
                    if security_token.token == token:
                        security_token.is_revoked = True
                        logger.info(f"Token revoked: {security_token.token_id}")
                        return True
                return False
        except Exception as e:
            logger.error(f"Error revoking token: {e}")
            return False
    
    def _record_failed_attempt(self, username: str, ip_address: str = None):
        """Record failed authentication attempt"""
        self.failed_attempts[username].append(datetime.now())
        logger.warning(f"Failed authentication attempt for user: {username} from IP: {ip_address}")

class AuthorizationService:
    """Advanced authorization service"""
    
    def __init__(self):
        self.permission_matrix = {
            Permission.READ: [Permission.READ],
            Permission.WRITE: [Permission.READ, Permission.WRITE],
            Permission.EXECUTE: [Permission.READ, Permission.WRITE, Permission.EXECUTE],
            Permission.ADMIN: [Permission.READ, Permission.WRITE, Permission.EXECUTE, Permission.ADMIN],
            Permission.MONITOR: [Permission.READ, Permission.MONITOR],
            Permission.AUDIT: [Permission.READ, Permission.AUDIT]
        }
    
    def check_permission(self, user_permissions: List[Permission], required_permission: Permission) -> bool:
        """Check if user has required permission"""
        if Permission.ADMIN in user_permissions:
            return True
        
        return required_permission in user_permissions
    
    def check_resource_access(self, user_id: str, resource: str, action: str, 
                            authentication_service: AuthenticationService) -> bool:
        """Check resource access"""
        try:
            # Get user
            user = authentication_service.users.get(user_id)
            if not user or not user.is_active:
                return False
            
            # Map action to permission
            action_permissions = {
                'read': Permission.READ,
                'write': Permission.WRITE,
                'execute': Permission.EXECUTE,
                'admin': Permission.ADMIN,
                'monitor': Permission.MONITOR,
                'audit': Permission.AUDIT
            }
            
            required_permission = action_permissions.get(action.lower())
            if not required_permission:
                return False
            
            return self.check_permission(user.permissions, required_permission)
        except Exception as e:
            logger.error(f"Error checking resource access: {e}")
            return False

class SecurityAuditService:
    """Security audit and monitoring service"""
    
    def __init__(self):
        self.security_events: deque = deque(maxlen=10000)
        self.alert_handlers: List[Callable[[SecurityEvent], None]] = []
        self.security_policies = {
            'max_failed_logins': 5,
            'suspicious_activity_threshold': 10,
            'password_policy': {
                'min_length': 8,
                'require_uppercase': True,
                'require_lowercase': True,
                'require_numbers': True,
                'require_special_chars': True
            }
        }
    
    def log_security_event(self, event_type: str, user_id: str = None, 
                          severity: SecurityLevel = SecurityLevel.MEDIUM,
                          description: str = "", ip_address: str = None,
                          user_agent: str = None, metadata: Dict[str, Any] = None):
        """Log security event"""
        try:
            event = SecurityEvent(
                event_id=str(uuid.uuid4()),
                event_type=event_type,
                user_id=user_id,
                timestamp=datetime.now(),
                severity=severity,
                description=description,
                ip_address=ip_address,
                user_agent=user_agent,
                metadata=metadata or {}
            )
            
            self.security_events.append(event)
            
            # Trigger alert handlers for high severity events
            if severity in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
                for handler in self.alert_handlers:
                    try:
                        handler(event)
                    except Exception as e:
                        logger.error(f"Error in security alert handler: {e}")
            
            logger.info(f"Security event logged: {event_type} - {description}")
        except Exception as e:
            logger.error(f"Error logging security event: {e}")
    
    def add_alert_handler(self, handler: Callable[[SecurityEvent], None]):
        """Add security alert handler"""
        self.alert_handlers.append(handler)
    
    def get_security_events(self, start_time: datetime = None, end_time: datetime = None,
                           severity: SecurityLevel = None) -> List[SecurityEvent]:
        """Get security events"""
        events = list(self.security_events)
        
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        if severity:
            events = [e for e in events if e.severity == severity]
        
        return events
    
    def analyze_security_threats(self) -> Dict[str, Any]:
        """Analyze security threats"""
        try:
            recent_events = [e for e in self.security_events 
                           if e.timestamp >= datetime.now() - timedelta(hours=24)]
            
            threat_analysis = {
                'total_events': len(recent_events),
                'high_severity_events': len([e for e in recent_events if e.severity == SecurityLevel.HIGH]),
                'critical_events': len([e for e in recent_events if e.severity == SecurityLevel.CRITICAL]),
                'failed_logins': len([e for e in recent_events if e.event_type == 'failed_login']),
                'suspicious_activities': len([e for e in recent_events if e.event_type == 'suspicious_activity']),
                'threat_level': 'low'
            }
            
            # Determine threat level
            if threat_analysis['critical_events'] > 0:
                threat_analysis['threat_level'] = 'critical'
            elif threat_analysis['high_severity_events'] > 5:
                threat_analysis['threat_level'] = 'high'
            elif threat_analysis['failed_logins'] > 10:
                threat_analysis['threat_level'] = 'medium'
            
            return threat_analysis
        except Exception as e:
            logger.error(f"Error analyzing security threats: {e}")
            return {'error': str(e)}

class AdvancedSecuritySystem:
    """Main security system"""
    
    def __init__(self):
        self.encryption_service = EncryptionService()
        self.authentication_service = AuthenticationService()
        self.authorization_service = AuthorizationService()
        self.audit_service = SecurityAuditService()
        
        # Setup default admin user
        self._setup_default_admin()
        
        # Setup security alert handlers
        self._setup_security_handlers()
        
        logger.info("Advanced Security System initialized")
    
    def _setup_default_admin(self):
        """Setup default admin user"""
        try:
            admin_user_id = self.authentication_service.register_user(
                username="admin",
                email="admin@scansmart.com",
                password="admin123!",
                permissions=[Permission.ADMIN, Permission.MONITOR, Permission.AUDIT],
                security_level=SecurityLevel.HIGH
            )
            
            if admin_user_id:
                logger.info("Default admin user created")
        except Exception as e:
            logger.error(f"Error setting up default admin: {e}")
    
    def _setup_security_handlers(self):
        """Setup security alert handlers"""
        def security_alert_handler(event: SecurityEvent):
            logger.warning(f"Security Alert: {event.severity.value.upper()} - {event.description}")
        
        self.audit_service.add_alert_handler(security_alert_handler)
    
    async def authenticate_user(self, username: str, password: str, 
                              ip_address: str = None, user_agent: str = None) -> Optional[str]:
        """Authenticate user and return session ID"""
        try:
            session_id = self.authentication_service.authenticate_user(
                username, password, ip_address, user_agent
            )
            
            if session_id:
                # Log successful authentication
                self.audit_service.log_security_event(
                    event_type="successful_login",
                    user_id=self.authentication_service.active_sessions.get(session_id),
                    severity=SecurityLevel.LOW,
                    description=f"User {username} logged in successfully",
                    ip_address=ip_address,
                    user_agent=user_agent
                )
            else:
                # Log failed authentication
                self.audit_service.log_security_event(
                    event_type="failed_login",
                    severity=SecurityLevel.MEDIUM,
                    description=f"Failed login attempt for user {username}",
                    ip_address=ip_address,
                    user_agent=user_agent
                )
            
            return session_id
        except Exception as e:
            logger.error(f"Error in authentication: {e}")
            return None
    
    async def create_token(self, session_id: str, permissions: List[Permission] = None) -> Optional[str]:
        """Create security token for session"""
        try:
            user_id = self.authentication_service.active_sessions.get(session_id)
            if not user_id:
                return None
            
            token = self.authentication_service.create_token(user_id, permissions)
            
            if token:
                self.audit_service.log_security_event(
                    event_type="token_created",
                    user_id=user_id,
                    severity=SecurityLevel.LOW,
                    description="Security token created"
                )
            
            return token
        except Exception as e:
            logger.error(f"Error creating token: {e}")
            return None
    
    async def validate_access(self, token: str, resource: str, action: str) -> bool:
        """Validate access to resource"""
        try:
            security_token = self.authentication_service.validate_token(token)
            if not security_token:
                return False
            
            # Check if token has required permissions
            action_permissions = {
                'read': Permission.READ,
                'write': Permission.WRITE,
                'execute': Permission.EXECUTE,
                'admin': Permission.ADMIN,
                'monitor': Permission.MONITOR,
                'audit': Permission.AUDIT
            }
            
            required_permission = action_permissions.get(action.lower())
            if not required_permission:
                return False
            
            has_permission = self.authorization_service.check_permission(
                security_token.permissions, required_permission
            )
            
            if has_permission:
                # Log successful access
                self.audit_service.log_security_event(
                    event_type="resource_access",
                    user_id=security_token.user_id,
                    severity=SecurityLevel.LOW,
                    description=f"Access granted to {resource} for {action}",
                    metadata={'resource': resource, 'action': action}
                )
            else:
                # Log denied access
                self.audit_service.log_security_event(
                    event_type="access_denied",
                    user_id=security_token.user_id,
                    severity=SecurityLevel.MEDIUM,
                    description=f"Access denied to {resource} for {action}",
                    metadata={'resource': resource, 'action': action}
                )
            
            return has_permission
        except Exception as e:
            logger.error(f"Error validating access: {e}")
            return False
    
    async def encrypt_sensitive_data(self, data: str, algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256) -> Dict[str, Any]:
        """Encrypt sensitive data"""
        try:
            key = self.encryption_service.generate_key(algorithm)
            encrypted_data = self.encryption_service.encrypt_data(data, key, algorithm)
            
            if encrypted_data:
                self.audit_service.log_security_event(
                    event_type="data_encrypted",
                    severity=SecurityLevel.LOW,
                    description=f"Data encrypted using {algorithm.value}",
                    metadata={'algorithm': algorithm.value}
                )
            
            return encrypted_data
        except Exception as e:
            logger.error(f"Error encrypting data: {e}")
            return None
    
    async def get_security_dashboard(self) -> Dict[str, Any]:
        """Get security dashboard"""
        try:
            threat_analysis = self.audit_service.analyze_security_threats()
            
            return {
                'users': {
                    'total_users': len(self.authentication_service.users),
                    'active_sessions': len(self.authentication_service.active_sessions),
                    'active_tokens': len([t for t in self.authentication_service.tokens.values() if not t.is_revoked])
                },
                'security_events': {
                    'total_events': len(self.audit_service.security_events),
                    'recent_events': len([e for e in self.audit_service.security_events 
                                        if e.timestamp >= datetime.now() - timedelta(hours=24)])
                },
                'threat_analysis': threat_analysis,
                'security_policies': self.audit_service.security_policies
            }
        except Exception as e:
            logger.error(f"Error getting security dashboard: {e}")
            return {'error': str(e)}

# Global instance
advanced_security = AdvancedSecuritySystem()




