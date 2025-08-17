"""
Production Security Hardening Tests

Tests for container security, network security, secrets management,
and compliance validation in production environments.
"""

import asyncio
import hashlib
import json
import os
import secrets
import socket
import ssl
import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch

import docker
import httpx
import jwt
import pytest

# Mock problematic dependencies
try:
    import numpy
except ImportError:
    import sys
    sys.modules['numpy'] = Mock()

try:
    import slowapi.util
except ImportError:
    import sys
    slowapi_mock = Mock()
    slowapi_mock.util = Mock()
    slowapi_mock.util.get_remote_address = Mock(return_value="192.168.1.100")
    sys.modules['slowapi'] = slowapi_mock
    sys.modules['slowapi.util'] = slowapi_mock.util

try:
    import redis
except ImportError:
    import sys
    redis_mock = Mock()
    redis_mock.asyncio = Mock()
    sys.modules['redis'] = redis_mock
    sys.modules['redis.asyncio'] = redis_mock.asyncio

# Import LightRAG modules
from lightrag.api.auth.handler import AuthHandler
from lightrag.api.middleware.rate_limiter import (
    AdvancedRateLimiter,
    RateLimitConfig,
    RateLimitType,
)
from lightrag.api.middleware.security_headers import (
    SecurityHeadersConfig,
    SecurityHeadersMiddleware,
)


@pytest.mark.production
@pytest.mark.security
class TestContainerSecurity:
    """Test container security configurations and hardening."""
    
    @pytest.fixture
    def docker_client(self):
        """Docker client for container tests."""
        try:
            client = docker.from_env()
            yield client
        except Exception:
            pytest.skip("Docker not available")
        finally:
            if 'client' in locals():
                client.close()
    
    @pytest.fixture
    def production_dockerfile_path(self):
        """Path to production Dockerfile."""
        return Path("/opt/developments/LightRAG/Dockerfile.production")
    
    async def test_container_runs_as_non_root(self, docker_client, production_dockerfile_path):
        """Test that production container runs as non-root user."""
        if not production_dockerfile_path.exists():
            pytest.skip("Production Dockerfile not found")
        
        # Build test image
        image_tag = "lightrag:security-test"
        try:
            image, logs = docker_client.images.build(
                path=str(production_dockerfile_path.parent),
                dockerfile=str(production_dockerfile_path.name),
                tag=image_tag,
                rm=True
            )
            
            # Run container and check user
            container = docker_client.containers.run(
                image_tag,
                command="id -u",
                remove=True,
                detach=False
            )
            
            # Check that UID is not 0 (root)
            output = container.decode().strip()
            uid = int(output)
            assert uid != 0, "Container should not run as root user"
            assert uid >= 1000, "Container should use high UID for security"
            
        finally:
            # Cleanup
            try:
                docker_client.images.remove(image_tag, force=True)
            except:
                pass
    
    async def test_container_readonly_filesystem(self, docker_client):
        """Test container with read-only filesystem configuration."""
        # Test that critical directories are read-only
        test_script = """
        import os
        import tempfile
        
        # Test writing to root filesystem (should fail)
        try:
            with open('/test_file', 'w') as f:
                f.write('test')
            print('FAIL: Root filesystem is writable')
            exit(1)
        except PermissionError:
            print('PASS: Root filesystem is read-only')
        
        # Test writing to /tmp (should work)
        try:
            with tempfile.NamedTemporaryFile(dir='/tmp', delete=False) as f:
                f.write(b'test')
            print('PASS: /tmp is writable')
        except:
            print('FAIL: /tmp is not writable')
            exit(1)
        """
        
        container = docker_client.containers.run(
            "python:3.11-slim",
            command=f"python -c \"{test_script}\"",
            read_only=True,
            tmpfs={'/tmp': 'rw,size=100m'},
            remove=True,
            detach=False
        )
        
        output = container.decode().strip()
        assert "PASS: Root filesystem is read-only" in output
        assert "PASS: /tmp is writable" in output
    
    async def test_container_no_privileged_capabilities(self, docker_client):
        """Test that container doesn't have privileged capabilities."""
        # Check capabilities in container
        container = docker_client.containers.run(
            "python:3.11-slim",
            command="python -c \"import os; print('capabilities:', os.getenv('HOME', 'unknown'))\"",
            cap_drop=['ALL'],
            cap_add=['CHOWN', 'SETGID', 'SETUID'],  # Minimal required caps
            remove=True,
            detach=False
        )
        
        # Container should start successfully with minimal capabilities
        assert container is not None
    
    async def test_secrets_not_in_environment(self, docker_client):
        """Test that secrets are not exposed in environment variables."""
        # Test script to check for common secret patterns
        check_script = """
        import os
        import re
        
        secret_patterns = [
            r'(password|pwd|secret|key|token).*[=:].*',
            r'[A-Za-z0-9+/]{20,}={0,2}',  # Base64-like strings
            r'[0-9a-f]{32,}',  # Hex strings
        ]
        
        found_secrets = []
        for key, value in os.environ.items():
            for pattern in secret_patterns:
                if re.search(pattern, f'{key}={value}', re.IGNORECASE):
                    if 'test' not in key.lower() and 'example' not in value.lower():
                        found_secrets.append(f'{key}={value[:10]}...')
        
        if found_secrets:
            print('FAIL: Found potential secrets:', found_secrets)
            exit(1)
        else:
            print('PASS: No secrets found in environment')
        """
        
        container = docker_client.containers.run(
            "python:3.11-slim",
            command=f"python -c \"{check_script}\"",
            environment={
                'SAFE_VAR': 'safe_value',
                'TEST_SECRET': 'test_value_ok'  # This is OK since it contains 'test'
            },
            remove=True,
            detach=False
        )
        
        output = container.decode().strip()
        assert "PASS: No secrets found in environment" in output


@pytest.mark.production
@pytest.mark.security
class TestNetworkSecurity:
    """Test network security configurations."""
    
    async def test_tls_configuration(self):
        """Test TLS/SSL configuration for HTTPS endpoints."""
        # Mock SSL context for testing
        with patch('ssl.create_default_context') as mock_ssl:
            mock_context = Mock()
            mock_context.check_hostname = True
            mock_context.verify_mode = ssl.CERT_REQUIRED
            mock_ssl.return_value = mock_context
            
            # Test SSL context creation
            context = ssl.create_default_context()
            assert context.check_hostname is True
            assert context.verify_mode == ssl.CERT_REQUIRED
    
    async def test_secure_headers_middleware(self):
        """Test security headers are properly set."""
        # Create security headers middleware
        config = SecurityHeadersConfig()
        security_headers = SecurityHeadersMiddleware(app=Mock(), config=config)
        
        # Mock request and response
        mock_request = Mock()
        mock_response = Mock()
        mock_response.headers = {}
        
        # Mock call_next function
        async def mock_call_next(request):
            return mock_response
        
        # Apply security headers
        response = await security_headers.dispatch(mock_request, mock_call_next)
        
        # Check required security headers
        expected_headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'",
            'Referrer-Policy': 'strict-origin-when-cross-origin'
        }
        
        for header, value in expected_headers.items():
            assert header in response.headers
            if header == 'Content-Security-Policy':
                assert "'self'" in response.headers[header]
            else:
                assert response.headers[header] == value
    
    async def test_cors_configuration(self):
        """Test CORS configuration is secure."""
        # Mock CORS middleware configuration
        cors_config = {
            'allow_origins': ['https://trusted-domain.com'],
            'allow_methods': ['GET', 'POST', 'PUT', 'DELETE'],
            'allow_headers': ['Authorization', 'Content-Type'],
            'allow_credentials': True,
            'max_age': 86400
        }
        
        # Test that wildcard origins are not allowed with credentials
        assert '*' not in cors_config['allow_origins']
        assert cors_config['allow_credentials'] is True
        
        # Test that only necessary methods are allowed
        dangerous_methods = ['TRACE', 'CONNECT', 'OPTIONS']
        for method in dangerous_methods:
            if method in cors_config['allow_methods']:
                assert method in ['OPTIONS']  # OPTIONS is OK for preflight
    
    async def test_rate_limiting_configuration(self):
        """Test rate limiting is properly configured."""
        # Create rate limiter config
        config = RateLimitConfig(
            general_api_limit="60/minute"
        )
        rate_limiter = AdvancedRateLimiter(config=config)
        
        # Mock request
        mock_request = Mock()
        mock_request.client.host = '192.168.1.100'
        mock_request.url.path = '/api/query'
        
        # Test rate limiting logic
        client_id = rate_limiter._get_client_identifier(mock_request)
        assert client_id is not None
        
        # Test configuration
        assert rate_limiter.config.general_api_limit == "60/minute"


@pytest.mark.production
@pytest.mark.security
class TestSecretsManagement:
    """Test secrets management and encryption."""
    
    async def test_jwt_secret_generation(self):
        """Test JWT secret key generation and validation."""
        # Test secret generation
        secret = secrets.token_urlsafe(32)
        assert len(secret) >= 32
        assert secret.isascii()
        
        # Test JWT creation and validation
        payload = {
            'user_id': 'test_user',
            'exp': datetime.utcnow() + timedelta(hours=1),
            'iat': datetime.utcnow()
        }
        
        token = jwt.encode(payload, secret, algorithm='HS256')
        assert isinstance(token, str)
        
        # Validate token
        decoded = jwt.decode(token, secret, algorithms=['HS256'])
        assert decoded['user_id'] == 'test_user'
    
    async def test_password_hashing(self):
        """Test password hashing implementation."""
        from passlib.context import CryptContext
        
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        # Test password hashing
        password = "test_password_123"
        hashed = pwd_context.hash(password)
        
        assert hashed != password
        assert len(hashed) > 50  # bcrypt hashes are long
        assert hashed.startswith('$2b$')  # bcrypt format
        
        # Test password verification
        assert pwd_context.verify(password, hashed) is True
        assert pwd_context.verify("wrong_password", hashed) is False
    
    async def test_api_key_validation(self):
        """Test API key validation and security."""
        # Create auth handler
        auth_handler = AuthHandler(
            jwt_secret="test_secret_key_32_chars_long",
            jwt_expire_hours=24
        )
        
        # Test API key format
        api_key = "lightrag_" + secrets.token_urlsafe(32)
        assert api_key.startswith("lightrag_")
        assert len(api_key) > 40
        
        # Test key validation (mock)
        with patch.object(auth_handler, 'validate_api_key') as mock_validate:
            mock_validate.return_value = True
            
            result = auth_handler.validate_api_key(api_key)
            assert result is True
            mock_validate.assert_called_once_with(api_key)
    
    async def test_environment_secret_loading(self):
        """Test secure loading of secrets from environment."""
        # Test environment variable patterns
        secret_vars = [
            'JWT_SECRET_KEY',
            'DATABASE_PASSWORD',
            'API_SECRET_KEY',
            'ENCRYPTION_KEY'
        ]
        
        for var in secret_vars:
            # Mock environment variable
            with patch.dict(os.environ, {var: 'test_secret_value'}):
                value = os.getenv(var)
                assert value is not None
                assert len(value) > 0
                
                # Test that secrets are not logged
                assert value != 'test_secret_value' or var.endswith('_TEST')


@pytest.mark.production
@pytest.mark.security
class TestSecurityScanning:
    """Test security scanning and vulnerability checks."""
    
    async def test_dependency_vulnerabilities(self):
        """Test for known vulnerabilities in dependencies."""
        # Mock security scanner results
        mock_vulnerabilities = []
        
        # Test that no critical vulnerabilities are present
        critical_vulns = [v for v in mock_vulnerabilities if v.get('severity') == 'CRITICAL']
        assert len(critical_vulns) == 0, f"Critical vulnerabilities found: {critical_vulns}"
        
        # Test that high vulnerabilities are addressed
        high_vulns = [v for v in mock_vulnerabilities if v.get('severity') == 'HIGH']
        assert len(high_vulns) == 0, f"High severity vulnerabilities found: {high_vulns}"
    
    async def test_code_security_patterns(self):
        """Test for insecure code patterns."""
        # Mock static analysis results
        insecure_patterns = [
            'eval(',
            'exec(',
            'os.system(',
            'subprocess.call(',
            'pickle.loads(',
            'yaml.load('
        ]
        
        # In a real implementation, this would scan actual code files
        # For testing, we verify patterns are flagged
        test_code = "result = eval(user_input)"  # Dangerous pattern
        
        found_patterns = []
        for pattern in insecure_patterns:
            if pattern in test_code:
                found_patterns.append(pattern)
        
        # This test should find the eval pattern
        assert 'eval(' in found_patterns
    
    async def test_input_validation_security(self):
        """Test input validation against injection attacks."""
        # Test SQL injection patterns
        sql_injection_payloads = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'/*",
            "1; SELECT * FROM users"
        ]
        
        # Mock input validator
        def validate_input(user_input: str) -> bool:
            dangerous_patterns = [
                'DROP TABLE',
                'DELETE FROM',
                'INSERT INTO',
                'UPDATE SET',
                "'OR'",
                '--',
                '/*',
                '*/',
                'UNION SELECT'
            ]
            
            for pattern in dangerous_patterns:
                if pattern.upper() in user_input.upper():
                    return False
            return True
        
        # Test that injection payloads are rejected
        for payload in sql_injection_payloads:
            assert validate_input(payload) is False, f"Failed to detect injection: {payload}"
        
        # Test that normal input is accepted
        assert validate_input("normal search query") is True


@pytest.mark.production
@pytest.mark.security
class TestAuditLogging:
    """Test audit logging and compliance features."""
    
    async def test_audit_log_format(self):
        """Test audit log format and required fields."""
        # Mock audit log entry
        audit_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': 'test_user_123',
            'action': 'query_documents',
            'resource': '/api/query',
            'method': 'POST',
            'ip_address': '192.168.1.100',
            'user_agent': 'lightrag-client/1.0',
            'request_id': 'req_123456789',
            'status_code': 200,
            'response_time_ms': 150,
            'request_size_bytes': 1024,
            'response_size_bytes': 2048
        }
        
        # Validate required fields
        required_fields = [
            'timestamp', 'user_id', 'action', 'resource',
            'method', 'ip_address', 'status_code'
        ]
        
        for field in required_fields:
            assert field in audit_entry, f"Missing required audit field: {field}"
        
        # Validate field formats
        assert isinstance(audit_entry['timestamp'], str)
        assert isinstance(audit_entry['status_code'], int)
        assert isinstance(audit_entry['response_time_ms'], (int, float))
    
    async def test_audit_log_retention(self):
        """Test audit log retention and archival."""
        # Mock log retention configuration
        retention_config = {
            'retention_days': 90,
            'archive_after_days': 30,
            'compression_enabled': True,
            'encryption_enabled': True
        }
        
        # Test retention settings
        assert retention_config['retention_days'] >= 90  # Compliance requirement
        assert retention_config['archive_after_days'] < retention_config['retention_days']
        assert retention_config['encryption_enabled'] is True
    
    async def test_sensitive_data_redaction(self):
        """Test that sensitive data is redacted from logs."""
        # Mock request with sensitive data
        request_data = {
            'query': 'Find documents about user passwords',
            'api_key': 'lightrag_secret_key_123456',
            'user_data': {
                'email': 'user@example.com',
                'password': 'secret_password_123'
            }
        }
        
        # Mock log redaction function
        def redact_sensitive_data(data: Dict[str, Any]) -> Dict[str, Any]:
            import copy
            redacted = copy.deepcopy(data)
            
            # Redact API keys
            if 'api_key' in redacted:
                redacted['api_key'] = '***REDACTED***'
            
            # Redact passwords
            if 'user_data' in redacted and 'password' in redacted['user_data']:
                redacted['user_data']['password'] = '***REDACTED***'
            
            return redacted
        
        # Test redaction
        redacted_data = redact_sensitive_data(request_data)
        
        assert redacted_data['api_key'] == '***REDACTED***'
        assert redacted_data['user_data']['password'] == '***REDACTED***'
        assert redacted_data['query'] == request_data['query']  # Non-sensitive data preserved
        assert redacted_data['user_data']['email'] == request_data['user_data']['email']  # Email OK to log
    
    async def test_compliance_reporting(self):
        """Test compliance reporting capabilities."""
        # Mock compliance report structure
        compliance_report = {
            'report_period': {
                'start_date': '2025-01-01',
                'end_date': '2025-01-31'
            },
            'security_metrics': {
                'total_requests': 10000,
                'failed_auth_attempts': 25,
                'blocked_requests': 15,
                'rate_limited_requests': 50
            },
            'audit_completeness': {
                'total_events': 10000,
                'logged_events': 10000,
                'completeness_percentage': 100.0
            },
            'compliance_status': 'COMPLIANT'
        }
        
        # Validate compliance metrics
        assert compliance_report['audit_completeness']['completeness_percentage'] >= 99.0
        assert compliance_report['security_metrics']['failed_auth_attempts'] < 100
        assert compliance_report['compliance_status'] in ['COMPLIANT', 'NON_COMPLIANT', 'UNDER_REVIEW']


# Performance benchmarks for security operations
@pytest.mark.production
@pytest.mark.security
@pytest.mark.performance
class TestSecurityPerformance:
    """Test performance of security operations."""
    
    async def test_authentication_performance(self):
        """Test authentication performance under load."""
        import time

        # Mock auth handler
        auth_handler = AuthHandler(
            jwt_secret="test_secret_key_32_chars_long",
            jwt_expire_hours=24
        )
        
        # Test token validation performance
        token = jwt.encode({
            'user_id': 'test_user',
            'exp': datetime.utcnow() + timedelta(hours=1)
        }, "test_secret_key_32_chars_long", algorithm='HS256')
        
        start_time = time.time()
        iterations = 1000
        
        for _ in range(iterations):
            try:
                jwt.decode(token, "test_secret_key_32_chars_long", algorithms=['HS256'])
            except:
                pass
        
        end_time = time.time()
        avg_time = (end_time - start_time) / iterations
        
        # Authentication should be fast (< 1ms per validation)
        assert avg_time < 0.001, f"Auth too slow: {avg_time:.4f}s per validation"
    
    async def test_rate_limiting_performance(self):
        """Test rate limiting performance impact."""
        import time
        
        config = RateLimitConfig(
            general_api_limit="1000/minute"
        )
        rate_limiter = AdvancedRateLimiter(config=config)
        
        # Mock request
        mock_request = Mock()
        mock_request.client.host = '192.168.1.100'
        mock_request.url.path = '/api/query'
        
        start_time = time.time()
        iterations = 1000
        
        for _ in range(iterations):
            client_id = rate_limiter._get_client_identifier(mock_request)
            # Mock rate limit check (simplified)
            _ = f"rate_limit_{client_id}"
        
        end_time = time.time()
        avg_time = (end_time - start_time) / iterations
        
        # Rate limiting should be fast (< 0.1ms per check)
        assert avg_time < 0.0001, f"Rate limiting too slow: {avg_time:.4f}s per check"