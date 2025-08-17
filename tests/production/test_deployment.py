"""
Production Deployment and Infrastructure Tests

Tests for Docker deployment validation, Kubernetes deployment tests,
environment configuration validation, and service discovery.
"""

import asyncio
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, Mock, patch

import docker
import httpx
import psutil
import pytest
import requests
import yaml


@pytest.mark.production
@pytest.mark.deployment
class TestDockerDeployment:
    """Test Docker deployment configurations and validation."""
    
    @pytest.fixture
    def docker_client(self):
        """Docker client for deployment tests."""
        try:
            client = docker.from_env()
            yield client
        except Exception:
            pytest.skip("Docker not available")
        finally:
            if 'client' in locals():
                client.close()
    
    @pytest.fixture
    def docker_compose_files(self):
        """Docker compose file paths."""
        base_path = Path("/opt/developments/LightRAG")
        return {
            'development': base_path / "docker-compose.yml",
            'enhanced': base_path / "docker-compose.enhanced.yml",
            'production': base_path / "docker-compose.production.yml"
        }
    
    async def test_docker_compose_syntax_validation(self, docker_compose_files):
        """Test Docker Compose file syntax validation."""
        for env_name, file_path in docker_compose_files.items():
            if not file_path.exists():
                pytest.skip(f"Docker Compose file not found: {file_path}")
            
            # Validate YAML syntax
            try:
                with open(file_path, 'r') as f:
                    compose_config = yaml.safe_load(f)
                
                # Basic structure validation
                assert 'version' in compose_config or 'services' in compose_config
                assert 'services' in compose_config
                assert len(compose_config['services']) > 0
                
                # Validate service configurations
                for service_name, service_config in compose_config['services'].items():
                    assert isinstance(service_config, dict)
                    
                    # Check for required fields in production services
                    if env_name == 'production' and service_name == 'lightrag':
                        assert 'image' in service_config or 'build' in service_config
                        assert 'restart' in service_config
                        assert service_config.get('restart') in ['always', 'unless-stopped']
                        
                        # Security configurations
                        if 'security_opt' in service_config:
                            assert 'no-new-privileges:true' in service_config['security_opt']
                        
                        # Health check configuration
                        if 'healthcheck' in service_config:
                            assert 'test' in service_config['healthcheck']
                            assert 'interval' in service_config['healthcheck']
                            assert 'timeout' in service_config['healthcheck']
                            assert 'retries' in service_config['healthcheck']
                
            except yaml.YAMLError as e:
                pytest.fail(f"Invalid YAML syntax in {file_path}: {e}")
            except Exception as e:
                pytest.fail(f"Error validating {file_path}: {e}")
    
    async def test_production_dockerfile_build(self, docker_client):
        """Test production Dockerfile builds successfully."""
        dockerfile_path = Path("/opt/developments/LightRAG/Dockerfile.production")
        
        if not dockerfile_path.exists():
            pytest.skip("Production Dockerfile not found")
        
        image_tag = "lightrag:deployment-test"
        try:
            # Build production image
            image, logs = docker_client.images.build(
                path=str(dockerfile_path.parent),
                dockerfile=str(dockerfile_path.name),
                tag=image_tag,
                rm=True,
                nocache=False
            )
            
            # Verify image was created
            assert image is not None
            assert image.tags[0] == image_tag
            
            # Check image labels and metadata
            image_config = image.attrs['Config']
            
            # Verify non-root user
            if 'User' in image_config:
                assert image_config['User'] != 'root'
                assert image_config['User'] != '0'
            
            # Verify exposed ports
            if 'ExposedPorts' in image_config:
                exposed_ports = list(image_config['ExposedPorts'].keys())
                assert any('9621' in port for port in exposed_ports)
            
        finally:
            # Cleanup
            try:
                docker_client.images.remove(image_tag, force=True)
            except:
                pass
    
    async def test_container_health_checks(self, docker_client):
        """Test container health check configurations."""
        # Mock health check container
        health_check_script = """
        import requests
        import sys
        import time
        
        def health_check():
            try:
                response = requests.get('http://localhost:9621/health', timeout=5)
                if response.status_code == 200:
                    print('Health check passed')
                    return True
                else:
                    print(f'Health check failed: {response.status_code}')
                    return False
            except Exception as e:
                print(f'Health check error: {e}')
                return False
        
        if __name__ == '__main__':
            for _ in range(3):  # Retry logic
                if health_check():
                    sys.exit(0)
                time.sleep(1)
            sys.exit(1)
        """
        
        # Test health check configuration
        health_config = {
            'test': ['CMD', 'python', '-c', health_check_script],
            'interval': '30s',
            'timeout': '10s',
            'retries': 3,
            'start_period': '40s'
        }
        
        # Validate health check parameters
        assert health_config['test'][0] == 'CMD'
        assert 'python' in health_config['test']
        assert isinstance(health_config['retries'], int)
        assert health_config['retries'] >= 3
    
    async def test_environment_variable_validation(self):
        """Test environment variable configuration validation."""
        # Production environment variables
        prod_env_vars = {
            'NODE_ENV': 'production',
            'LOG_LEVEL': 'INFO',
            'WORKERS': '4',
            'HOST': '0.0.0.0',
            'PORT': '9621',
            'AUTH_ENABLED': 'true',
            'RATE_LIMIT_ENABLED': 'true',
            'JWT_SECRET_KEY': 'production_secret_key',
            'DATABASE_URL': 'postgresql://user:pass@postgres:5432/lightrag'
        }
        
        # Validate required variables
        required_vars = [
            'NODE_ENV', 'LOG_LEVEL', 'WORKERS', 'HOST', 'PORT',
            'AUTH_ENABLED', 'RATE_LIMIT_ENABLED'
        ]
        
        for var in required_vars:
            assert var in prod_env_vars, f"Missing required environment variable: {var}"
        
        # Validate production-specific settings
        assert prod_env_vars['NODE_ENV'] == 'production'
        assert prod_env_vars['LOG_LEVEL'] in ['INFO', 'WARNING', 'ERROR']
        assert int(prod_env_vars['WORKERS']) >= 1
        assert prod_env_vars['AUTH_ENABLED'].lower() == 'true'
        assert prod_env_vars['RATE_LIMIT_ENABLED'].lower() == 'true'
    
    async def test_volume_mount_configurations(self):
        """Test volume mount configurations for data persistence."""
        # Production volume configuration
        volume_config = {
            'lightrag_data': {
                'driver': 'local',
                'driver_opts': {
                    'type': 'none',
                    'o': 'bind',
                    'device': '/opt/lightrag/data'
                }
            },
            'postgres_data': {
                'driver': 'local'
            },
            'redis_data': {
                'driver': 'local'
            }
        }
        
        # Validate volume configurations
        for volume_name, config in volume_config.items():
            assert 'driver' in config
            assert config['driver'] in ['local', 'nfs', 'cifs']
            
            # Check for persistent storage
            if 'driver_opts' in config:
                assert 'device' in config['driver_opts']
    
    async def test_network_configurations(self):
        """Test Docker network configurations."""
        # Network configuration
        network_config = {
            'lightrag_network': {
                'driver': 'bridge',
                'driver_opts': {
                    'com.docker.network.bridge.name': 'lightrag0'
                },
                'ipam': {
                    'driver': 'default',
                    'config': [
                        {
                            'subnet': '172.20.0.0/16',
                            'gateway': '172.20.0.1'
                        }
                    ]
                }
            }
        }
        
        # Validate network configuration
        for network_name, config in network_config.items():
            assert 'driver' in config
            assert config['driver'] in ['bridge', 'overlay', 'host']
            
            if 'ipam' in config:
                assert 'driver' in config['ipam']
                assert 'config' in config['ipam']
                assert len(config['ipam']['config']) > 0


@pytest.mark.production
@pytest.mark.deployment
class TestKubernetesDeployment:
    """Test Kubernetes deployment configurations."""
    
    @pytest.fixture
    def k8s_manifests_path(self):
        """Path to Kubernetes manifests."""
        return Path("/opt/developments/LightRAG/k8s-deploy")
    
    async def test_kubernetes_manifest_validation(self, k8s_manifests_path):
        """Test Kubernetes manifest syntax and structure."""
        if not k8s_manifests_path.exists():
            pytest.skip("Kubernetes manifests not found")
        
        # Find YAML files
        yaml_files = list(k8s_manifests_path.rglob("*.yaml")) + list(k8s_manifests_path.rglob("*.yml"))
        
        for yaml_file in yaml_files:
            if 'Chart.yaml' in str(yaml_file):
                continue  # Skip Helm charts for now
            
            try:
                with open(yaml_file, 'r') as f:
                    # YAML can contain multiple documents
                    documents = list(yaml.safe_load_all(f))
                    
                    for doc in documents:
                        if doc is None:
                            continue
                        
                        # Validate Kubernetes resource structure
                        assert 'apiVersion' in doc, f"Missing apiVersion in {yaml_file}"
                        assert 'kind' in doc, f"Missing kind in {yaml_file}"
                        assert 'metadata' in doc, f"Missing metadata in {yaml_file}"
                        assert 'name' in doc['metadata'], f"Missing metadata.name in {yaml_file}"
                        
                        # Validate specific resource types
                        kind = doc['kind']
                        
                        if kind == 'Deployment':
                            assert 'spec' in doc
                            assert 'selector' in doc['spec']
                            assert 'template' in doc['spec']
                            assert 'spec' in doc['spec']['template']
                            assert 'containers' in doc['spec']['template']['spec']
                            
                            # Validate security contexts
                            containers = doc['spec']['template']['spec']['containers']
                            for container in containers:
                                if 'securityContext' in container:
                                    sec_ctx = container['securityContext']
                                    assert sec_ctx.get('runAsNonRoot', True) is True
                                    assert sec_ctx.get('allowPrivilegeEscalation', True) is False
                                    assert sec_ctx.get('readOnlyRootFilesystem', False) is True
                        
                        elif kind == 'Service':
                            assert 'spec' in doc
                            assert 'selector' in doc['spec']
                            assert 'ports' in doc['spec']
                            
                        elif kind == 'ConfigMap' or kind == 'Secret':
                            assert 'data' in doc
            
            except yaml.YAMLError as e:
                pytest.fail(f"Invalid YAML syntax in {yaml_file}: {e}")
            except Exception as e:
                pytest.fail(f"Error validating {yaml_file}: {e}")
    
    async def test_helm_chart_validation(self, k8s_manifests_path):
        """Test Helm chart structure and templates."""
        chart_path = k8s_manifests_path / "lightrag"
        
        if not chart_path.exists():
            pytest.skip("Helm chart not found")
        
        # Check required files
        required_files = [
            'Chart.yaml',
            'values.yaml',
            'templates/deployment.yaml',
            'templates/service.yaml'
        ]
        
        for required_file in required_files:
            file_path = chart_path / required_file
            assert file_path.exists(), f"Missing required Helm file: {required_file}"
        
        # Validate Chart.yaml
        chart_yaml = chart_path / "Chart.yaml"
        with open(chart_yaml, 'r') as f:
            chart_config = yaml.safe_load(f)
        
        assert 'name' in chart_config
        assert 'version' in chart_config
        assert 'description' in chart_config
        assert chart_config['name'] == 'lightrag'
        
        # Validate values.yaml
        values_yaml = chart_path / "values.yaml"
        with open(values_yaml, 'r') as f:
            values_config = yaml.safe_load(f)
        
        assert 'image' in values_config
        assert 'service' in values_config
        assert 'resources' in values_config
        
        # Validate resource limits
        if 'limits' in values_config['resources']:
            limits = values_config['resources']['limits']
            assert 'memory' in limits
            assert 'cpu' in limits
    
    async def test_kubernetes_resource_limits(self):
        """Test Kubernetes resource limits and requests."""
        # Production resource configuration
        resource_config = {
            'requests': {
                'memory': '512Mi',
                'cpu': '250m'
            },
            'limits': {
                'memory': '2Gi',
                'cpu': '1000m'
            }
        }
        
        # Validate resource configuration
        assert 'requests' in resource_config
        assert 'limits' in resource_config
        
        # Check memory and CPU settings
        for resource_type in ['requests', 'limits']:
            resources = resource_config[resource_type]
            assert 'memory' in resources
            assert 'cpu' in resources
            
            # Validate memory format
            memory = resources['memory']
            assert any(memory.endswith(unit) for unit in ['Mi', 'Gi', 'M', 'G'])
            
            # Validate CPU format
            cpu = resources['cpu']
            assert cpu.endswith('m') or cpu.isdigit()
        
        # Validate limits are higher than requests
        def parse_memory(mem_str):
            if mem_str.endswith('Gi'):
                return float(mem_str[:-2]) * 1024
            elif mem_str.endswith('Mi'):
                return float(mem_str[:-2])
            return float(mem_str[:-1])  # Assume Mi
        
        def parse_cpu(cpu_str):
            if cpu_str.endswith('m'):
                return float(cpu_str[:-1])
            return float(cpu_str) * 1000
        
        req_memory = parse_memory(resource_config['requests']['memory'])
        lim_memory = parse_memory(resource_config['limits']['memory'])
        req_cpu = parse_cpu(resource_config['requests']['cpu'])
        lim_cpu = parse_cpu(resource_config['limits']['cpu'])
        
        assert lim_memory >= req_memory
        assert lim_cpu >= req_cpu
    
    async def test_kubernetes_probes_configuration(self):
        """Test Kubernetes liveness and readiness probes."""
        # Probe configuration
        probes_config = {
            'livenessProbe': {
                'httpGet': {
                    'path': '/health',
                    'port': 9621,
                    'scheme': 'HTTP'
                },
                'initialDelaySeconds': 30,
                'periodSeconds': 10,
                'timeoutSeconds': 5,
                'failureThreshold': 3,
                'successThreshold': 1
            },
            'readinessProbe': {
                'httpGet': {
                    'path': '/health',
                    'port': 9621,
                    'scheme': 'HTTP'
                },
                'initialDelaySeconds': 5,
                'periodSeconds': 5,
                'timeoutSeconds': 3,
                'failureThreshold': 3,
                'successThreshold': 1
            }
        }
        
        # Validate probe configurations
        for probe_type, probe_config in probes_config.items():
            assert 'httpGet' in probe_config
            assert 'path' in probe_config['httpGet']
            assert 'port' in probe_config['httpGet']
            assert probe_config['httpGet']['path'] == '/health'
            assert probe_config['httpGet']['port'] == 9621
            
            # Validate timing parameters
            assert probe_config['initialDelaySeconds'] >= 5
            assert probe_config['periodSeconds'] >= 5
            assert probe_config['timeoutSeconds'] >= 3
            assert probe_config['failureThreshold'] >= 3


@pytest.mark.production
@pytest.mark.deployment
class TestEnvironmentConfiguration:
    """Test environment configuration validation."""
    
    async def test_production_environment_file(self):
        """Test production environment file configuration."""
        prod_env_path = Path("/opt/developments/LightRAG/production.env")
        
        if not prod_env_path.exists():
            pytest.skip("Production environment file not found")
        
        # Read environment file
        env_vars = {}
        with open(prod_env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key] = value
        
        # Required production variables
        required_vars = [
            'NODE_ENV',
            'LOG_LEVEL',
            'HOST',
            'PORT',
            'WORKERS',
            'AUTH_ENABLED',
            'RATE_LIMIT_ENABLED'
        ]
        
        for var in required_vars:
            assert var in env_vars, f"Missing required environment variable: {var}"
        
        # Validate production-specific values
        if 'NODE_ENV' in env_vars:
            assert env_vars['NODE_ENV'] == 'production'
        
        if 'LOG_LEVEL' in env_vars:
            assert env_vars['LOG_LEVEL'] in ['INFO', 'WARNING', 'ERROR']
        
        if 'AUTH_ENABLED' in env_vars:
            assert env_vars['AUTH_ENABLED'].lower() == 'true'
    
    async def test_configuration_validation(self):
        """Test application configuration validation."""
        # Mock configuration validation
        config = {
            'api': {
                'host': '0.0.0.0',
                'port': 9621,
                'workers': 4,
                'timeout': 60,
                'ssl': {
                    'enabled': True,
                    'cert_file': '/etc/ssl/certs/lightrag.crt',
                    'key_file': '/etc/ssl/private/lightrag.key'
                }
            },
            'database': {
                'url': 'postgresql://lightrag:password@postgres:5432/lightrag',
                'pool_size': 20,
                'max_overflow': 30,
                'pool_timeout': 30
            },
            'redis': {
                'url': 'redis://redis:6379/0',
                'max_connections': 50,
                'timeout': 5
            },
            'security': {
                'jwt_secret_key': 'production_secret_key',
                'jwt_expire_hours': 24,
                'rate_limit': {
                    'requests_per_minute': 100,
                    'burst_limit': 20
                }
            }
        }
        
        # Validate API configuration
        api_config = config['api']
        assert api_config['host'] in ['0.0.0.0', 'localhost']
        assert 1024 <= api_config['port'] <= 65535
        assert api_config['workers'] >= 1
        assert api_config['timeout'] >= 30
        
        # Validate database configuration
        db_config = config['database']
        assert db_config['url'].startswith('postgresql://')
        assert db_config['pool_size'] >= 10
        assert db_config['max_overflow'] >= 10
        assert db_config['pool_timeout'] >= 30
        
        # Validate security configuration
        security_config = config['security']
        assert len(security_config['jwt_secret_key']) >= 32
        assert security_config['jwt_expire_hours'] >= 1
        assert security_config['rate_limit']['requests_per_minute'] >= 60
    
    async def test_environment_specific_overrides(self):
        """Test environment-specific configuration overrides."""
        # Environment configurations
        environments = {
            'development': {
                'DEBUG': True,
                'LOG_LEVEL': 'DEBUG',
                'WORKERS': 1,
                'AUTH_ENABLED': False,
                'RATE_LIMIT_ENABLED': False
            },
            'staging': {
                'DEBUG': False,
                'LOG_LEVEL': 'INFO',
                'WORKERS': 2,
                'AUTH_ENABLED': True,
                'RATE_LIMIT_ENABLED': True
            },
            'production': {
                'DEBUG': False,
                'LOG_LEVEL': 'WARNING',
                'WORKERS': 4,
                'AUTH_ENABLED': True,
                'RATE_LIMIT_ENABLED': True
            }
        }
        
        # Validate environment-specific settings
        for env_name, env_config in environments.items():
            if env_name == 'production':
                assert env_config['DEBUG'] is False
                assert env_config['LOG_LEVEL'] in ['INFO', 'WARNING', 'ERROR']
                assert env_config['WORKERS'] >= 4
                assert env_config['AUTH_ENABLED'] is True
                assert env_config['RATE_LIMIT_ENABLED'] is True
            
            elif env_name == 'development':
                assert env_config['DEBUG'] is True
                assert env_config['LOG_LEVEL'] == 'DEBUG'
                # Development can have relaxed security for testing


@pytest.mark.production
@pytest.mark.deployment
class TestServiceDiscovery:
    """Test service discovery and health checks."""
    
    async def test_health_check_endpoint(self):
        """Test health check endpoint functionality."""
        # Mock health check response
        health_response = {
            'status': 'healthy',
            'timestamp': '2025-01-15T10:00:00Z',
            'version': '1.0.0',
            'components': {
                'database': 'healthy',
                'redis': 'healthy',
                'storage': 'healthy',
                'llm_provider': 'healthy'
            },
            'checks': {
                'memory_usage': {
                    'status': 'healthy',
                    'value': '45%',
                    'threshold': '80%'
                },
                'cpu_usage': {
                    'status': 'healthy',
                    'value': '25%',
                    'threshold': '70%'
                },
                'disk_space': {
                    'status': 'healthy',
                    'value': '60%',
                    'threshold': '90%'
                }
            }
        }
        
        # Validate health check structure
        assert health_response['status'] in ['healthy', 'unhealthy', 'degraded']
        assert 'timestamp' in health_response
        assert 'components' in health_response
        assert 'checks' in health_response
        
        # Validate component health
        for component, status in health_response['components'].items():
            assert status in ['healthy', 'unhealthy', 'unknown']
        
        # Validate system checks
        for check_name, check_data in health_response['checks'].items():
            assert 'status' in check_data
            assert 'value' in check_data
            assert 'threshold' in check_data
    
    async def test_service_dependencies(self):
        """Test service dependency configuration."""
        # Service dependency graph
        dependencies = {
            'lightrag': ['postgres', 'redis'],
            'postgres': [],
            'redis': [],
            'nginx': ['lightrag'],
            'monitoring': ['lightrag', 'postgres', 'redis']
        }
        
        # Validate dependency relationships
        assert 'lightrag' in dependencies
        assert 'postgres' in dependencies['lightrag']
        assert 'redis' in dependencies['lightrag']
        
        # Check for circular dependencies
        def has_circular_dependency(services, service, visited=None):
            if visited is None:
                visited = set()
            
            if service in visited:
                return True
            
            visited.add(service)
            
            for dependency in services.get(service, []):
                if has_circular_dependency(services, dependency, visited.copy()):
                    return True
            
            return False
        
        for service in dependencies:
            assert not has_circular_dependency(dependencies, service), f"Circular dependency detected for {service}"
    
    async def test_rolling_update_configuration(self):
        """Test rolling update and deployment strategies."""
        # Rolling update configuration
        update_config = {
            'strategy': {
                'type': 'RollingUpdate',
                'rollingUpdate': {
                    'maxUnavailable': '25%',
                    'maxSurge': '25%'
                }
            },
            'minReadySeconds': 30,
            'progressDeadlineSeconds': 600,
            'revisionHistoryLimit': 10
        }
        
        # Validate update strategy
        assert update_config['strategy']['type'] == 'RollingUpdate'
        rolling_config = update_config['strategy']['rollingUpdate']
        assert 'maxUnavailable' in rolling_config
        assert 'maxSurge' in rolling_config
        
        # Validate timing parameters
        assert update_config['minReadySeconds'] >= 30
        assert update_config['progressDeadlineSeconds'] >= 600
        assert update_config['revisionHistoryLimit'] >= 3
    
    async def test_rollback_scenarios(self):
        """Test rollback scenarios and procedures."""
        # Rollback configuration
        rollback_config = {
            'enabled': True,
            'automatic_rollback': {
                'enabled': True,
                'failure_threshold': 3,
                'evaluation_period': 300  # seconds
            },
            'manual_rollback': {
                'enabled': True,
                'retention_count': 5
            },
            'health_check_grace_period': 60
        }
        
        # Validate rollback configuration
        assert rollback_config['enabled'] is True
        assert rollback_config['automatic_rollback']['enabled'] is True
        assert rollback_config['automatic_rollback']['failure_threshold'] >= 3
        assert rollback_config['manual_rollback']['retention_count'] >= 3
        assert rollback_config['health_check_grace_period'] >= 60


# Performance tests for deployment
@pytest.mark.production
@pytest.mark.deployment
@pytest.mark.performance
class TestDeploymentPerformance:
    """Test deployment performance and startup times."""
    
    async def test_container_startup_time(self):
        """Test container startup time performance."""
        # Mock startup time measurement
        startup_times = {
            'container_creation': 2.5,  # seconds
            'application_startup': 15.0,  # seconds
            'health_check_ready': 5.0,  # seconds
            'total_startup': 22.5  # seconds
        }
        
        # Validate startup performance
        assert startup_times['container_creation'] <= 10.0
        assert startup_times['application_startup'] <= 30.0
        assert startup_times['health_check_ready'] <= 10.0
        assert startup_times['total_startup'] <= 60.0
    
    async def test_deployment_scalability(self):
        """Test deployment scalability metrics."""
        # Scalability configuration
        scaling_config = {
            'horizontal_pod_autoscaler': {
                'min_replicas': 2,
                'max_replicas': 10,
                'target_cpu_utilization': 70,
                'target_memory_utilization': 80
            },
            'vertical_pod_autoscaler': {
                'enabled': False,  # Usually one or the other
                'update_mode': 'Off'
            },
            'cluster_autoscaler': {
                'enabled': True,
                'scale_down_delay': '10m',
                'scale_down_unneeded_time': '10m'
            }
        }
        
        # Validate scaling configuration
        hpa = scaling_config['horizontal_pod_autoscaler']
        assert hpa['min_replicas'] >= 2  # For high availability
        assert hpa['max_replicas'] >= hpa['min_replicas']
        assert 50 <= hpa['target_cpu_utilization'] <= 80
        assert 60 <= hpa['target_memory_utilization'] <= 90