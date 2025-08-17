"""
Production Container and Orchestration Tests

Tests for container image security scanning, resource limits and quotas,
pod autoscaling tests, service mesh integration, and multi-region deployment tests.
"""

import asyncio
import json
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import docker
import pytest
import yaml


@dataclass
class ContainerScanResult:
    """Container security scan result."""
    image_name: str
    scan_timestamp: datetime
    vulnerabilities: Dict[str, int]  # severity -> count
    total_vulnerabilities: int
    critical_vulnerabilities: int
    high_vulnerabilities: int
    compliance_score: float
    scan_duration: float


@dataclass
class ResourceQuota:
    """Kubernetes resource quota definition."""
    name: str
    namespace: str
    cpu_limit: str
    memory_limit: str
    storage_limit: str
    pod_count_limit: int
    current_usage: Dict[str, Any]


@pytest.mark.production
@pytest.mark.containers
class TestContainerSecurity:
    """Test container image security scanning and hardening."""
    
    @pytest.fixture
    def security_scanner(self):
        """Mock container security scanner."""
        
        class MockSecurityScanner:
            def __init__(self):
                self.scan_results = {}
                self.vulnerability_database = self._load_mock_vulnerability_db()
            
            def _load_mock_vulnerability_db(self) -> Dict[str, Any]:
                """Load mock vulnerability database."""
                return {
                    'python:3.11-slim': {
                        'critical': 0,
                        'high': 2,
                        'medium': 5,
                        'low': 12,
                        'vulnerabilities': [
                            {
                                'id': 'CVE-2023-1234',
                                'severity': 'high',
                                'package': 'openssl',
                                'description': 'Buffer overflow in OpenSSL',
                                'fixed_version': '1.1.1t'
                            },
                            {
                                'id': 'CVE-2023-5678',
                                'severity': 'high',
                                'package': 'curl',
                                'description': 'Remote code execution in curl',
                                'fixed_version': '7.88.1'
                            }
                        ]
                    },
                    'lightrag:latest': {
                        'critical': 1,
                        'high': 3,
                        'medium': 8,
                        'low': 15,
                        'vulnerabilities': [
                            {
                                'id': 'CVE-2023-9999',
                                'severity': 'critical',
                                'package': 'numpy',
                                'description': 'Critical vulnerability in NumPy',
                                'fixed_version': '1.24.3'
                            }
                        ]
                    },
                    'lightrag:production': {
                        'critical': 0,
                        'high': 0,
                        'medium': 2,
                        'low': 5,
                        'vulnerabilities': []
                    }
                }
            
            async def scan_image(self, image_name: str) -> ContainerScanResult:
                """Scan container image for vulnerabilities."""
                scan_start = time.time()
                
                # Simulate scan delay
                await asyncio.sleep(0.1)
                
                # Get vulnerability data for image
                vuln_data = self.vulnerability_database.get(image_name, {
                    'critical': 0, 'high': 0, 'medium': 1, 'low': 2,
                    'vulnerabilities': []
                })
                
                vulnerabilities = {
                    'critical': vuln_data['critical'],
                    'high': vuln_data['high'],
                    'medium': vuln_data['medium'],
                    'low': vuln_data['low']
                }
                
                total_vulns = sum(vulnerabilities.values())
                critical_vulns = vulnerabilities['critical']
                high_vulns = vulnerabilities['high']
                
                # Calculate compliance score (0-100)
                compliance_score = max(0, 100 - (critical_vulns * 25 + high_vulns * 10 + vulnerabilities['medium'] * 2))
                
                scan_duration = time.time() - scan_start
                
                result = ContainerScanResult(
                    image_name=image_name,
                    scan_timestamp=datetime.utcnow(),
                    vulnerabilities=vulnerabilities,
                    total_vulnerabilities=total_vulns,
                    critical_vulnerabilities=critical_vulns,
                    high_vulnerabilities=high_vulns,
                    compliance_score=compliance_score,
                    scan_duration=scan_duration
                )
                
                self.scan_results[image_name] = result
                return result
            
            def get_scan_history(self, image_name: str) -> List[ContainerScanResult]:
                """Get scan history for an image."""
                return [self.scan_results.get(image_name)] if image_name in self.scan_results else []
            
            def generate_security_report(self) -> Dict[str, Any]:
                """Generate overall security report."""
                total_images = len(self.scan_results)
                if total_images == 0:
                    return {'status': 'no_scans_performed'}
                
                total_critical = sum(result.critical_vulnerabilities for result in self.scan_results.values())
                total_high = sum(result.high_vulnerabilities for result in self.scan_results.values())
                avg_compliance = sum(result.compliance_score for result in self.scan_results.values()) / total_images
                
                return {
                    'total_images_scanned': total_images,
                    'total_critical_vulnerabilities': total_critical,
                    'total_high_vulnerabilities': total_high,
                    'average_compliance_score': avg_compliance,
                    'compliant_images': sum(1 for result in self.scan_results.values() if result.compliance_score >= 80),
                    'scan_timestamp': datetime.utcnow()
                }
        
        return MockSecurityScanner()
    
    async def test_base_image_security_scan(self, security_scanner):
        """Test security scanning of base images."""
        
        base_images = [
            'python:3.11-slim',
            'nginx:alpine',
            'postgres:15-alpine',
            'redis:7-alpine'
        ]
        
        scan_results = []
        for image in base_images:
            result = await security_scanner.scan_image(image)
            scan_results.append(result)
        
        # Validate scan results
        for result in scan_results:
            assert result.scan_timestamp is not None
            assert result.scan_duration > 0
            assert result.total_vulnerabilities >= 0
            assert 0 <= result.compliance_score <= 100
            
            # Critical security requirements for production
            assert result.critical_vulnerabilities == 0, f"Critical vulnerabilities found in {result.image_name}"
            assert result.high_vulnerabilities <= 3, f"Too many high vulnerabilities in {result.image_name}"
            assert result.compliance_score >= 70, f"Compliance score too low for {result.image_name}: {result.compliance_score}"
    
    async def test_application_image_security_scan(self, security_scanner):
        """Test security scanning of application images."""
        
        # Test different versions of application image
        app_images = [
            'lightrag:latest',
            'lightrag:production',
            'lightrag:v1.0.0'
        ]
        
        scan_results = []
        for image in app_images:
            result = await security_scanner.scan_image(image)
            scan_results.append(result)
        
        # Production image should have highest security standards
        production_result = next((r for r in scan_results if 'production' in r.image_name), None)
        assert production_result is not None
        
        # Production image security requirements
        assert production_result.critical_vulnerabilities == 0, "Production image has critical vulnerabilities"
        assert production_result.high_vulnerabilities == 0, "Production image has high vulnerabilities"
        assert production_result.compliance_score >= 90, f"Production compliance too low: {production_result.compliance_score}"
        
        # Latest/development images can have some vulnerabilities but should be managed
        for result in scan_results:
            if 'production' not in result.image_name:
                assert result.critical_vulnerabilities <= 1, f"Too many critical vulnerabilities in {result.image_name}"
                assert result.high_vulnerabilities <= 5, f"Too many high vulnerabilities in {result.image_name}"
    
    async def test_vulnerability_remediation_tracking(self, security_scanner):
        """Test vulnerability remediation tracking."""
        
        # Scan image with known vulnerabilities
        vulnerable_image = 'lightrag:latest'
        initial_scan = await security_scanner.scan_image(vulnerable_image)
        
        # Simulate remediation by scanning "fixed" version
        remediated_image = 'lightrag:production'
        remediated_scan = await security_scanner.scan_image(remediated_image)
        
        # Validate remediation effectiveness
        assert remediated_scan.critical_vulnerabilities < initial_scan.critical_vulnerabilities
        assert remediated_scan.high_vulnerabilities < initial_scan.high_vulnerabilities
        assert remediated_scan.compliance_score > initial_scan.compliance_score
        
        # Generate security report
        security_report = security_scanner.generate_security_report()
        
        assert security_report['total_images_scanned'] == 2
        assert security_report['average_compliance_score'] > 70
        assert security_report['compliant_images'] >= 1
    
    async def test_continuous_security_monitoring(self, security_scanner):
        """Test continuous security monitoring setup."""
        
        # Simulate regular scanning schedule
        images_to_monitor = [
            'lightrag:production',
            'nginx:alpine',
            'postgres:15-alpine'
        ]
        
        # Perform initial scans
        baseline_results = {}
        for image in images_to_monitor:
            result = await security_scanner.scan_image(image)
            baseline_results[image] = result
        
        # Simulate passage of time and re-scanning
        await asyncio.sleep(0.1)
        
        # Re-scan to check for new vulnerabilities
        current_results = {}
        for image in images_to_monitor:
            result = await security_scanner.scan_image(image)
            current_results[image] = result
        
        # Validate monitoring capabilities
        for image in images_to_monitor:
            baseline = baseline_results[image]
            current = current_results[image]
            
            # Check for security regression
            assert current.critical_vulnerabilities <= baseline.critical_vulnerabilities, \
                f"Security regression in {image}: critical vulnerabilities increased"
            
            # Production image should maintain high standards
            if 'production' in image:
                assert current.compliance_score >= 90, \
                    f"Production image compliance degraded: {current.compliance_score}"


@pytest.mark.production
@pytest.mark.containers
class TestResourceLimitsAndQuotas:
    """Test Kubernetes resource limits and quotas."""
    
    @pytest.fixture
    def resource_manager(self):
        """Mock Kubernetes resource manager."""
        
        class MockResourceManager:
            def __init__(self):
                self.quotas = {}
                self.pod_specs = {}
                self.resource_usage = {}
            
            def create_resource_quota(self, quota: ResourceQuota):
                """Create a resource quota."""
                self.quotas[f"{quota.namespace}:{quota.name}"] = quota
            
            def set_pod_resource_limits(self, pod_name: str, namespace: str, 
                                      cpu_request: str, cpu_limit: str,
                                      memory_request: str, memory_limit: str):
                """Set resource limits for a pod."""
                self.pod_specs[f"{namespace}:{pod_name}"] = {
                    'resources': {
                        'requests': {
                            'cpu': cpu_request,
                            'memory': memory_request
                        },
                        'limits': {
                            'cpu': cpu_limit,
                            'memory': memory_limit
                        }
                    }
                }
            
            def simulate_resource_usage(self, namespace: str, pod_count: int, 
                                      avg_cpu_usage: float, avg_memory_usage_mb: int):
                """Simulate resource usage."""
                self.resource_usage[namespace] = {
                    'pod_count': pod_count,
                    'cpu_usage': avg_cpu_usage * pod_count,
                    'memory_usage_mb': avg_memory_usage_mb * pod_count,
                    'timestamp': datetime.utcnow()
                }
            
            def check_quota_compliance(self, namespace: str) -> Dict[str, Any]:
                """Check resource quota compliance."""
                quota_key = f"{namespace}:resource-quota"
                if quota_key not in self.quotas:
                    return {'status': 'no_quota_defined'}
                
                quota = self.quotas[quota_key]
                usage = self.resource_usage.get(namespace, {})
                
                # Parse resource values (simplified)
                def parse_cpu(cpu_str):
                    if cpu_str.endswith('m'):
                        return float(cpu_str[:-1]) / 1000
                    return float(cpu_str)
                
                def parse_memory(mem_str):
                    if mem_str.endswith('Gi'):
                        return float(mem_str[:-2]) * 1024
                    elif mem_str.endswith('Mi'):
                        return float(mem_str[:-2])
                    return float(mem_str)
                
                quota_cpu = parse_cpu(quota.cpu_limit)
                quota_memory = parse_memory(quota.memory_limit)
                
                used_cpu = usage.get('cpu_usage', 0)
                used_memory = usage.get('memory_usage_mb', 0)
                used_pods = usage.get('pod_count', 0)
                
                return {
                    'namespace': namespace,
                    'quota_compliance': {
                        'cpu': {
                            'limit': quota_cpu,
                            'used': used_cpu,
                            'percentage': (used_cpu / quota_cpu) * 100 if quota_cpu > 0 else 0
                        },
                        'memory': {
                            'limit_mb': quota_memory,
                            'used_mb': used_memory,
                            'percentage': (used_memory / quota_memory) * 100 if quota_memory > 0 else 0
                        },
                        'pods': {
                            'limit': quota.pod_count_limit,
                            'used': used_pods,
                            'percentage': (used_pods / quota.pod_count_limit) * 100 if quota.pod_count_limit > 0 else 0
                        }
                    },
                    'violations': []
                }
            
            def get_resource_recommendations(self, namespace: str) -> Dict[str, Any]:
                """Get resource optimization recommendations."""
                usage = self.resource_usage.get(namespace, {})
                quota_check = self.check_quota_compliance(namespace)
                
                recommendations = []
                
                if quota_check.get('quota_compliance'):
                    compliance = quota_check['quota_compliance']
                    
                    # CPU recommendations
                    cpu_usage = compliance['cpu']['percentage']
                    if cpu_usage > 80:
                        recommendations.append({
                            'type': 'scale_up',
                            'resource': 'cpu',
                            'current_usage': cpu_usage,
                            'recommendation': 'Increase CPU quota or optimize applications'
                        })
                    elif cpu_usage < 20:
                        recommendations.append({
                            'type': 'scale_down',
                            'resource': 'cpu',
                            'current_usage': cpu_usage,
                            'recommendation': 'Consider reducing CPU allocation'
                        })
                    
                    # Memory recommendations
                    memory_usage = compliance['memory']['percentage']
                    if memory_usage > 80:
                        recommendations.append({
                            'type': 'scale_up',
                            'resource': 'memory',
                            'current_usage': memory_usage,
                            'recommendation': 'Increase memory quota or optimize applications'
                        })
                    elif memory_usage < 20:
                        recommendations.append({
                            'type': 'scale_down',
                            'resource': 'memory',
                            'current_usage': memory_usage,
                            'recommendation': 'Consider reducing memory allocation'
                        })
                
                return {
                    'namespace': namespace,
                    'recommendations': recommendations,
                    'optimization_score': max(0, 100 - len(recommendations) * 20)
                }
        
        return MockResourceManager()
    
    async def test_production_resource_quotas(self, resource_manager):
        """Test production resource quota configuration."""
        
        # Define production resource quotas
        production_quotas = [
            ResourceQuota(
                name='lightrag-production-quota',
                namespace='lightrag-production',
                cpu_limit='16',      # 16 cores
                memory_limit='32Gi', # 32 GB
                storage_limit='500Gi', # 500 GB
                pod_count_limit=50,
                current_usage={}
            ),
            ResourceQuota(
                name='monitoring-quota',
                namespace='monitoring',
                cpu_limit='4',       # 4 cores
                memory_limit='8Gi',  # 8 GB
                storage_limit='100Gi', # 100 GB
                pod_count_limit=20,
                current_usage={}
            )
        ]
        
        # Create resource quotas
        for quota in production_quotas:
            resource_manager.create_resource_quota(quota)
        
        # Simulate resource usage
        resource_manager.simulate_resource_usage(
            namespace='lightrag-production',
            pod_count=10,
            avg_cpu_usage=0.8,   # 0.8 cores per pod
            avg_memory_usage_mb=1536  # 1.5 GB per pod
        )
        
        resource_manager.simulate_resource_usage(
            namespace='monitoring',
            pod_count=5,
            avg_cpu_usage=0.4,   # 0.4 cores per pod
            avg_memory_usage_mb=512   # 512 MB per pod
        )
        
        # Check quota compliance
        production_compliance = resource_manager.check_quota_compliance('lightrag-production')
        monitoring_compliance = resource_manager.check_quota_compliance('monitoring')
        
        # Validate production quota compliance
        prod_quota = production_compliance['quota_compliance']
        assert prod_quota['cpu']['percentage'] < 80, f"Production CPU usage too high: {prod_quota['cpu']['percentage']}%"
        assert prod_quota['memory']['percentage'] < 80, f"Production memory usage too high: {prod_quota['memory']['percentage']}%"
        assert prod_quota['pods']['percentage'] < 80, f"Production pod count too high: {prod_quota['pods']['percentage']}%"
        
        # Validate monitoring quota compliance
        mon_quota = monitoring_compliance['quota_compliance']
        assert mon_quota['cpu']['percentage'] < 80, f"Monitoring CPU usage too high: {mon_quota['cpu']['percentage']}%"
        assert mon_quota['memory']['percentage'] < 80, f"Monitoring memory usage too high: {mon_quota['memory']['percentage']}%"
    
    async def test_pod_resource_limits(self, resource_manager):
        """Test individual pod resource limits."""
        
        # Set resource limits for different pod types
        pod_configurations = [
            {
                'name': 'lightrag-api',
                'namespace': 'lightrag-production',
                'cpu_request': '500m',
                'cpu_limit': '2000m',
                'memory_request': '1Gi',
                'memory_limit': '4Gi'
            },
            {
                'name': 'lightrag-worker',
                'namespace': 'lightrag-production',
                'cpu_request': '1000m',
                'cpu_limit': '4000m',
                'memory_request': '2Gi',
                'memory_limit': '8Gi'
            },
            {
                'name': 'postgres',
                'namespace': 'lightrag-production',
                'cpu_request': '1000m',
                'cpu_limit': '2000m',
                'memory_request': '4Gi',
                'memory_limit': '8Gi'
            }
        ]
        
        # Configure pod resource limits
        for config in pod_configurations:
            resource_manager.set_pod_resource_limits(
                pod_name=config['name'],
                namespace=config['namespace'],
                cpu_request=config['cpu_request'],
                cpu_limit=config['cpu_limit'],
                memory_request=config['memory_request'],
                memory_limit=config['memory_limit']
            )
        
        # Validate pod configurations
        for config in pod_configurations:
            pod_key = f"{config['namespace']}:{config['name']}"
            pod_spec = resource_manager.pod_specs[pod_key]
            
            resources = pod_spec['resources']
            
            # Validate requests are less than limits
            cpu_request_val = float(resources['requests']['cpu'][:-1]) if resources['requests']['cpu'].endswith('m') else float(resources['requests']['cpu']) * 1000
            cpu_limit_val = float(resources['limits']['cpu'][:-1]) if resources['limits']['cpu'].endswith('m') else float(resources['limits']['cpu']) * 1000
            
            assert cpu_request_val <= cpu_limit_val, f"CPU request exceeds limit for {config['name']}"
            
            # Validate memory format and limits
            assert resources['requests']['memory'].endswith('Gi'), f"Invalid memory request format for {config['name']}"
            assert resources['limits']['memory'].endswith('Gi'), f"Invalid memory limit format for {config['name']}"
    
    async def test_resource_optimization_recommendations(self, resource_manager):
        """Test resource optimization recommendations."""
        
        # Create quota for testing
        test_quota = ResourceQuota(
            name='test-quota',
            namespace='test-namespace',
            cpu_limit='8',
            memory_limit='16Gi',
            storage_limit='100Gi',
            pod_count_limit=20,
            current_usage={}
        )
        
        resource_manager.create_resource_quota(test_quota)
        
        # Test high resource usage scenario
        resource_manager.simulate_resource_usage(
            namespace='test-namespace',
            pod_count=15,
            avg_cpu_usage=0.5,    # High CPU usage (7.5/8 cores)
            avg_memory_usage_mb=1000  # High memory usage (15GB/16GB)
        )
        
        recommendations = resource_manager.get_resource_recommendations('test-namespace')
        
        # Should recommend scaling up
        scale_up_recommendations = [r for r in recommendations['recommendations'] if r['type'] == 'scale_up']
        assert len(scale_up_recommendations) >= 1, "Should recommend scaling up for high resource usage"
        
        # Test low resource usage scenario
        resource_manager.simulate_resource_usage(
            namespace='test-namespace',
            pod_count=2,
            avg_cpu_usage=0.1,    # Low CPU usage
            avg_memory_usage_mb=200   # Low memory usage
        )
        
        recommendations = resource_manager.get_resource_optimization_recommendations('test-namespace')
        
        # Should recommend scaling down
        scale_down_recommendations = [r for r in recommendations['recommendations'] if r['type'] == 'scale_down']
        assert len(scale_down_recommendations) >= 1, "Should recommend scaling down for low resource usage"
        
        # Optimization score should be reasonable
        assert 0 <= recommendations['optimization_score'] <= 100


@pytest.mark.production
@pytest.mark.containers
class TestPodAutoscaling:
    """Test Kubernetes pod autoscaling functionality."""
    
    @pytest.fixture
    def autoscaler(self):
        """Mock Kubernetes Horizontal Pod Autoscaler."""
        
        class MockHPA:
            def __init__(self):
                self.hpa_configs = {}
                self.pod_metrics = {}
                self.scaling_events = []
                self.current_replicas = {}
            
            def create_hpa(self, name: str, namespace: str, target_deployment: str,
                          min_replicas: int, max_replicas: int,
                          target_cpu_percentage: int, target_memory_percentage: int = None):
                """Create HPA configuration."""
                
                hpa_config = {
                    'name': name,
                    'namespace': namespace,
                    'target_deployment': target_deployment,
                    'min_replicas': min_replicas,
                    'max_replicas': max_replicas,
                    'target_cpu_percentage': target_cpu_percentage,
                    'target_memory_percentage': target_memory_percentage,
                    'enabled': True,
                    'created_at': datetime.utcnow()
                }
                
                self.hpa_configs[f"{namespace}:{name}"] = hpa_config
                self.current_replicas[f"{namespace}:{target_deployment}"] = min_replicas
            
            def update_pod_metrics(self, namespace: str, deployment: str,
                                 cpu_percentage: float, memory_percentage: float,
                                 current_replicas: int):
                """Update pod metrics for autoscaling decisions."""
                
                self.pod_metrics[f"{namespace}:{deployment}"] = {
                    'cpu_percentage': cpu_percentage,
                    'memory_percentage': memory_percentage,
                    'current_replicas': current_replicas,
                    'timestamp': datetime.utcnow()
                }
            
            async def evaluate_scaling(self, namespace: str, deployment: str) -> Dict[str, Any]:
                """Evaluate if scaling is needed."""
                
                # Find HPA config for deployment
                hpa_config = None
                for hpa_key, config in self.hpa_configs.items():
                    if config['target_deployment'] == deployment and config['namespace'] == namespace:
                        hpa_config = config
                        break
                
                if not hpa_config or not hpa_config['enabled']:
                    return {'scaling_action': 'none', 'reason': 'no_hpa_configured'}
                
                metrics_key = f"{namespace}:{deployment}"
                if metrics_key not in self.pod_metrics:
                    return {'scaling_action': 'none', 'reason': 'no_metrics_available'}
                
                metrics = self.pod_metrics[metrics_key]
                current_replicas = self.current_replicas.get(metrics_key, hpa_config['min_replicas'])
                
                # Scaling logic
                cpu_percentage = metrics['cpu_percentage']
                target_cpu = hpa_config['target_cpu_percentage']
                
                scaling_decision = {
                    'namespace': namespace,
                    'deployment': deployment,
                    'current_replicas': current_replicas,
                    'cpu_usage': cpu_percentage,
                    'target_cpu': target_cpu,
                    'scaling_action': 'none',
                    'target_replicas': current_replicas,
                    'reason': ''
                }
                
                # Scale up conditions
                if cpu_percentage > target_cpu + 10:  # 10% buffer
                    new_replicas = min(
                        current_replicas + max(1, current_replicas // 2),  # Scale up by 50%
                        hpa_config['max_replicas']
                    )
                    if new_replicas > current_replicas:
                        scaling_decision['scaling_action'] = 'scale_up'
                        scaling_decision['target_replicas'] = new_replicas
                        scaling_decision['reason'] = f'CPU usage {cpu_percentage}% > target {target_cpu}%'
                
                # Scale down conditions
                elif cpu_percentage < target_cpu - 20:  # 20% buffer for scale down
                    new_replicas = max(
                        current_replicas - max(1, current_replicas // 4),  # Scale down by 25%
                        hpa_config['min_replicas']
                    )
                    if new_replicas < current_replicas:
                        scaling_decision['scaling_action'] = 'scale_down'
                        scaling_decision['target_replicas'] = new_replicas
                        scaling_decision['reason'] = f'CPU usage {cpu_percentage}% < target {target_cpu}%'
                
                return scaling_decision
            
            async def execute_scaling(self, scaling_decision: Dict[str, Any]) -> Dict[str, Any]:
                """Execute scaling action."""
                
                if scaling_decision['scaling_action'] == 'none':
                    return {'status': 'no_action_needed'}
                
                namespace = scaling_decision['namespace']
                deployment = scaling_decision['deployment']
                target_replicas = scaling_decision['target_replicas']
                
                # Update current replicas
                metrics_key = f"{namespace}:{deployment}"
                self.current_replicas[metrics_key] = target_replicas
                
                # Record scaling event
                scaling_event = {
                    'timestamp': datetime.utcnow(),
                    'namespace': namespace,
                    'deployment': deployment,
                    'action': scaling_decision['scaling_action'],
                    'from_replicas': scaling_decision['current_replicas'],
                    'to_replicas': target_replicas,
                    'reason': scaling_decision['reason'],
                    'cpu_usage': scaling_decision['cpu_usage']
                }
                
                self.scaling_events.append(scaling_event)
                
                return {
                    'status': 'scaling_executed',
                    'event': scaling_event
                }
            
            def get_scaling_history(self, namespace: str = None, deployment: str = None) -> List[Dict[str, Any]]:
                """Get scaling event history."""
                events = self.scaling_events
                
                if namespace:
                    events = [e for e in events if e['namespace'] == namespace]
                if deployment:
                    events = [e for e in events if e['deployment'] == deployment]
                
                return sorted(events, key=lambda x: x['timestamp'], reverse=True)
        
        return MockHPA()
    
    async def test_hpa_configuration(self, autoscaler):
        """Test HPA configuration for different services."""
        
        # Configure HPA for different services
        hpa_configs = [
            {
                'name': 'lightrag-api-hpa',
                'namespace': 'lightrag-production',
                'target_deployment': 'lightrag-api',
                'min_replicas': 2,
                'max_replicas': 10,
                'target_cpu_percentage': 70
            },
            {
                'name': 'lightrag-worker-hpa',
                'namespace': 'lightrag-production',
                'target_deployment': 'lightrag-worker',
                'min_replicas': 1,
                'max_replicas': 20,
                'target_cpu_percentage': 80
            }
        ]
        
        # Create HPA configurations
        for config in hpa_configs:
            autoscaler.create_hpa(**config)
        
        # Validate HPA configurations
        assert len(autoscaler.hpa_configs) == len(hpa_configs)
        
        for config in hpa_configs:
            hpa_key = f"{config['namespace']}:{config['name']}"
            stored_config = autoscaler.hpa_configs[hpa_key]
            
            assert stored_config['min_replicas'] >= 1
            assert stored_config['max_replicas'] > stored_config['min_replicas']
            assert 50 <= stored_config['target_cpu_percentage'] <= 90
            assert stored_config['enabled'] is True
    
    async def test_scale_up_scenario(self, autoscaler):
        """Test pod scaling up under high load."""
        
        # Configure HPA
        autoscaler.create_hpa(
            name='api-hpa',
            namespace='production',
            target_deployment='lightrag-api',
            min_replicas=2,
            max_replicas=8,
            target_cpu_percentage=70
        )
        
        # Simulate high CPU usage
        autoscaler.update_pod_metrics(
            namespace='production',
            deployment='lightrag-api',
            cpu_percentage=85,  # High CPU usage
            memory_percentage=60,
            current_replicas=2
        )
        
        # Evaluate scaling
        scaling_decision = await autoscaler.evaluate_scaling('production', 'lightrag-api')
        
        # Should decide to scale up
        assert scaling_decision['scaling_action'] == 'scale_up'
        assert scaling_decision['target_replicas'] > scaling_decision['current_replicas']
        assert scaling_decision['target_replicas'] <= 8  # Max replicas
        
        # Execute scaling
        scaling_result = await autoscaler.execute_scaling(scaling_decision)
        assert scaling_result['status'] == 'scaling_executed'
        
        # Verify scaling event recorded
        scaling_events = autoscaler.get_scaling_history('production', 'lightrag-api')
        assert len(scaling_events) == 1
        assert scaling_events[0]['action'] == 'scale_up'
        assert scaling_events[0]['cpu_usage'] == 85
    
    async def test_scale_down_scenario(self, autoscaler):
        """Test pod scaling down under low load."""
        
        # Configure HPA
        autoscaler.create_hpa(
            name='worker-hpa',
            namespace='production',
            target_deployment='lightrag-worker',
            min_replicas=1,
            max_replicas=10,
            target_cpu_percentage=80
        )
        
        # Set initial replicas to higher number
        autoscaler.current_replicas['production:lightrag-worker'] = 6
        
        # Simulate low CPU usage
        autoscaler.update_pod_metrics(
            namespace='production',
            deployment='lightrag-worker',
            cpu_percentage=35,  # Low CPU usage
            memory_percentage=40,
            current_replicas=6
        )
        
        # Evaluate scaling
        scaling_decision = await autoscaler.evaluate_scaling('production', 'lightrag-worker')
        
        # Should decide to scale down
        assert scaling_decision['scaling_action'] == 'scale_down'
        assert scaling_decision['target_replicas'] < scaling_decision['current_replicas']
        assert scaling_decision['target_replicas'] >= 1  # Min replicas
        
        # Execute scaling
        scaling_result = await autoscaler.execute_scaling(scaling_decision)
        assert scaling_result['status'] == 'scaling_executed'
        
        # Verify scaling event
        scaling_events = autoscaler.get_scaling_history('production', 'lightrag-worker')
        assert len(scaling_events) == 1
        assert scaling_events[0]['action'] == 'scale_down'
        assert scaling_events[0]['cpu_usage'] == 35
    
    async def test_scaling_limits_and_boundaries(self, autoscaler):
        """Test scaling respects configured limits and boundaries."""
        
        # Configure HPA with strict limits
        autoscaler.create_hpa(
            name='boundary-test-hpa',
            namespace='test',
            target_deployment='test-app',
            min_replicas=2,
            max_replicas=5,
            target_cpu_percentage=70
        )
        
        # Test scaling up beyond max replicas
        autoscaler.current_replicas['test:test-app'] = 5  # Already at max
        autoscaler.update_pod_metrics(
            namespace='test',
            deployment='test-app',
            cpu_percentage=90,  # Very high usage
            memory_percentage=80,
            current_replicas=5
        )
        
        scaling_decision = await autoscaler.evaluate_scaling('test', 'test-app')
        
        # Should not scale beyond max replicas
        assert scaling_decision['target_replicas'] <= 5
        
        # Test scaling down below min replicas
        autoscaler.current_replicas['test:test-app'] = 2  # At min
        autoscaler.update_pod_metrics(
            namespace='test',
            deployment='test-app',
            cpu_percentage=10,  # Very low usage
            memory_percentage=20,
            current_replicas=2
        )
        
        scaling_decision = await autoscaler.evaluate_scaling('test', 'test-app')
        
        # Should not scale below min replicas
        assert scaling_decision['target_replicas'] >= 2
    
    async def test_scaling_stability_and_thrashing_prevention(self, autoscaler):
        """Test that scaling prevents thrashing and maintains stability."""
        
        # Configure HPA
        autoscaler.create_hpa(
            name='stability-hpa',
            namespace='production',
            target_deployment='stable-app',
            min_replicas=2,
            max_replicas=8,
            target_cpu_percentage=70
        )
        
        # Simulate multiple scaling scenarios
        scaling_scenarios = [
            (75, 'should_scale_up'),      # Slightly above target
            (72, 'should_not_scale'),     # Within tolerance
            (68, 'should_not_scale'),     # Within tolerance
            (85, 'should_scale_up'),      # Well above target
            (45, 'should_scale_down'),    # Well below target
            (65, 'should_not_scale')      # Within tolerance
        ]
        
        current_replicas = 3
        stability_violations = 0
        
        for cpu_usage, expected_action in scaling_scenarios:
            autoscaler.update_pod_metrics(
                namespace='production',
                deployment='stable-app',
                cpu_percentage=cpu_usage,
                memory_percentage=50,
                current_replicas=current_replicas
            )
            
            scaling_decision = await autoscaler.evaluate_scaling('production', 'stable-app')
            
            # Check for unnecessary scaling (thrashing)
            if expected_action == 'should_not_scale' and scaling_decision['scaling_action'] != 'none':
                stability_violations += 1
            
            if scaling_decision['scaling_action'] != 'none':
                await autoscaler.execute_scaling(scaling_decision)
                current_replicas = scaling_decision['target_replicas']
        
        # Should have minimal stability violations
        assert stability_violations <= 1, f"Too many stability violations: {stability_violations}"
        
        # Check scaling history for patterns
        scaling_history = autoscaler.get_scaling_history('production', 'stable-app')
        
        # Should not have rapid back-and-forth scaling
        if len(scaling_history) >= 2:
            recent_actions = [event['action'] for event in scaling_history[:2]]
            assert not (recent_actions[0] == 'scale_up' and recent_actions[1] == 'scale_down'), \
                "Detected scaling thrashing"


@pytest.mark.production
@pytest.mark.containers
class TestMultiRegionDeployment:
    """Test multi-region deployment scenarios."""
    
    @pytest.fixture
    def multi_region_manager(self):
        """Mock multi-region deployment manager."""
        
        class MockMultiRegionManager:
            def __init__(self):
                self.regions = {
                    'us-east-1': {'status': 'active', 'latency_ms': 50, 'capacity': 100},
                    'us-west-2': {'status': 'active', 'latency_ms': 80, 'capacity': 80},
                    'eu-west-1': {'status': 'active', 'latency_ms': 120, 'capacity': 60},
                    'ap-southeast-1': {'status': 'standby', 'latency_ms': 200, 'capacity': 40}
                }
                self.deployments = {}
                self.traffic_routing = {}
                self.failover_events = []
            
            async def deploy_to_region(self, region: str, app_version: str, 
                                     replicas: int = 3) -> Dict[str, Any]:
                """Deploy application to a specific region."""
                
                if region not in self.regions:
                    raise ValueError(f"Unknown region: {region}")
                
                # Simulate deployment
                deployment = {
                    'region': region,
                    'app_version': app_version,
                    'replicas': replicas,
                    'status': 'deploying',
                    'deployment_start': datetime.utcnow(),
                    'health_check_url': f'https://{region}.lightrag.example.com/health'
                }
                
                # Simulate deployment time
                await asyncio.sleep(0.1)
                
                deployment['status'] = 'healthy'
                deployment['deployment_end'] = datetime.utcnow()
                deployment['deployment_duration'] = (
                    deployment['deployment_end'] - deployment['deployment_start']
                ).total_seconds()
                
                self.deployments[region] = deployment
                return deployment
            
            def configure_traffic_routing(self, routing_config: Dict[str, int]):
                """Configure traffic routing percentages across regions."""
                
                total_percentage = sum(routing_config.values())
                if total_percentage != 100:
                    raise ValueError(f"Traffic percentages must sum to 100, got {total_percentage}")
                
                self.traffic_routing = routing_config.copy()
            
            async def check_region_health(self, region: str) -> Dict[str, Any]:
                """Check health of a specific region."""
                
                if region not in self.regions:
                    return {'status': 'unknown', 'reason': 'region_not_found'}
                
                region_info = self.regions[region]
                deployment = self.deployments.get(region)
                
                health_status = {
                    'region': region,
                    'region_status': region_info['status'],
                    'deployment_status': deployment['status'] if deployment else 'not_deployed',
                    'latency_ms': region_info['latency_ms'],
                    'capacity_percentage': region_info['capacity'],
                    'timestamp': datetime.utcnow()
                }
                
                # Determine overall health
                if (region_info['status'] == 'active' and 
                    deployment and deployment['status'] == 'healthy'):
                    health_status['overall_health'] = 'healthy'
                elif region_info['status'] == 'standby':
                    health_status['overall_health'] = 'standby'
                else:
                    health_status['overall_health'] = 'unhealthy'
                
                return health_status
            
            async def trigger_regional_failover(self, failed_region: str, 
                                              target_region: str = None) -> Dict[str, Any]:
                """Trigger failover from failed region to target region."""
                
                failover_event = {
                    'event_id': f"failover_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    'failed_region': failed_region,
                    'target_region': target_region,
                    'start_time': datetime.utcnow(),
                    'status': 'in_progress'
                }
                
                # Determine target region if not specified
                if not target_region:
                    available_regions = [
                        region for region, info in self.regions.items()
                        if region != failed_region and info['status'] == 'active'
                    ]
                    if available_regions:
                        # Choose region with lowest latency
                        target_region = min(available_regions, 
                                          key=lambda r: self.regions[r]['latency_ms'])
                        failover_event['target_region'] = target_region
                    else:
                        failover_event['status'] = 'failed'
                        failover_event['reason'] = 'no_available_regions'
                        return failover_event
                
                # Simulate failover process
                failover_steps = [
                    'detect_failure',
                    'validate_target_region',
                    'redirect_traffic',
                    'scale_up_target_region',
                    'verify_service_health'
                ]
                
                for step in failover_steps:
                    await asyncio.sleep(0.02)  # Simulate step processing
                
                # Update traffic routing
                if failed_region in self.traffic_routing:
                    failed_traffic = self.traffic_routing[failed_region]
                    del self.traffic_routing[failed_region]
                    
                    if target_region in self.traffic_routing:
                        self.traffic_routing[target_region] += failed_traffic
                    else:
                        self.traffic_routing[target_region] = failed_traffic
                
                # Mark failed region as unhealthy
                self.regions[failed_region]['status'] = 'failed'
                
                failover_event['end_time'] = datetime.utcnow()
                failover_event['duration'] = (
                    failover_event['end_time'] - failover_event['start_time']
                ).total_seconds()
                failover_event['status'] = 'completed'
                
                self.failover_events.append(failover_event)
                return failover_event
            
            def get_global_deployment_status(self) -> Dict[str, Any]:
                """Get global deployment status across all regions."""
                
                total_regions = len(self.regions)
                healthy_regions = sum(1 for info in self.regions.values() if info['status'] == 'active')
                deployed_regions = len(self.deployments)
                
                avg_latency = sum(info['latency_ms'] for info in self.regions.values()) / total_regions
                total_capacity = sum(info['capacity'] for info in self.regions.values())
                
                return {
                    'total_regions': total_regions,
                    'healthy_regions': healthy_regions,
                    'deployed_regions': deployed_regions,
                    'availability_percentage': (healthy_regions / total_regions) * 100,
                    'average_latency_ms': avg_latency,
                    'total_capacity': total_capacity,
                    'traffic_routing': self.traffic_routing.copy(),
                    'last_updated': datetime.utcnow()
                }
        
        return MockMultiRegionManager()
    
    async def test_multi_region_deployment(self, multi_region_manager):
        """Test deployment across multiple regions."""
        
        # Deploy to primary regions
        primary_regions = ['us-east-1', 'us-west-2', 'eu-west-1']
        app_version = 'v1.2.3'
        
        deployment_results = []
        for region in primary_regions:
            result = await multi_region_manager.deploy_to_region(
                region=region,
                app_version=app_version,
                replicas=3
            )
            deployment_results.append(result)
        
        # Validate deployments
        for result in deployment_results:
            assert result['status'] == 'healthy'
            assert result['app_version'] == app_version
            assert result['replicas'] == 3
            assert result['deployment_duration'] > 0
        
        # Check global deployment status
        global_status = multi_region_manager.get_global_deployment_status()
        
        assert global_status['deployed_regions'] == 3
        assert global_status['healthy_regions'] >= 3
        assert global_status['availability_percentage'] >= 75  # At least 3/4 regions healthy
    
    async def test_traffic_routing_configuration(self, multi_region_manager):
        """Test traffic routing across regions."""
        
        # Deploy to regions first
        regions = ['us-east-1', 'us-west-2', 'eu-west-1']
        for region in regions:
            await multi_region_manager.deploy_to_region(region, 'v1.0.0')
        
        # Configure traffic routing
        traffic_config = {
            'us-east-1': 50,  # 50% traffic
            'us-west-2': 30,  # 30% traffic
            'eu-west-1': 20   # 20% traffic
        }
        
        multi_region_manager.configure_traffic_routing(traffic_config)
        
        # Validate traffic routing
        global_status = multi_region_manager.get_global_deployment_status()
        assert global_status['traffic_routing'] == traffic_config
        
        # Test invalid traffic configuration
        with pytest.raises(ValueError):
            multi_region_manager.configure_traffic_routing({
                'us-east-1': 60,
                'us-west-2': 50  # Total = 110% (invalid)
            })
    
    async def test_regional_failover(self, multi_region_manager):
        """Test regional failover scenarios."""
        
        # Deploy to multiple regions
        regions = ['us-east-1', 'us-west-2', 'eu-west-1']
        for region in regions:
            await multi_region_manager.deploy_to_region(region, 'v1.0.0')
        
        # Configure initial traffic routing
        multi_region_manager.configure_traffic_routing({
            'us-east-1': 60,
            'us-west-2': 25,
            'eu-west-1': 15
        })
        
        # Trigger failover from primary region
        failover_result = await multi_region_manager.trigger_regional_failover(
            failed_region='us-east-1',
            target_region='us-west-2'
        )
        
        # Validate failover execution
        assert failover_result['status'] == 'completed'
        assert failover_result['failed_region'] == 'us-east-1'
        assert failover_result['target_region'] == 'us-west-2'
        assert failover_result['duration'] > 0
        
        # Check traffic has been redirected
        global_status = multi_region_manager.get_global_deployment_status()
        routing = global_status['traffic_routing']
        
        assert 'us-east-1' not in routing  # Failed region removed
        assert routing['us-west-2'] == 85  # Original 25% + failed region's 60%
        assert routing['eu-west-1'] == 15  # Unchanged
        
        # Validate region status
        assert multi_region_manager.regions['us-east-1']['status'] == 'failed'
    
    async def test_automatic_regional_failover(self, multi_region_manager):
        """Test automatic failover without specified target region."""
        
        # Deploy to multiple regions
        regions = ['us-east-1', 'us-west-2', 'eu-west-1']
        for region in regions:
            await multi_region_manager.deploy_to_region(region, 'v1.0.0')
        
        # Configure traffic routing
        multi_region_manager.configure_traffic_routing({
            'us-east-1': 50,
            'us-west-2': 30,
            'eu-west-1': 20
        })
        
        # Trigger automatic failover (no target specified)
        failover_result = await multi_region_manager.trigger_regional_failover(
            failed_region='eu-west-1'
        )
        
        # Should automatically choose best target region
        assert failover_result['status'] == 'completed'
        assert failover_result['target_region'] in ['us-east-1', 'us-west-2']
        
        # Should choose region with lowest latency (us-east-1 has 50ms)
        assert failover_result['target_region'] == 'us-east-1'
    
    async def test_multi_region_health_monitoring(self, multi_region_manager):
        """Test health monitoring across multiple regions."""
        
        # Deploy to all regions
        all_regions = list(multi_region_manager.regions.keys())
        for region in all_regions:
            await multi_region_manager.deploy_to_region(region, 'v1.0.0')
        
        # Check health of each region
        health_results = []
        for region in all_regions:
            health = await multi_region_manager.check_region_health(region)
            health_results.append(health)
        
        # Validate health checks
        for health in health_results:
            assert 'region' in health
            assert 'overall_health' in health
            assert 'latency_ms' in health
            assert 'capacity_percentage' in health
            
            # Active regions should be healthy after deployment
            if multi_region_manager.regions[health['region']]['status'] == 'active':
                assert health['overall_health'] == 'healthy'
            elif multi_region_manager.regions[health['region']]['status'] == 'standby':
                assert health['overall_health'] == 'standby'
        
        # Global health should be good
        global_status = multi_region_manager.get_global_deployment_status()
        assert global_status['availability_percentage'] >= 75
        assert global_status['average_latency_ms'] < 200