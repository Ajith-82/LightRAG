"""
Production Monitoring and Observability Tests

Tests for metrics collection validation, log aggregation tests,
distributed tracing, alert configuration tests, and dashboard validation.
"""

import asyncio
import json
import logging
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest


@dataclass
class MetricPoint:
    """Represents a single metric data point."""
    timestamp: datetime
    value: float
    labels: Dict[str, str]
    metric_name: str


@dataclass
class LogEntry:
    """Represents a log entry."""
    timestamp: datetime
    level: str
    message: str
    service: str
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    labels: Optional[Dict[str, str]] = None


@dataclass
class Alert:
    """Represents an alert configuration."""
    name: str
    condition: str
    threshold: float
    duration: int
    severity: str
    enabled: bool


@pytest.mark.production
@pytest.mark.monitoring
class TestMetricsCollection:
    """Test metrics collection and validation."""
    
    @pytest.fixture
    def metrics_collector(self):
        """Mock metrics collector."""
        
        class MockMetricsCollector:
            def __init__(self):
                self.metrics = defaultdict(list)
                self.collection_interval = 15  # seconds
                self.is_collecting = False
            
            def record_metric(self, name: str, value: float, labels: Dict[str, str] = None):
                """Record a metric data point."""
                metric_point = MetricPoint(
                    timestamp=datetime.utcnow(),
                    value=value,
                    labels=labels or {},
                    metric_name=name
                )
                self.metrics[name].append(metric_point)
            
            def get_metrics(self, name: str, start_time: datetime = None, end_time: datetime = None) -> List[MetricPoint]:
                """Get metrics within time range."""
                metrics = self.metrics.get(name, [])
                
                if start_time or end_time:
                    filtered_metrics = []
                    for metric in metrics:
                        if start_time and metric.timestamp < start_time:
                            continue
                        if end_time and metric.timestamp > end_time:
                            continue
                        filtered_metrics.append(metric)
                    return filtered_metrics
                
                return metrics
            
            def start_collection(self):
                """Start metrics collection."""
                self.is_collecting = True
            
            def stop_collection(self):
                """Stop metrics collection."""
                self.is_collecting = False
        
        return MockMetricsCollector()
    
    async def test_application_metrics_collection(self, metrics_collector):
        """Test collection of application-specific metrics."""
        
        # Define expected application metrics
        expected_metrics = [
            'http_requests_total',
            'http_request_duration_seconds',
            'lightrag_queries_total',
            'lightrag_query_duration_seconds',
            'lightrag_documents_processed_total',
            'lightrag_vector_search_duration_seconds',
            'lightrag_graph_traversal_duration_seconds',
            'lightrag_embeddings_generated_total',
            'lightrag_cache_hits_total',
            'lightrag_cache_misses_total'
        ]
        
        # Simulate metric collection
        metrics_collector.start_collection()
        
        # Generate sample metrics
        for metric_name in expected_metrics:
            for i in range(10):
                if 'total' in metric_name:
                    value = i + 1  # Counter metric
                elif 'duration' in metric_name:
                    value = 0.1 + (i * 0.05)  # Duration in seconds
                else:
                    value = float(i)
                
                labels = {
                    'service': 'lightrag',
                    'environment': 'production',
                    'instance': f'instance_{i % 3}'
                }
                
                if 'http' in metric_name:
                    labels['method'] = 'POST'
                    labels['endpoint'] = '/api/query'
                    labels['status_code'] = '200'
                
                metrics_collector.record_metric(metric_name, value, labels)
        
        # Validate metric collection
        for metric_name in expected_metrics:
            collected_metrics = metrics_collector.get_metrics(metric_name)
            assert len(collected_metrics) > 0, f"No metrics collected for {metric_name}"
            
            # Validate metric structure
            for metric in collected_metrics[:3]:  # Check first 3 metrics
                assert isinstance(metric.timestamp, datetime)
                assert isinstance(metric.value, (int, float))
                assert isinstance(metric.labels, dict)
                assert metric.metric_name == metric_name
                
                # Validate required labels
                assert 'service' in metric.labels
                assert 'environment' in metric.labels
                assert 'instance' in metric.labels
        
        metrics_collector.stop_collection()
    
    async def test_system_metrics_collection(self, metrics_collector):
        """Test collection of system-level metrics."""
        
        # System metrics to collect
        system_metrics = [
            'system_cpu_usage_percent',
            'system_memory_usage_bytes',
            'system_memory_usage_percent',
            'system_disk_usage_bytes',
            'system_disk_usage_percent',
            'system_network_bytes_sent',
            'system_network_bytes_received',
            'system_load_average_1m',
            'system_load_average_5m',
            'system_load_average_15m'
        ]
        
        # Mock system metrics collection
        import random
        
        for metric_name in system_metrics:
            for i in range(10):
                if 'percent' in metric_name:
                    value = random.uniform(20, 80)  # Percentage values
                elif 'bytes' in metric_name:
                    value = random.uniform(1000000, 10000000)  # Byte values
                elif 'load_average' in metric_name:
                    value = random.uniform(0.1, 2.0)  # Load average values
                else:
                    value = random.uniform(10, 100)
                
                labels = {
                    'host': f'lightrag-{i % 3}',
                    'environment': 'production'
                }
                
                if 'disk' in metric_name:
                    labels['mount_point'] = '/'
                elif 'network' in metric_name:
                    labels['interface'] = 'eth0'
                
                metrics_collector.record_metric(metric_name, value, labels)
        
        # Validate system metrics
        for metric_name in system_metrics:
            collected_metrics = metrics_collector.get_metrics(metric_name)
            assert len(collected_metrics) > 0, f"No system metrics collected for {metric_name}"
            
            # Validate metric values are within expected ranges
            for metric in collected_metrics:
                if 'percent' in metric_name:
                    assert 0 <= metric.value <= 100, f"Invalid percentage value: {metric.value}"
                elif 'bytes' in metric_name:
                    assert metric.value >= 0, f"Invalid byte value: {metric.value}"
                elif 'load_average' in metric_name:
                    assert metric.value >= 0, f"Invalid load average: {metric.value}"
    
    async def test_custom_business_metrics(self, metrics_collector):
        """Test collection of custom business metrics."""
        
        # Custom business metrics
        business_metrics = [
            'lightrag_active_users_total',
            'lightrag_knowledge_base_size_bytes',
            'lightrag_entities_total',
            'lightrag_relationships_total',
            'lightrag_documents_total',
            'lightrag_query_accuracy_score',
            'lightrag_user_satisfaction_score',
            'lightrag_api_key_usage_total'
        ]
        
        # Generate business metrics
        for metric_name in business_metrics:
            for i in range(10):
                if 'total' in metric_name:
                    value = 1000 + (i * 100)  # Growing totals
                elif 'score' in metric_name:
                    value = 0.7 + (i * 0.03)  # Score between 0.7-1.0
                elif 'size_bytes' in metric_name:
                    value = 1000000 + (i * 100000)  # Growing size
                else:
                    value = float(i * 10)
                
                labels = {
                    'service': 'lightrag',
                    'tenant': f'tenant_{i % 5}',
                    'environment': 'production'
                }
                
                metrics_collector.record_metric(metric_name, value, labels)
        
        # Validate business metrics
        for metric_name in business_metrics:
            collected_metrics = metrics_collector.get_metrics(metric_name)
            assert len(collected_metrics) > 0, f"No business metrics collected for {metric_name}"
            
            # Validate business logic
            for metric in collected_metrics:
                if 'score' in metric_name:
                    assert 0 <= metric.value <= 1, f"Invalid score value: {metric.value}"
                elif 'total' in metric_name:
                    assert metric.value >= 0, f"Invalid total value: {metric.value}"
                
                # Validate business labels
                assert 'tenant' in metric.labels, "Missing tenant label in business metric"
    
    async def test_metrics_aggregation(self, metrics_collector):
        """Test metrics aggregation and calculation."""
        
        # Generate sample metrics for aggregation
        metric_name = 'http_request_duration_seconds'
        durations = [0.1, 0.2, 0.15, 0.3, 0.25, 0.18, 0.22, 0.35, 0.12, 0.28]
        
        for i, duration in enumerate(durations):
            labels = {
                'endpoint': '/api/query',
                'method': 'POST',
                'status_code': '200'
            }
            metrics_collector.record_metric(metric_name, duration, labels)
        
        # Get collected metrics
        collected_metrics = metrics_collector.get_metrics(metric_name)
        
        # Calculate aggregations
        values = [m.value for m in collected_metrics]
        
        # Aggregation calculations
        avg_duration = sum(values) / len(values)
        min_duration = min(values)
        max_duration = max(values)
        
        # Percentile calculations (simplified)
        sorted_values = sorted(values)
        p50_index = int(len(sorted_values) * 0.5)
        p95_index = int(len(sorted_values) * 0.95)
        p99_index = int(len(sorted_values) * 0.99)
        
        p50_duration = sorted_values[p50_index]
        p95_duration = sorted_values[min(p95_index, len(sorted_values) - 1)]
        p99_duration = sorted_values[min(p99_index, len(sorted_values) - 1)]
        
        # Validate aggregations
        assert 0 < avg_duration < 1, f"Invalid average duration: {avg_duration}"
        assert min_duration <= avg_duration <= max_duration
        assert p50_duration <= p95_duration <= p99_duration
        
        # Performance thresholds
        assert avg_duration < 0.5, f"Average request duration too high: {avg_duration}"
        assert p95_duration < 1.0, f"P95 request duration too high: {p95_duration}"
        assert p99_duration < 2.0, f"P99 request duration too high: {p99_duration}"


@pytest.mark.production
@pytest.mark.monitoring
class TestLogAggregation:
    """Test log aggregation and analysis."""
    
    @pytest.fixture
    def log_aggregator(self):
        """Mock log aggregator."""
        
        class MockLogAggregator:
            def __init__(self):
                self.logs = []
                self.indexes = defaultdict(list)
            
            def ingest_log(self, log_entry: LogEntry):
                """Ingest a log entry."""
                self.logs.append(log_entry)
                
                # Create indexes for faster searching
                self.indexes['service'].append((len(self.logs) - 1, log_entry.service))
                self.indexes['level'].append((len(self.logs) - 1, log_entry.level))
                
                if log_entry.trace_id:
                    self.indexes['trace_id'].append((len(self.logs) - 1, log_entry.trace_id))
            
            def search_logs(self, 
                          service: str = None, 
                          level: str = None, 
                          trace_id: str = None,
                          start_time: datetime = None,
                          end_time: datetime = None,
                          limit: int = 100) -> List[LogEntry]:
                """Search logs with filters."""
                
                filtered_logs = []
                
                for log_entry in self.logs:
                    # Apply filters
                    if service and log_entry.service != service:
                        continue
                    if level and log_entry.level != level:
                        continue
                    if trace_id and log_entry.trace_id != trace_id:
                        continue
                    if start_time and log_entry.timestamp < start_time:
                        continue
                    if end_time and log_entry.timestamp > end_time:
                        continue
                    
                    filtered_logs.append(log_entry)
                    
                    if len(filtered_logs) >= limit:
                        break
                
                return filtered_logs
            
            def get_log_stats(self) -> Dict[str, Any]:
                """Get log statistics."""
                stats = {
                    'total_logs': len(self.logs),
                    'levels': defaultdict(int),
                    'services': defaultdict(int),
                    'errors_last_hour': 0,
                    'warnings_last_hour': 0
                }
                
                now = datetime.utcnow()
                one_hour_ago = now - timedelta(hours=1)
                
                for log_entry in self.logs:
                    stats['levels'][log_entry.level] += 1
                    stats['services'][log_entry.service] += 1
                    
                    if log_entry.timestamp >= one_hour_ago:
                        if log_entry.level == 'ERROR':
                            stats['errors_last_hour'] += 1
                        elif log_entry.level == 'WARNING':
                            stats['warnings_last_hour'] += 1
                
                return stats
        
        return MockLogAggregator()
    
    async def test_structured_logging_ingestion(self, log_aggregator):
        """Test ingestion of structured logs."""
        
        # Generate structured log entries
        log_entries = [
            LogEntry(
                timestamp=datetime.utcnow(),
                level='INFO',
                message='Query processed successfully',
                service='lightrag-api',
                trace_id='trace_123',
                span_id='span_456',
                labels={'endpoint': '/api/query', 'user_id': 'user_789'}
            ),
            LogEntry(
                timestamp=datetime.utcnow(),
                level='ERROR',
                message='Database connection failed',
                service='lightrag-api',
                trace_id='trace_124',
                span_id='span_457',
                labels={'database': 'postgres', 'retry_count': '3'}
            ),
            LogEntry(
                timestamp=datetime.utcnow(),
                level='WARNING',
                message='High memory usage detected',
                service='lightrag-worker',
                trace_id='trace_125',
                span_id='span_458',
                labels={'memory_usage_percent': '85', 'threshold': '80'}
            ),
            LogEntry(
                timestamp=datetime.utcnow(),
                level='DEBUG',
                message='Vector similarity search completed',
                service='lightrag-worker',
                trace_id='trace_123',
                span_id='span_459',
                labels={'search_duration_ms': '150', 'results_count': '10'}
            )
        ]
        
        # Ingest log entries
        for log_entry in log_entries:
            log_aggregator.ingest_log(log_entry)
        
        # Validate log ingestion
        all_logs = log_aggregator.search_logs()
        assert len(all_logs) == len(log_entries)
        
        # Test log searching by service
        api_logs = log_aggregator.search_logs(service='lightrag-api')
        assert len(api_logs) == 2
        assert all(log.service == 'lightrag-api' for log in api_logs)
        
        # Test log searching by level
        error_logs = log_aggregator.search_logs(level='ERROR')
        assert len(error_logs) == 1
        assert error_logs[0].message == 'Database connection failed'
        
        # Test log searching by trace_id
        trace_logs = log_aggregator.search_logs(trace_id='trace_123')
        assert len(trace_logs) == 2
        assert all(log.trace_id == 'trace_123' for log in trace_logs)
    
    async def test_log_pattern_detection(self, log_aggregator):
        """Test detection of log patterns and anomalies."""
        
        # Generate logs with patterns
        error_patterns = [
            'Database connection timeout',
            'Out of memory error',
            'Authentication failed',
            'Rate limit exceeded',
            'Service unavailable'
        ]
        
        # Generate normal and error logs
        normal_logs_count = 100
        error_logs_count = 15
        
        # Generate normal logs
        for i in range(normal_logs_count):
            log_entry = LogEntry(
                timestamp=datetime.utcnow() - timedelta(minutes=i),
                level='INFO',
                message=f'Processing request {i}',
                service='lightrag-api',
                labels={'request_id': f'req_{i}'}
            )
            log_aggregator.ingest_log(log_entry)
        
        # Generate error logs with patterns
        for i in range(error_logs_count):
            error_message = error_patterns[i % len(error_patterns)]
            log_entry = LogEntry(
                timestamp=datetime.utcnow() - timedelta(minutes=i * 2),
                level='ERROR',
                message=error_message,
                service='lightrag-api',
                labels={'error_code': f'E{i:03d}'}
            )
            log_aggregator.ingest_log(log_entry)
        
        # Analyze log patterns
        stats = log_aggregator.get_log_stats()
        
        # Validate log statistics
        assert stats['total_logs'] == normal_logs_count + error_logs_count
        assert stats['levels']['INFO'] == normal_logs_count
        assert stats['levels']['ERROR'] == error_logs_count
        
        # Test error rate calculation
        error_rate = stats['levels']['ERROR'] / stats['total_logs']
        assert error_rate < 0.2, f"Error rate too high: {error_rate:.2%}"
        
        # Pattern detection for common errors
        all_logs = log_aggregator.search_logs()
        error_messages = [log.message for log in all_logs if log.level == 'ERROR']
        
        # Count error patterns
        pattern_counts = defaultdict(int)
        for message in error_messages:
            for pattern in error_patterns:
                if pattern in message:
                    pattern_counts[pattern] += 1
        
        # Validate pattern detection
        assert len(pattern_counts) > 0, "No error patterns detected"
        for pattern, count in pattern_counts.items():
            assert count > 0, f"Pattern '{pattern}' not found in logs"
    
    async def test_log_correlation_and_tracing(self, log_aggregator):
        """Test log correlation using trace IDs."""
        
        # Create correlated log entries for a request trace
        trace_id = 'trace_request_flow_001'
        
        # Request flow: API -> Worker -> Database -> Worker -> API
        trace_logs = [
            LogEntry(
                timestamp=datetime.utcnow(),
                level='INFO',
                message='Received query request',
                service='lightrag-api',
                trace_id=trace_id,
                span_id='span_001',
                labels={'endpoint': '/api/query', 'method': 'POST'}
            ),
            LogEntry(
                timestamp=datetime.utcnow() + timedelta(milliseconds=10),
                level='DEBUG',
                message='Forwarding to worker',
                service='lightrag-api',
                trace_id=trace_id,
                span_id='span_002',
                labels={'worker_id': 'worker_1'}
            ),
            LogEntry(
                timestamp=datetime.utcnow() + timedelta(milliseconds=50),
                level='INFO',
                message='Processing query',
                service='lightrag-worker',
                trace_id=trace_id,
                span_id='span_003',
                labels={'query_type': 'hybrid', 'worker_id': 'worker_1'}
            ),
            LogEntry(
                timestamp=datetime.utcnow() + timedelta(milliseconds=100),
                level='DEBUG',
                message='Executing vector search',
                service='lightrag-worker',
                trace_id=trace_id,
                span_id='span_004',
                labels={'search_type': 'similarity', 'vector_dim': '768'}
            ),
            LogEntry(
                timestamp=datetime.utcnow() + timedelta(milliseconds=150),
                level='DEBUG',
                message='Database query executed',
                service='lightrag-database',
                trace_id=trace_id,
                span_id='span_005',
                labels={'query_duration_ms': '45', 'result_count': '25'}
            ),
            LogEntry(
                timestamp=datetime.utcnow() + timedelta(milliseconds=200),
                level='INFO',
                message='Query completed',
                service='lightrag-worker',
                trace_id=trace_id,
                span_id='span_006',
                labels={'response_size_kb': '12', 'cache_hit': 'false'}
            ),
            LogEntry(
                timestamp=datetime.utcnow() + timedelta(milliseconds=220),
                level='INFO',
                message='Response sent to client',
                service='lightrag-api',
                trace_id=trace_id,
                span_id='span_007',
                labels={'status_code': '200', 'response_time_ms': '220'}
            )
        ]
        
        # Ingest trace logs
        for log_entry in trace_logs:
            log_aggregator.ingest_log(log_entry)
        
        # Test trace correlation
        correlated_logs = log_aggregator.search_logs(trace_id=trace_id)
        assert len(correlated_logs) == len(trace_logs)
        
        # Validate trace flow
        sorted_logs = sorted(correlated_logs, key=lambda x: x.timestamp)
        
        # Check service flow
        expected_service_flow = [
            'lightrag-api', 'lightrag-api', 'lightrag-worker', 
            'lightrag-worker', 'lightrag-database', 'lightrag-worker', 'lightrag-api'
        ]
        
        actual_service_flow = [log.service for log in sorted_logs]
        assert actual_service_flow == expected_service_flow
        
        # Validate span progression
        span_ids = [log.span_id for log in sorted_logs]
        assert len(set(span_ids)) == len(span_ids), "Duplicate span IDs found"
        assert all(span_id.startswith('span_') for span_id in span_ids)
        
        # Calculate request duration
        start_time = sorted_logs[0].timestamp
        end_time = sorted_logs[-1].timestamp
        request_duration = (end_time - start_time).total_seconds() * 1000  # milliseconds
        
        assert request_duration < 1000, f"Request duration too high: {request_duration}ms"
    
    async def test_log_retention_and_archival(self, log_aggregator):
        """Test log retention and archival policies."""
        
        # Mock retention policy
        retention_policy = {
            'hot_storage_days': 7,
            'warm_storage_days': 30,
            'cold_storage_days': 90,
            'delete_after_days': 365
        }
        
        # Generate logs across different time periods
        now = datetime.utcnow()
        time_periods = [
            ('hot', now - timedelta(days=3)),      # Hot storage
            ('warm', now - timedelta(days=15)),     # Warm storage  
            ('cold', now - timedelta(days=60)),     # Cold storage
            ('archive', now - timedelta(days=200)),  # Archive
            ('delete', now - timedelta(days=400))    # Should be deleted
        ]
        
        logs_by_period = {}
        
        for period_name, period_time in time_periods:
            period_logs = []
            for i in range(10):
                log_entry = LogEntry(
                    timestamp=period_time + timedelta(hours=i),
                    level='INFO',
                    message=f'Log from {period_name} period',
                    service='lightrag-api',
                    labels={'period': period_name, 'log_index': str(i)}
                )
                log_aggregator.ingest_log(log_entry)
                period_logs.append(log_entry)
            
            logs_by_period[period_name] = period_logs
        
        # Test retention policy application
        def apply_retention_policy(logs: List[LogEntry], policy: Dict[str, int]) -> Dict[str, List[LogEntry]]:
            """Apply retention policy to logs."""
            now = datetime.utcnow()
            categorized_logs = {
                'hot': [],
                'warm': [],
                'cold': [],
                'archive': [],
                'delete': []
            }
            
            for log in logs:
                age_days = (now - log.timestamp).days
                
                if age_days <= policy['hot_storage_days']:
                    categorized_logs['hot'].append(log)
                elif age_days <= policy['warm_storage_days']:
                    categorized_logs['warm'].append(log)
                elif age_days <= policy['cold_storage_days']:
                    categorized_logs['cold'].append(log)
                elif age_days <= policy['delete_after_days']:
                    categorized_logs['archive'].append(log)
                else:
                    categorized_logs['delete'].append(log)
            
            return categorized_logs
        
        all_logs = log_aggregator.search_logs(limit=1000)
        categorized = apply_retention_policy(all_logs, retention_policy)
        
        # Validate retention policy
        assert len(categorized['hot']) > 0, "No logs in hot storage"
        assert len(categorized['warm']) > 0, "No logs in warm storage"
        assert len(categorized['cold']) > 0, "No logs in cold storage"
        assert len(categorized['archive']) > 0, "No logs in archive"
        assert len(categorized['delete']) > 0, "No logs marked for deletion"
        
        # Validate storage tiering
        total_logs = sum(len(logs) for logs in categorized.values())
        assert total_logs == len(all_logs), "Log count mismatch in retention policy"


@pytest.mark.production
@pytest.mark.monitoring
class TestDistributedTracing:
    """Test distributed tracing implementation."""
    
    @pytest.fixture
    def trace_collector(self):
        """Mock distributed trace collector."""
        
        class MockTraceCollector:
            def __init__(self):
                self.traces = {}
                self.spans = {}
            
            def create_trace(self, trace_id: str, operation_name: str) -> Dict[str, Any]:
                """Create a new trace."""
                trace = {
                    'trace_id': trace_id,
                    'operation_name': operation_name,
                    'start_time': datetime.utcnow(),
                    'spans': [],
                    'status': 'active'
                }
                self.traces[trace_id] = trace
                return trace
            
            def create_span(self, trace_id: str, span_id: str, parent_span_id: str = None, 
                          operation_name: str = '', service_name: str = '') -> Dict[str, Any]:
                """Create a new span."""
                span = {
                    'span_id': span_id,
                    'trace_id': trace_id,
                    'parent_span_id': parent_span_id,
                    'operation_name': operation_name,
                    'service_name': service_name,
                    'start_time': datetime.utcnow(),
                    'end_time': None,
                    'duration_ms': None,
                    'tags': {},
                    'logs': [],
                    'status': 'active'
                }
                
                self.spans[span_id] = span
                
                if trace_id in self.traces:
                    self.traces[trace_id]['spans'].append(span_id)
                
                return span
            
            def finish_span(self, span_id: str, status: str = 'ok'):
                """Finish a span."""
                if span_id in self.spans:
                    span = self.spans[span_id]
                    span['end_time'] = datetime.utcnow()
                    span['duration_ms'] = (span['end_time'] - span['start_time']).total_seconds() * 1000
                    span['status'] = status
            
            def add_span_tag(self, span_id: str, key: str, value: str):
                """Add tag to span."""
                if span_id in self.spans:
                    self.spans[span_id]['tags'][key] = value
            
            def add_span_log(self, span_id: str, message: str, level: str = 'info'):
                """Add log to span."""
                if span_id in self.spans:
                    self.spans[span_id]['logs'].append({
                        'timestamp': datetime.utcnow(),
                        'message': message,
                        'level': level
                    })
            
            def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
                """Get complete trace with all spans."""
                if trace_id not in self.traces:
                    return None
                
                trace = self.traces[trace_id].copy()
                trace['spans'] = [self.spans[span_id] for span_id in trace['spans']]
                return trace
        
        return MockTraceCollector()
    
    async def test_trace_creation_and_propagation(self, trace_collector):
        """Test trace creation and context propagation."""
        
        # Create a distributed trace for a query request
        trace_id = 'trace_distributed_query_001'
        
        # Create root trace
        trace = trace_collector.create_trace(trace_id, 'query_request')
        
        # Create spans for distributed services
        spans = [
            {
                'span_id': 'span_api_gateway',
                'parent_span_id': None,
                'operation_name': 'receive_request',
                'service_name': 'api-gateway'
            },
            {
                'span_id': 'span_auth_service',
                'parent_span_id': 'span_api_gateway',
                'operation_name': 'authenticate_user',
                'service_name': 'auth-service'
            },
            {
                'span_id': 'span_lightrag_api',
                'parent_span_id': 'span_api_gateway',
                'operation_name': 'process_query',
                'service_name': 'lightrag-api'
            },
            {
                'span_id': 'span_vector_search',
                'parent_span_id': 'span_lightrag_api',
                'operation_name': 'vector_similarity_search',
                'service_name': 'lightrag-worker'
            },
            {
                'span_id': 'span_graph_traversal',
                'parent_span_id': 'span_lightrag_api',
                'operation_name': 'knowledge_graph_traversal',
                'service_name': 'lightrag-worker'
            },
            {
                'span_id': 'span_database_query',
                'parent_span_id': 'span_vector_search',
                'operation_name': 'vector_database_query',
                'service_name': 'vector-database'
            }
        ]
        
        # Create and configure spans
        for span_config in spans:
            span = trace_collector.create_span(
                trace_id=trace_id,
                span_id=span_config['span_id'],
                parent_span_id=span_config['parent_span_id'],
                operation_name=span_config['operation_name'],
                service_name=span_config['service_name']
            )
            
            # Add relevant tags
            trace_collector.add_span_tag(span_config['span_id'], 'service.name', span_config['service_name'])
            trace_collector.add_span_tag(span_config['span_id'], 'operation.name', span_config['operation_name'])
            
            # Add operation-specific tags
            if 'auth' in span_config['service_name']:
                trace_collector.add_span_tag(span_config['span_id'], 'user.id', 'user_123')
                trace_collector.add_span_tag(span_config['span_id'], 'auth.method', 'jwt')
            elif 'vector' in span_config['operation_name']:
                trace_collector.add_span_tag(span_config['span_id'], 'vector.dimension', '768')
                trace_collector.add_span_tag(span_config['span_id'], 'search.top_k', '10')
            elif 'graph' in span_config['operation_name']:
                trace_collector.add_span_tag(span_config['span_id'], 'graph.max_depth', '3')
                trace_collector.add_span_tag(span_config['span_id'], 'traversal.algorithm', 'bfs')
            
            # Add logs to spans
            trace_collector.add_span_log(span_config['span_id'], f"Started {span_config['operation_name']}")
            
            # Simulate processing time
            await asyncio.sleep(0.01)
            
            # Finish span
            trace_collector.finish_span(span_config['span_id'], 'ok')
            trace_collector.add_span_log(span_config['span_id'], f"Completed {span_config['operation_name']}")
        
        # Validate trace structure
        complete_trace = trace_collector.get_trace(trace_id)
        assert complete_trace is not None
        assert complete_trace['trace_id'] == trace_id
        assert len(complete_trace['spans']) == len(spans)
        
        # Validate span hierarchy
        span_dict = {span['span_id']: span for span in complete_trace['spans']}
        
        # Check parent-child relationships
        for span in complete_trace['spans']:
            if span['parent_span_id']:
                assert span['parent_span_id'] in span_dict, f"Parent span {span['parent_span_id']} not found"
                parent_span = span_dict[span['parent_span_id']]
                assert parent_span['start_time'] <= span['start_time'], "Child span started before parent"
        
        # Validate span durations
        for span in complete_trace['spans']:
            assert span['duration_ms'] is not None, f"Duration not set for span {span['span_id']}"
            assert span['duration_ms'] >= 0, f"Invalid duration for span {span['span_id']}"
            assert span['status'] == 'ok', f"Span {span['span_id']} not completed successfully"
    
    async def test_trace_error_handling(self, trace_collector):
        """Test error handling and propagation in traces."""
        
        trace_id = 'trace_error_handling_001'
        
        # Create trace with error scenario
        trace = trace_collector.create_trace(trace_id, 'query_with_errors')
        
        # Create spans with simulated errors
        error_scenarios = [
            {
                'span_id': 'span_api_request',
                'operation_name': 'api_request',
                'service_name': 'lightrag-api',
                'status': 'ok'
            },
            {
                'span_id': 'span_auth_check',
                'operation_name': 'authentication',
                'service_name': 'auth-service',
                'status': 'error',
                'error_message': 'Invalid JWT token'
            },
            {
                'span_id': 'span_fallback_auth',
                'operation_name': 'fallback_authentication',
                'service_name': 'auth-service',
                'status': 'ok'
            },
            {
                'span_id': 'span_database_error',
                'operation_name': 'database_query',
                'service_name': 'database',
                'status': 'error',
                'error_message': 'Connection timeout'
            },
            {
                'span_id': 'span_retry_query',
                'operation_name': 'retry_database_query',
                'service_name': 'database',
                'status': 'ok'
            }
        ]
        
        for scenario in error_scenarios:
            span = trace_collector.create_span(
                trace_id=trace_id,
                span_id=scenario['span_id'],
                operation_name=scenario['operation_name'],
                service_name=scenario['service_name']
            )
            
            # Add tags
            trace_collector.add_span_tag(scenario['span_id'], 'service.name', scenario['service_name'])
            
            if scenario['status'] == 'error':
                # Add error information
                trace_collector.add_span_tag(scenario['span_id'], 'error', 'true')
                trace_collector.add_span_tag(scenario['span_id'], 'error.kind', 'service_error')
                trace_collector.add_span_log(
                    scenario['span_id'], 
                    scenario['error_message'], 
                    level='error'
                )
            
            # Finish span with appropriate status
            trace_collector.finish_span(scenario['span_id'], scenario['status'])
        
        # Validate error handling in trace
        complete_trace = trace_collector.get_trace(trace_id)
        error_spans = [span for span in complete_trace['spans'] if span['status'] == 'error']
        success_spans = [span for span in complete_trace['spans'] if span['status'] == 'ok']
        
        assert len(error_spans) == 2, f"Expected 2 error spans, got {len(error_spans)}"
        assert len(success_spans) == 3, f"Expected 3 success spans, got {len(success_spans)}"
        
        # Validate error span details
        for error_span in error_spans:
            assert 'error' in error_span['tags']
            assert error_span['tags']['error'] == 'true'
            assert any(log['level'] == 'error' for log in error_span['logs'])
    
    async def test_trace_performance_analysis(self, trace_collector):
        """Test performance analysis using trace data."""
        
        # Create multiple traces for performance analysis
        trace_ids = [f'trace_perf_analysis_{i:03d}' for i in range(10)]
        
        operation_performance = defaultdict(list)
        service_performance = defaultdict(list)
        
        for trace_id in trace_ids:
            trace = trace_collector.create_trace(trace_id, 'performance_test')
            
            # Standard operation flow
            operations = [
                ('api_request', 'lightrag-api', 50, 150),      # 50-150ms
                ('authentication', 'auth-service', 10, 50),    # 10-50ms
                ('query_processing', 'lightrag-worker', 200, 500),  # 200-500ms
                ('vector_search', 'vector-db', 100, 300),      # 100-300ms
                ('graph_traversal', 'graph-db', 150, 400),     # 150-400ms
                ('response_assembly', 'lightrag-api', 20, 80)  # 20-80ms
            ]
            
            for i, (operation, service, min_ms, max_ms) in enumerate(operations):
                span_id = f'span_{trace_id}_{i}'
                
                span = trace_collector.create_span(
                    trace_id=trace_id,
                    span_id=span_id,
                    operation_name=operation,
                    service_name=service
                )
                
                # Simulate processing time
                import random
                processing_time_ms = random.uniform(min_ms, max_ms)
                await asyncio.sleep(processing_time_ms / 1000)  # Convert to seconds
                
                trace_collector.finish_span(span_id, 'ok')
                
                # Collect performance data
                span_data = trace_collector.spans[span_id]
                operation_performance[operation].append(span_data['duration_ms'])
                service_performance[service].append(span_data['duration_ms'])
        
        # Analyze performance metrics
        performance_analysis = {}
        
        for operation, durations in operation_performance.items():
            performance_analysis[operation] = {
                'avg_duration_ms': sum(durations) / len(durations),
                'min_duration_ms': min(durations),
                'max_duration_ms': max(durations),
                'call_count': len(durations)
            }
        
        # Validate performance expectations
        expected_performance = {
            'api_request': {'max_avg': 150, 'max_p95': 200},
            'authentication': {'max_avg': 50, 'max_p95': 80},
            'query_processing': {'max_avg': 500, 'max_p95': 600},
            'vector_search': {'max_avg': 300, 'max_p95': 400},
            'graph_traversal': {'max_avg': 400, 'max_p95': 500},
            'response_assembly': {'max_avg': 80, 'max_p95': 120}
        }
        
        for operation, metrics in performance_analysis.items():
            if operation in expected_performance:
                expected = expected_performance[operation]
                assert metrics['avg_duration_ms'] <= expected['max_avg'], \
                    f"{operation} average duration too high: {metrics['avg_duration_ms']:.1f}ms"
                assert metrics['call_count'] == len(trace_ids), \
                    f"Unexpected call count for {operation}: {metrics['call_count']}"


@pytest.mark.production
@pytest.mark.monitoring
class TestAlertConfiguration:
    """Test alert configuration and triggering."""
    
    @pytest.fixture
    def alert_manager(self):
        """Mock alert manager."""
        
        class MockAlertManager:
            def __init__(self):
                self.alerts = {}
                self.fired_alerts = []
                self.alert_history = []
            
            def add_alert(self, alert: Alert):
                """Add alert configuration."""
                self.alerts[alert.name] = alert
            
            def evaluate_alerts(self, metrics: Dict[str, float]):
                """Evaluate alerts against current metrics."""
                for alert_name, alert in self.alerts.items():
                    if not alert.enabled:
                        continue
                    
                    if self._evaluate_condition(alert.condition, metrics, alert.threshold):
                        self._fire_alert(alert)
            
            def _evaluate_condition(self, condition: str, metrics: Dict[str, float], threshold: float) -> bool:
                """Evaluate alert condition."""
                if 'greater_than' in condition:
                    metric_name = condition.replace('greater_than:', '')
                    return metrics.get(metric_name, 0) > threshold
                elif 'less_than' in condition:
                    metric_name = condition.replace('less_than:', '')
                    return metrics.get(metric_name, 0) < threshold
                elif 'rate_increase' in condition:
                    # Simplified rate increase detection
                    metric_name = condition.replace('rate_increase:', '')
                    return metrics.get(metric_name, 0) > threshold
                return False
            
            def _fire_alert(self, alert: Alert):
                """Fire an alert."""
                alert_instance = {
                    'alert_name': alert.name,
                    'condition': alert.condition,
                    'threshold': alert.threshold,
                    'severity': alert.severity,
                    'timestamp': datetime.utcnow(),
                    'status': 'firing'
                }
                
                self.fired_alerts.append(alert_instance)
                self.alert_history.append(alert_instance)
            
            def get_active_alerts(self) -> List[Dict[str, Any]]:
                """Get currently active alerts."""
                return [alert for alert in self.fired_alerts if alert['status'] == 'firing']
            
            def resolve_alert(self, alert_name: str):
                """Resolve an alert."""
                for alert in self.fired_alerts:
                    if alert['alert_name'] == alert_name and alert['status'] == 'firing':
                        alert['status'] = 'resolved'
                        alert['resolved_at'] = datetime.utcnow()
        
        return MockAlertManager()
    
    async def test_performance_alerts(self, alert_manager):
        """Test performance-related alerts."""
        
        # Configure performance alerts
        performance_alerts = [
            Alert(
                name='high_response_time',
                condition='greater_than:avg_response_time_ms',
                threshold=1000,  # 1 second
                duration=300,    # 5 minutes
                severity='warning',
                enabled=True
            ),
            Alert(
                name='low_throughput',
                condition='less_than:requests_per_second',
                threshold=10,
                duration=300,
                severity='critical',
                enabled=True
            ),
            Alert(
                name='high_error_rate',
                condition='greater_than:error_rate_percent',
                threshold=5,  # 5%
                duration=60,  # 1 minute
                severity='critical',
                enabled=True
            ),
            Alert(
                name='p95_latency_high',
                condition='greater_than:p95_response_time_ms',
                threshold=2000,  # 2 seconds
                duration=180,     # 3 minutes
                severity='warning',
                enabled=True
            )
        ]
        
        # Add alerts to manager
        for alert in performance_alerts:
            alert_manager.add_alert(alert)
        
        # Test normal conditions (no alerts should fire)
        normal_metrics = {
            'avg_response_time_ms': 200,
            'requests_per_second': 50,
            'error_rate_percent': 1,
            'p95_response_time_ms': 500
        }
        
        alert_manager.evaluate_alerts(normal_metrics)
        active_alerts = alert_manager.get_active_alerts()
        assert len(active_alerts) == 0, f"Unexpected alerts fired: {active_alerts}"
        
        # Test degraded performance (should trigger alerts)
        degraded_metrics = {
            'avg_response_time_ms': 1500,  # High response time
            'requests_per_second': 5,      # Low throughput
            'error_rate_percent': 8,       # High error rate
            'p95_response_time_ms': 2500   # High P95 latency
        }
        
        alert_manager.evaluate_alerts(degraded_metrics)
        active_alerts = alert_manager.get_active_alerts()
        
        # All alerts should fire
        assert len(active_alerts) == 4, f"Expected 4 alerts, got {len(active_alerts)}"
        
        # Validate alert details
        alert_names = [alert['alert_name'] for alert in active_alerts]
        expected_alerts = ['high_response_time', 'low_throughput', 'high_error_rate', 'p95_latency_high']
        
        for expected_alert in expected_alerts:
            assert expected_alert in alert_names, f"Alert {expected_alert} not fired"
        
        # Validate alert severities
        critical_alerts = [alert for alert in active_alerts if alert['severity'] == 'critical']
        warning_alerts = [alert for alert in active_alerts if alert['severity'] == 'warning']
        
        assert len(critical_alerts) == 2, f"Expected 2 critical alerts, got {len(critical_alerts)}"
        assert len(warning_alerts) == 2, f"Expected 2 warning alerts, got {len(warning_alerts)}"
    
    async def test_resource_alerts(self, alert_manager):
        """Test resource utilization alerts."""
        
        # Configure resource alerts
        resource_alerts = [
            Alert(
                name='high_cpu_usage',
                condition='greater_than:cpu_usage_percent',
                threshold=80,
                duration=300,
                severity='warning',
                enabled=True
            ),
            Alert(
                name='high_memory_usage',
                condition='greater_than:memory_usage_percent',
                threshold=85,
                duration=180,
                severity='critical',
                enabled=True
            ),
            Alert(
                name='disk_space_low',
                condition='greater_than:disk_usage_percent',
                threshold=90,
                duration=60,
                severity='critical',
                enabled=True
            ),
            Alert(
                name='connection_pool_exhausted',
                condition='greater_than:db_connections_percent',
                threshold=95,
                duration=30,
                severity='critical',
                enabled=True
            )
        ]
        
        for alert in resource_alerts:
            alert_manager.add_alert(alert)
        
        # Test resource pressure scenarios
        resource_pressure_metrics = {
            'cpu_usage_percent': 85,    # High CPU
            'memory_usage_percent': 90, # High memory
            'disk_usage_percent': 95,   # Low disk space
            'db_connections_percent': 98 # Connection pool exhausted
        }
        
        alert_manager.evaluate_alerts(resource_pressure_metrics)
        active_alerts = alert_manager.get_active_alerts()
        
        assert len(active_alerts) == 4, f"Expected 4 resource alerts, got {len(active_alerts)}"
        
        # All should be high severity due to resource pressure
        critical_alerts = [alert for alert in active_alerts if alert['severity'] == 'critical']
        assert len(critical_alerts) == 3, f"Expected 3 critical resource alerts, got {len(critical_alerts)}"
    
    async def test_business_metric_alerts(self, alert_manager):
        """Test business metric alerts."""
        
        # Configure business metric alerts
        business_alerts = [
            Alert(
                name='user_satisfaction_low',
                condition='less_than:user_satisfaction_score',
                threshold=0.7,  # 70%
                duration=600,   # 10 minutes
                severity='warning',
                enabled=True
            ),
            Alert(
                name='query_accuracy_degraded',
                condition='less_than:query_accuracy_score',
                threshold=0.8,  # 80%
                duration=300,   # 5 minutes
                severity='critical',
                enabled=True
            ),
            Alert(
                name='document_processing_backlog',
                condition='greater_than:pending_documents_count',
                threshold=1000,
                duration=900,   # 15 minutes
                severity='warning',
                enabled=True
            ),
            Alert(
                name='api_key_abuse',
                condition='rate_increase:api_requests_per_key',
                threshold=100,  # 100% increase
                duration=60,    # 1 minute
                severity='critical',
                enabled=True
            )
        ]
        
        for alert in business_alerts:
            alert_manager.add_alert(alert)
        
        # Test business metric degradation
        business_metrics = {
            'user_satisfaction_score': 0.65,    # Low satisfaction
            'query_accuracy_score': 0.75,       # Degraded accuracy
            'pending_documents_count': 1500,     # High backlog
            'api_requests_per_key': 150          # API abuse
        }
        
        alert_manager.evaluate_alerts(business_metrics)
        active_alerts = alert_manager.get_active_alerts()
        
        assert len(active_alerts) == 4, f"Expected 4 business alerts, got {len(active_alerts)}"
        
        # Validate business impact assessment
        business_critical_alerts = [
            alert for alert in active_alerts 
            if alert['alert_name'] in ['query_accuracy_degraded', 'api_key_abuse']
        ]
        assert len(business_critical_alerts) == 2, "Business critical alerts not properly identified"
    
    async def test_alert_escalation_and_resolution(self, alert_manager):
        """Test alert escalation and resolution workflows."""
        
        # Configure escalation alert
        escalation_alert = Alert(
            name='service_unavailable',
            condition='greater_than:error_rate_percent',
            threshold=50,  # 50% error rate
            duration=60,   # 1 minute
            severity='critical',
            enabled=True
        )
        
        alert_manager.add_alert(escalation_alert)
        
        # Trigger alert
        critical_metrics = {'error_rate_percent': 75}
        alert_manager.evaluate_alerts(critical_metrics)
        
        active_alerts = alert_manager.get_active_alerts()
        assert len(active_alerts) == 1
        assert active_alerts[0]['alert_name'] == 'service_unavailable'
        assert active_alerts[0]['severity'] == 'critical'
        
        # Test alert resolution
        alert_manager.resolve_alert('service_unavailable')
        
        # Verify alert is resolved
        active_alerts_after_resolution = alert_manager.get_active_alerts()
        resolved_alerts = [alert for alert in alert_manager.fired_alerts if alert['status'] == 'resolved']
        
        assert len(active_alerts_after_resolution) == 0, "Alert not properly resolved"
        assert len(resolved_alerts) == 1, "Alert resolution not recorded"
        assert 'resolved_at' in resolved_alerts[0], "Resolution timestamp not recorded"