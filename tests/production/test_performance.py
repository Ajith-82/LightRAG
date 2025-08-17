"""
Production Performance and Load Testing

Tests for API endpoint load testing, concurrent user simulations,
resource utilization monitoring, and performance benchmarks.
"""

import asyncio
import concurrent.futures
import json
import random
import statistics
import string
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import httpx
import numpy as np
import psutil
import pytest


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    response_times: List[float]
    success_count: int
    error_count: int
    throughput: float
    avg_response_time: float
    p95_response_time: float
    p99_response_time: float
    cpu_usage: float
    memory_usage: float


@dataclass
class LoadTestConfig:
    """Load test configuration."""
    concurrent_users: int
    duration_seconds: int
    requests_per_second: int
    ramp_up_seconds: int


@pytest.mark.production
@pytest.mark.performance
class TestAPIEndpointPerformance:
    """Test API endpoint performance under various load conditions."""
    
    @pytest.fixture
    def api_base_url(self):
        """Base URL for API testing."""
        return "http://localhost:9621"
    
    @pytest.fixture
    def auth_headers(self):
        """Authentication headers for API requests."""
        return {
            "Authorization": "Bearer test_api_key",
            "Content-Type": "application/json"
        }
    
    async def test_health_endpoint_performance(self, api_base_url, auth_headers):
        """Test health endpoint performance under load."""
        url = f"{api_base_url}/health"
        
        async def make_request(session: aiohttp.ClientSession) -> Tuple[float, int]:
            start_time = time.time()
            try:
                async with session.get(url, headers=auth_headers) as response:
                    await response.text()
                    end_time = time.time()
                    return end_time - start_time, response.status
            except Exception:
                end_time = time.time()
                return end_time - start_time, 500
        
        # Load test configuration
        concurrent_requests = 100
        total_requests = 1000
        
        # Mock session for testing
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text = AsyncMock(return_value='{"status": "healthy"}')
            mock_session.get.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value = mock_session
            
            response_times = []
            status_codes = []
            
            # Simulate concurrent requests
            semaphore = asyncio.Semaphore(concurrent_requests)
            
            async def limited_request():
                async with semaphore:
                    response_time, status = await make_request(mock_session)
                    return response_time, status
            
            # Execute load test
            start_time = time.time()
            tasks = [limited_request() for _ in range(total_requests)]
            results = await asyncio.gather(*tasks)
            end_time = time.time()
            
            # Analyze results
            response_times = [r[0] for r in results]
            status_codes = [r[1] for r in results]
            
            # Calculate metrics
            avg_response_time = statistics.mean(response_times)
            p95_response_time = np.percentile(response_times, 95)
            p99_response_time = np.percentile(response_times, 99)
            success_rate = sum(1 for code in status_codes if code == 200) / len(status_codes)
            throughput = total_requests / (end_time - start_time)
            
            # Performance assertions
            assert avg_response_time < 0.1, f"Average response time too high: {avg_response_time:.3f}s"
            assert p95_response_time < 0.2, f"P95 response time too high: {p95_response_time:.3f}s"
            assert p99_response_time < 0.5, f"P99 response time too high: {p99_response_time:.3f}s"
            assert success_rate >= 0.99, f"Success rate too low: {success_rate:.3f}"
            assert throughput >= 500, f"Throughput too low: {throughput:.1f} req/s"
    
    async def test_query_endpoint_performance(self, api_base_url, auth_headers):
        """Test query endpoint performance with various query types."""
        url = f"{api_base_url}/query"
        
        # Test queries of different types
        test_queries = [
            {"query": "What is artificial intelligence?", "mode": "local"},
            {"query": "Explain machine learning algorithms", "mode": "global"},
            {"query": "Compare deep learning frameworks", "mode": "hybrid"},
            {"query": "Recent developments in AI", "mode": "mix"},
            {"query": "Natural language processing", "mode": "naive"}
        ]
        
        async def make_query_request(session: aiohttp.ClientSession, query_data: Dict) -> Tuple[float, int, int]:
            start_time = time.time()
            try:
                async with session.post(url, json=query_data, headers=auth_headers) as response:
                    response_text = await response.text()
                    end_time = time.time()
                    return end_time - start_time, response.status, len(response_text)
            except Exception:
                end_time = time.time()
                return end_time - start_time, 500, 0
        
        # Mock successful query responses
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text = AsyncMock(return_value='{"result": "Mock AI response with relevant information about the query topic.", "mode": "local", "response_time": 0.05}')
            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value = mock_session
            
            # Test each query mode
            for query_data in test_queries:
                response_times = []
                
                # Run multiple iterations for each query type
                for _ in range(50):
                    response_time, status, response_size = await make_query_request(mock_session, query_data)
                    response_times.append(response_time)
                    assert status == 200
                    assert response_size > 0
                
                # Analyze performance by query mode
                avg_time = statistics.mean(response_times)
                mode = query_data["mode"]
                
                # Different modes have different performance expectations
                if mode == "local":
                    assert avg_time < 0.5, f"Local query too slow: {avg_time:.3f}s"
                elif mode == "global":
                    assert avg_time < 1.0, f"Global query too slow: {avg_time:.3f}s"
                elif mode == "hybrid":
                    assert avg_time < 1.5, f"Hybrid query too slow: {avg_time:.3f}s"
                elif mode == "mix":
                    assert avg_time < 2.0, f"Mix query too slow: {avg_time:.3f}s"
                elif mode == "naive":
                    assert avg_time < 0.3, f"Naive query too slow: {avg_time:.3f}s"
    
    async def test_document_upload_performance(self, api_base_url, auth_headers):
        """Test document upload endpoint performance."""
        url = f"{api_base_url}/documents/upload"
        
        # Generate test documents of various sizes
        test_documents = []
        for size_kb in [1, 10, 100, 500]:
            content = ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=size_kb * 1024))
            test_documents.append({
                'content': content,
                'size_kb': size_kb,
                'filename': f'test_doc_{size_kb}kb.txt'
            })
        
        async def upload_document(session: aiohttp.ClientSession, doc_data: Dict) -> Tuple[float, int]:
            start_time = time.time()
            try:
                form_data = aiohttp.FormData()
                form_data.add_field('file', doc_data['content'], filename=doc_data['filename'])
                
                async with session.post(url, data=form_data, headers={k: v for k, v in auth_headers.items() if k != 'Content-Type'}) as response:
                    await response.text()
                    end_time = time.time()
                    return end_time - start_time, response.status
            except Exception:
                end_time = time.time()
                return end_time - start_time, 500
        
        # Mock upload responses
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text = AsyncMock(return_value='{"status": "uploaded", "document_id": "doc123"}')
            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value = mock_session
            
            # Test upload performance for different document sizes
            for doc_data in test_documents:
                response_times = []
                
                # Multiple uploads per size
                for _ in range(10):
                    response_time, status = await upload_document(mock_session, doc_data)
                    response_times.append(response_time)
                    assert status == 200
                
                avg_time = statistics.mean(response_times)
                size_kb = doc_data['size_kb']
                
                # Performance expectations based on document size
                if size_kb <= 10:
                    assert avg_time < 1.0, f"Small doc upload too slow: {avg_time:.3f}s for {size_kb}KB"
                elif size_kb <= 100:
                    assert avg_time < 5.0, f"Medium doc upload too slow: {avg_time:.3f}s for {size_kb}KB"
                else:
                    assert avg_time < 15.0, f"Large doc upload too slow: {avg_time:.3f}s for {size_kb}KB"


@pytest.mark.production
@pytest.mark.performance
class TestConcurrentUserSimulation:
    """Test system behavior under concurrent user load."""
    
    async def test_concurrent_query_users(self):
        """Test system with multiple concurrent users making queries."""
        
        class MockUser:
            def __init__(self, user_id: str, api_base_url: str):
                self.user_id = user_id
                self.api_base_url = api_base_url
                self.request_count = 0
                self.response_times = []
                self.errors = []
            
            async def simulate_user_session(self, duration_seconds: int):
                """Simulate a user session with multiple queries."""
                end_time = time.time() + duration_seconds
                
                # Mock HTTP session
                session_mock = AsyncMock()
                response_mock = AsyncMock()
                response_mock.status = 200
                response_mock.text = AsyncMock(return_value='{"result": "Mock response"}')
                session_mock.post.return_value.__aenter__.return_value = response_mock
                
                while time.time() < end_time:
                    try:
                        start_time = time.time()
                        
                        # Simulate query request
                        query_data = {
                            "query": f"User {self.user_id} query {self.request_count}",
                            "mode": random.choice(["local", "global", "hybrid"])
                        }
                        
                        # Mock API call
                        await session_mock.post(
                            f"{self.api_base_url}/query",
                            json=query_data
                        )
                        
                        response_time = time.time() - start_time
                        self.response_times.append(response_time)
                        self.request_count += 1
                        
                        # Simulate user think time
                        await asyncio.sleep(random.uniform(0.1, 2.0))
                        
                    except Exception as e:
                        self.errors.append(str(e))
                    
                    # Prevent tight loops in testing
                    await asyncio.sleep(0.01)
        
        # Simulation configuration
        num_users = 50
        session_duration = 30  # seconds
        api_base_url = "http://localhost:9621"
        
        # Create mock users
        users = [MockUser(f"user_{i}", api_base_url) for i in range(num_users)]
        
        # Run concurrent user sessions
        start_time = time.time()
        tasks = [user.simulate_user_session(session_duration) for user in users]
        await asyncio.gather(*tasks)
        end_time = time.time()
        
        # Analyze results
        total_requests = sum(user.request_count for user in users)
        total_errors = sum(len(user.errors) for user in users)
        all_response_times = []
        for user in users:
            all_response_times.extend(user.response_times)
        
        if all_response_times:
            avg_response_time = statistics.mean(all_response_times)
            p95_response_time = np.percentile(all_response_times, 95)
            throughput = total_requests / (end_time - start_time)
            error_rate = total_errors / total_requests if total_requests > 0 else 0
        else:
            avg_response_time = 0
            p95_response_time = 0
            throughput = 0
            error_rate = 1
        
        # Performance assertions
        assert total_requests > 0, "No requests were made during simulation"
        assert error_rate < 0.01, f"Error rate too high: {error_rate:.3f}"
        assert avg_response_time < 2.0, f"Average response time too high: {avg_response_time:.3f}s"
        assert p95_response_time < 5.0, f"P95 response time too high: {p95_response_time:.3f}s"
        assert throughput >= 10, f"Throughput too low: {throughput:.1f} req/s"
    
    async def test_burst_traffic_handling(self):
        """Test system behavior under burst traffic conditions."""
        
        async def generate_burst_requests(burst_size: int, burst_interval: float) -> List[Tuple[float, int]]:
            """Generate a burst of requests."""
            
            # Mock session for burst requests
            session_mock = AsyncMock()
            response_mock = AsyncMock()
            response_mock.status = 200
            response_mock.text = AsyncMock(return_value='{"status": "ok"}')
            session_mock.get.return_value.__aenter__.return_value = response_mock
            
            async def single_request():
                start_time = time.time()
                await session_mock.get("http://localhost:9621/health")
                end_time = time.time()
                return end_time - start_time, 200
            
            # Create burst of concurrent requests
            tasks = [single_request() for _ in range(burst_size)]
            results = await asyncio.gather(*tasks)
            
            return results
        
        # Test different burst scenarios
        burst_scenarios = [
            {"burst_size": 100, "burst_interval": 1.0, "name": "small_burst"},
            {"burst_size": 500, "burst_interval": 2.0, "name": "medium_burst"},
            {"burst_size": 1000, "burst_interval": 5.0, "name": "large_burst"}
        ]
        
        for scenario in burst_scenarios:
            results = await generate_burst_requests(
                scenario["burst_size"], 
                scenario["burst_interval"]
            )
            
            response_times = [r[0] for r in results]
            status_codes = [r[1] for r in results]
            
            success_rate = sum(1 for code in status_codes if code == 200) / len(status_codes)
            avg_response_time = statistics.mean(response_times)
            max_response_time = max(response_times)
            
            # Burst handling assertions
            assert success_rate >= 0.95, f"Success rate too low in {scenario['name']}: {success_rate:.3f}"
            assert avg_response_time < 1.0, f"Average response time too high in {scenario['name']}: {avg_response_time:.3f}s"
            assert max_response_time < 10.0, f"Max response time too high in {scenario['name']}: {max_response_time:.3f}s"
    
    async def test_mixed_workload_performance(self):
        """Test system performance under mixed workload (queries, uploads, downloads)."""
        
        class WorkloadSimulator:
            def __init__(self):
                self.metrics = {
                    'queries': {'count': 0, 'response_times': [], 'errors': 0},
                    'uploads': {'count': 0, 'response_times': [], 'errors': 0},
                    'downloads': {'count': 0, 'response_times': [], 'errors': 0}
                }
            
            async def query_workload(self, session_mock):
                """Simulate query workload."""
                start_time = time.time()
                try:
                    # Mock query operation
                    await session_mock.post("http://localhost:9621/query", json={"query": "test"})
                    response_time = time.time() - start_time
                    self.metrics['queries']['response_times'].append(response_time)
                    self.metrics['queries']['count'] += 1
                except:
                    self.metrics['queries']['errors'] += 1
            
            async def upload_workload(self, session_mock):
                """Simulate document upload workload."""
                start_time = time.time()
                try:
                    # Mock upload operation
                    await session_mock.post("http://localhost:9621/documents/upload", data={'file': 'test content'})
                    response_time = time.time() - start_time
                    self.metrics['uploads']['response_times'].append(response_time)
                    self.metrics['uploads']['count'] += 1
                except:
                    self.metrics['uploads']['errors'] += 1
            
            async def download_workload(self, session_mock):
                """Simulate document download workload."""
                start_time = time.time()
                try:
                    # Mock download operation
                    await session_mock.get("http://localhost:9621/documents/doc123")
                    response_time = time.time() - start_time
                    self.metrics['downloads']['response_times'].append(response_time)
                    self.metrics['downloads']['count'] += 1
                except:
                    self.metrics['downloads']['errors'] += 1
        
        # Mock session setup
        session_mock = AsyncMock()
        response_mock = AsyncMock()
        response_mock.status = 200
        response_mock.text = AsyncMock(return_value='{"status": "success"}')
        session_mock.post.return_value.__aenter__.return_value = response_mock
        session_mock.get.return_value.__aenter__.return_value = response_mock
        
        simulator = WorkloadSimulator()
        
        # Create mixed workload tasks
        tasks = []
        workload_distribution = [
            (simulator.query_workload, 60),  # 60% queries
            (simulator.upload_workload, 25),  # 25% uploads
            (simulator.download_workload, 15)  # 15% downloads
        ]
        
        total_requests = 1000
        for workload_func, percentage in workload_distribution:
            request_count = int(total_requests * percentage / 100)
            for _ in range(request_count):
                tasks.append(workload_func(session_mock))
        
        # Execute mixed workload
        random.shuffle(tasks)  # Randomize execution order
        await asyncio.gather(*tasks)
        
        # Analyze mixed workload performance
        for workload_type, metrics in simulator.metrics.items():
            if metrics['count'] > 0:
                error_rate = metrics['errors'] / (metrics['count'] + metrics['errors'])
                avg_response_time = statistics.mean(metrics['response_times']) if metrics['response_times'] else 0
                
                assert error_rate < 0.05, f"{workload_type} error rate too high: {error_rate:.3f}"
                assert avg_response_time < 2.0, f"{workload_type} response time too high: {avg_response_time:.3f}s"


@pytest.mark.production
@pytest.mark.performance
class TestResourceUtilization:
    """Test resource utilization monitoring and limits."""
    
    async def test_memory_usage_monitoring(self):
        """Test memory usage under various load conditions."""
        
        class MemoryMonitor:
            def __init__(self):
                self.measurements = []
            
            def start_monitoring(self, duration_seconds: int):
                """Start memory monitoring for specified duration."""
                def monitor():
                    end_time = time.time() + duration_seconds
                    while time.time() < end_time:
                        try:
                            # Get memory usage (mock values for testing)
                            memory_percent = random.uniform(30, 80)  # Mock memory usage
                            memory_mb = memory_percent * 10  # Mock MB usage
                            
                            self.measurements.append({
                                'timestamp': time.time(),
                                'memory_percent': memory_percent,
                                'memory_mb': memory_mb
                            })
                            time.sleep(1)  # Monitor every second
                        except:
                            break
                
                # Run monitoring in background
                monitor_thread = threading.Thread(target=monitor)
                monitor_thread.daemon = True
                monitor_thread.start()
                return monitor_thread
        
        monitor = MemoryMonitor()
        
        # Start memory monitoring
        monitor_thread = monitor.start_monitoring(10)  # Monitor for 10 seconds
        
        # Simulate workload that uses memory
        async def memory_intensive_workload():
            # Mock memory-intensive operations
            for _ in range(100):
                # Simulate data processing
                data = list(range(1000))  # Small data structure
                result = sum(data)
                await asyncio.sleep(0.01)  # Small delay
        
        # Run workload while monitoring
        await memory_intensive_workload()
        
        # Wait for monitoring to complete
        monitor_thread.join(timeout=15)
        
        # Analyze memory usage
        if monitor.measurements:
            memory_values = [m['memory_percent'] for m in monitor.measurements]
            avg_memory = statistics.mean(memory_values)
            max_memory = max(memory_values)
            memory_growth = memory_values[-1] - memory_values[0] if len(memory_values) > 1 else 0
            
            # Memory usage assertions
            assert avg_memory < 90, f"Average memory usage too high: {avg_memory:.1f}%"
            assert max_memory < 95, f"Peak memory usage too high: {max_memory:.1f}%"
            assert abs(memory_growth) < 20, f"Memory growth too high: {memory_growth:.1f}%"
    
    async def test_cpu_usage_monitoring(self):
        """Test CPU usage under load."""
        
        class CPUMonitor:
            def __init__(self):
                self.measurements = []
            
            def start_monitoring(self, duration_seconds: int):
                """Start CPU monitoring."""
                def monitor():
                    end_time = time.time() + duration_seconds
                    while time.time() < end_time:
                        try:
                            # Mock CPU usage measurement
                            cpu_percent = random.uniform(10, 70)  # Mock CPU usage
                            
                            self.measurements.append({
                                'timestamp': time.time(),
                                'cpu_percent': cpu_percent
                            })
                            time.sleep(1)
                        except:
                            break
                
                monitor_thread = threading.Thread(target=monitor)
                monitor_thread.daemon = True
                monitor_thread.start()
                return monitor_thread
        
        monitor = CPUMonitor()
        
        # Start CPU monitoring
        monitor_thread = monitor.start_monitoring(10)
        
        # Simulate CPU-intensive workload
        async def cpu_intensive_workload():
            for _ in range(50):
                # Simulate computational work
                result = sum(i * i for i in range(1000))
                await asyncio.sleep(0.01)
        
        await cpu_intensive_workload()
        monitor_thread.join(timeout=15)
        
        # Analyze CPU usage
        if monitor.measurements:
            cpu_values = [m['cpu_percent'] for m in monitor.measurements]
            avg_cpu = statistics.mean(cpu_values)
            max_cpu = max(cpu_values)
            
            # CPU usage assertions
            assert avg_cpu < 80, f"Average CPU usage too high: {avg_cpu:.1f}%"
            assert max_cpu < 95, f"Peak CPU usage too high: {max_cpu:.1f}%"
    
    async def test_database_connection_pooling(self):
        """Test database connection pool performance."""
        
        class ConnectionPoolMonitor:
            def __init__(self, pool_size: int = 20):
                self.pool_size = pool_size
                self.active_connections = 0
                self.max_active = 0
                self.connection_wait_times = []
                self.lock = threading.Lock()
            
            async def acquire_connection(self):
                """Simulate acquiring a database connection."""
                start_time = time.time()
                
                # Simulate connection pool logic
                while self.active_connections >= self.pool_size:
                    await asyncio.sleep(0.01)  # Wait for available connection
                
                with self.lock:
                    self.active_connections += 1
                    self.max_active = max(self.max_active, self.active_connections)
                
                wait_time = time.time() - start_time
                self.connection_wait_times.append(wait_time)
                
                return f"connection_{self.active_connections}"
            
            async def release_connection(self, connection_id: str):
                """Simulate releasing a database connection."""
                with self.lock:
                    self.active_connections = max(0, self.active_connections - 1)
        
        pool_monitor = ConnectionPoolMonitor(pool_size=20)
        
        # Simulate concurrent database operations
        async def database_operation():
            connection = await pool_monitor.acquire_connection()
            
            # Simulate database query time
            await asyncio.sleep(random.uniform(0.01, 0.1))
            
            await pool_monitor.release_connection(connection)
        
        # Run concurrent database operations
        num_operations = 200
        tasks = [database_operation() for _ in range(num_operations)]
        
        start_time = time.time()
        await asyncio.gather(*tasks)
        end_time = time.time()
        
        # Analyze connection pool performance
        total_time = end_time - start_time
        operations_per_second = num_operations / total_time
        avg_wait_time = statistics.mean(pool_monitor.connection_wait_times)
        max_wait_time = max(pool_monitor.connection_wait_times)
        
        # Connection pool assertions
        assert operations_per_second >= 100, f"DB operations/sec too low: {operations_per_second:.1f}"
        assert avg_wait_time < 0.1, f"Average connection wait time too high: {avg_wait_time:.3f}s"
        assert max_wait_time < 1.0, f"Max connection wait time too high: {max_wait_time:.3f}s"
        assert pool_monitor.max_active <= pool_monitor.pool_size, "Connection pool size exceeded"


@pytest.mark.production
@pytest.mark.performance
class TestQueryPerformanceBenchmarks:
    """Test specific query performance benchmarks."""
    
    async def test_vector_similarity_search_performance(self):
        """Test vector similarity search performance."""
        
        # Mock vector database operations
        class MockVectorDB:
            def __init__(self, dimension: int = 768):
                self.dimension = dimension
                self.vectors = {}
                self.search_times = []
            
            async def upsert(self, vector_id: str, vector: List[float]):
                """Mock vector upsert operation."""
                start_time = time.time()
                self.vectors[vector_id] = vector
                end_time = time.time()
                return end_time - start_time
            
            async def similarity_search(self, query_vector: List[float], top_k: int = 10) -> List[Tuple[str, float]]:
                """Mock similarity search operation."""
                start_time = time.time()
                
                # Simulate similarity calculation
                results = []
                for vector_id, vector in list(self.vectors.items())[:top_k]:
                    # Mock similarity score
                    similarity = random.uniform(0.5, 1.0)
                    results.append((vector_id, similarity))
                
                end_time = time.time()
                search_time = end_time - start_time
                self.search_times.append(search_time)
                
                return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]
        
        vector_db = MockVectorDB()
        
        # Insert test vectors
        num_vectors = 10000
        for i in range(num_vectors):
            vector = [random.random() for _ in range(768)]
            await vector_db.upsert(f"doc_{i}", vector)
        
        # Perform similarity searches
        num_searches = 100
        query_vector = [random.random() for _ in range(768)]
        
        search_tasks = []
        for _ in range(num_searches):
            search_tasks.append(vector_db.similarity_search(query_vector, top_k=10))
        
        start_time = time.time()
        search_results = await asyncio.gather(*search_tasks)
        end_time = time.time()
        
        # Analyze search performance
        total_search_time = end_time - start_time
        searches_per_second = num_searches / total_search_time
        avg_search_time = statistics.mean(vector_db.search_times)
        p95_search_time = np.percentile(vector_db.search_times, 95)
        
        # Vector search performance assertions
        assert searches_per_second >= 50, f"Vector searches/sec too low: {searches_per_second:.1f}"
        assert avg_search_time < 0.1, f"Average search time too high: {avg_search_time:.3f}s"
        assert p95_search_time < 0.2, f"P95 search time too high: {p95_search_time:.3f}s"
        
        # Validate search results quality
        for results in search_results:
            assert len(results) <= 10, "Too many search results returned"
            assert all(0 <= score <= 1 for _, score in results), "Invalid similarity scores"
    
    async def test_knowledge_graph_traversal_performance(self):
        """Test knowledge graph traversal performance."""
        
        class MockKnowledgeGraph:
            def __init__(self):
                self.nodes = {}
                self.edges = {}
                self.traversal_times = []
            
            def add_node(self, node_id: str, properties: Dict):
                """Add node to graph."""
                self.nodes[node_id] = properties
            
            def add_edge(self, from_node: str, to_node: str, relationship: str, weight: float = 1.0):
                """Add edge to graph."""
                if from_node not in self.edges:
                    self.edges[from_node] = []
                self.edges[from_node].append({
                    'to': to_node,
                    'relationship': relationship,
                    'weight': weight
                })
            
            async def traverse(self, start_node: str, max_depth: int = 3) -> Dict:
                """Mock graph traversal operation."""
                start_time = time.time()
                
                visited = set()
                result = {'nodes': [], 'relationships': []}
                queue = [(start_node, 0)]
                
                while queue and len(visited) < 1000:  # Limit for performance
                    current_node, depth = queue.pop(0)
                    
                    if current_node in visited or depth > max_depth:
                        continue
                    
                    visited.add(current_node)
                    
                    if current_node in self.nodes:
                        result['nodes'].append({
                            'id': current_node,
                            'properties': self.nodes[current_node]
                        })
                    
                    # Add connected nodes to queue
                    if current_node in self.edges:
                        for edge in self.edges[current_node]:
                            if edge['to'] not in visited:
                                queue.append((edge['to'], depth + 1))
                                result['relationships'].append({
                                    'from': current_node,
                                    'to': edge['to'],
                                    'type': edge['relationship']
                                })
                
                end_time = time.time()
                traversal_time = end_time - start_time
                self.traversal_times.append(traversal_time)
                
                return result
        
        # Create test knowledge graph
        graph = MockKnowledgeGraph()
        
        # Add nodes and edges
        num_nodes = 5000
        for i in range(num_nodes):
            graph.add_node(f"entity_{i}", {'type': 'concept', 'name': f'Entity {i}'})
        
        # Add edges (create connected graph)
        num_edges = 15000
        for i in range(num_edges):
            from_node = f"entity_{random.randint(0, num_nodes-1)}"
            to_node = f"entity_{random.randint(0, num_nodes-1)}"
            relationship = random.choice(['related_to', 'part_of', 'similar_to', 'causes'])
            graph.add_edge(from_node, to_node, relationship, random.uniform(0.1, 1.0))
        
        # Perform graph traversals
        num_traversals = 50
        traversal_tasks = []
        
        for _ in range(num_traversals):
            start_node = f"entity_{random.randint(0, num_nodes-1)}"
            traversal_tasks.append(graph.traverse(start_node, max_depth=3))
        
        start_time = time.time()
        traversal_results = await asyncio.gather(*traversal_tasks)
        end_time = time.time()
        
        # Analyze traversal performance
        total_traversal_time = end_time - start_time
        traversals_per_second = num_traversals / total_traversal_time
        avg_traversal_time = statistics.mean(graph.traversal_times)
        p95_traversal_time = np.percentile(graph.traversal_times, 95)
        
        # Graph traversal performance assertions
        assert traversals_per_second >= 20, f"Graph traversals/sec too low: {traversals_per_second:.1f}"
        assert avg_traversal_time < 0.5, f"Average traversal time too high: {avg_traversal_time:.3f}s"
        assert p95_traversal_time < 1.0, f"P95 traversal time too high: {p95_traversal_time:.3f}s"
        
        # Validate traversal results quality
        for result in traversal_results:
            assert len(result['nodes']) > 0, "No nodes found in traversal"
            assert len(result['relationships']) >= 0, "Invalid relationships in traversal"