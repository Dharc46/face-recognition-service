
"""
Performance tests for Face Recognition Service
"""
import pytest
import requests
import time
import json
import os
import threading
import psutil
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from PIL import Image
import tempfile
import numpy as np


class PerformanceTestConfig:
    """Configuration for performance tests"""
    BASE_URL = "http://localhost:8000"
    TEST_ITERATIONS = 10
    CONCURRENT_USERS = [1, 5, 10, 20]
    LARGE_FILE_SIZE_MB = 5
    TIMEOUT = 30  # seconds


class TestImageGenerator:
    """Generate test images for performance testing"""
    
    @staticmethod
    def create_test_image(width=640, height=480, format='JPEG'):
        """Create a test image in memory"""
        # Create a random RGB image
        image_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        image = Image.fromarray(image_array)
        
        # Save to BytesIO
        img_buffer = BytesIO()
        image.save(img_buffer, format=format)
        img_buffer.seek(0)
        
        return img_buffer
    
    @staticmethod
    def create_large_test_image(size_mb=5):
        """Create a large test image"""
        # Calculate dimensions for approximately the target size
        target_pixels = (size_mb * 1024 * 1024) // 3  # 3 bytes per RGB pixel
        width = int(np.sqrt(target_pixels))
        height = width
        
        return TestImageGenerator.create_test_image(width, height)


class PerformanceMetrics:
    """Collect and analyze performance metrics"""
    
    def __init__(self):
        self.response_times = []
        self.cpu_usage = []
        self.memory_usage = []
        self.error_count = 0
        self.success_count = 0
    
    def add_response_time(self, response_time):
        self.response_times.append(response_time)
    
    def add_system_metrics(self):
        self.cpu_usage.append(psutil.cpu_percent())
        self.memory_usage.append(psutil.virtual_memory().percent)
    
    def add_result(self, success):
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
    
    def get_statistics(self):
        if not self.response_times:
            return None
        
        return {
            'response_times': {
                'min': min(self.response_times),
                'max': max(self.response_times),
                'mean': statistics.mean(self.response_times),
                'median': statistics.median(self.response_times),
                'p95': np.percentile(self.response_times, 95),
                'p99': np.percentile(self.response_times, 99)
            },
            'system_metrics': {
                'avg_cpu': statistics.mean(self.cpu_usage) if self.cpu_usage else 0,
                'max_cpu': max(self.cpu_usage) if self.cpu_usage else 0,
                'avg_memory': statistics.mean(self.memory_usage) if self.memory_usage else 0,
                'max_memory': max(self.memory_usage) if self.memory_usage else 0
            },
            'success_rate': self.success_count / (self.success_count + self.error_count) * 100,
            'total_requests': self.success_count + self.error_count
        }


class TestPerformanceBasic:
    """Basic performance tests for individual endpoints"""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup for each test method"""
        self.config = PerformanceTestConfig()
        self.base_url = self.config.BASE_URL
        
        # Verify service is running
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            assert response.status_code == 200
        except requests.exceptions.RequestException:
            pytest.skip("Service is not running")
    
    def test_health_endpoint_performance(self):
        """Test health endpoint response time"""
        metrics = PerformanceMetrics()
        
        for _ in range(self.config.TEST_ITERATIONS):
            start_time = time.time()
            response = requests.get(f"{self.base_url}/health")
            end_time = time.time()
            
            response_time = (end_time - start_time) * 1000  # Convert to ms
            metrics.add_response_time(response_time)
            metrics.add_result(response.status_code == 200)
            metrics.add_system_metrics()
        
        stats = metrics.get_statistics()
        print(f"\n=== Health Endpoint Performance ===")
        print(f"Average response time: {stats['response_times']['mean']:.2f}ms")
        print(f"95th percentile: {stats['response_times']['p95']:.2f}ms")
        print(f"Success rate: {stats['success_rate']:.1f}%")
        
        # Assert performance requirements
        assert stats['response_times']['mean'] < 2060  # Less than 100ms average
        assert stats['success_rate'] == 100  # 100% success rate
    
    def test_recognition_endpoint_performance(self):
        """Test recognition endpoint with different image sizes"""
        image_sizes = [
            ('small', 320, 240),
            ('medium', 640, 480),
            ('large', 1280, 720)
        ]
        
        for size_name, width, height in image_sizes:
            metrics = PerformanceMetrics()
            
            for _ in range(5):  # Fewer iterations for recognition tests
                # Create test image
                test_image = TestImageGenerator.create_test_image(width, height)
                
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/recognition",
                    files={'image': ('test.jpg', test_image, 'image/jpeg')},
                    timeout=self.config.TIMEOUT
                )
                end_time = time.time()
                
                response_time = (end_time - start_time) * 1000
                metrics.add_response_time(response_time)
                metrics.add_result(response.status_code == 200)
                metrics.add_system_metrics()
                
                # Reset image buffer
                test_image.seek(0)
            
            stats = metrics.get_statistics()
            print(f"\n=== Recognition Performance ({size_name} {width}x{height}) ===")
            print(f"Average response time: {stats['response_times']['mean']:.2f}ms")
            print(f"95th percentile: {stats['response_times']['p95']:.2f}ms")
            print(f"Success rate: {stats['success_rate']:.1f}%")
            print(f"Average CPU usage: {stats['system_metrics']['avg_cpu']:.1f}%")
            print(f"Average Memory usage: {stats['system_metrics']['avg_memory']:.1f}%")
    
    def test_register_endpoint_performance(self):
        """Test registration endpoint performance"""
        metrics = PerformanceMetrics()
        
        for i in range(3):  # Limited iterations for registration
            # Create multiple test images for registration
            test_images = []
            for j in range(3):
                img = TestImageGenerator.create_test_image(640, 480)
                test_images.append(('images', (f'test_{j}.jpg', img, 'image/jpeg')))
            
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/register",
                data={'name': f'test_person_{i}'},
                files=test_images,
                timeout=self.config.TIMEOUT
            )
            end_time = time.time()
            
            response_time = (end_time - start_time) * 1000
            metrics.add_response_time(response_time)
            metrics.add_result(response.status_code == 200)
            metrics.add_system_metrics()
            
            # Reset image buffers
            for _, (_, img, _) in test_images:
                img.seek(0)
        
        stats = metrics.get_statistics()
        print(f"\n=== Registration Performance ===")
        print(f"Average response time: {stats['response_times']['mean']:.2f}ms")
        print(f"95th percentile: {stats['response_times']['p95']:.2f}ms")
        print(f"Success rate: {stats['success_rate']:.1f}%")
        print(f"Max CPU usage: {stats['system_metrics']['max_cpu']:.1f}%")
        print(f"Max Memory usage: {stats['system_metrics']['max_memory']:.1f}%")


class TestPerformanceConcurrency:
    """Test concurrent request handling"""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.config = PerformanceTestConfig()
        self.base_url = self.config.BASE_URL
    
    def make_recognition_request(self):
        """Make a single recognition request"""
        test_image = TestImageGenerator.create_test_image(640, 480)
        
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.base_url}/recognition",
                files={'image': ('test.jpg', test_image, 'image/jpeg')},
                timeout=self.config.TIMEOUT
            )
            end_time = time.time()
            return {
                'success': response.status_code == 200,
                'response_time': (end_time - start_time) * 1000,
                'status_code': response.status_code
            }
        except Exception as e:
            end_time = time.time()
            return {
                'success': False,
                'response_time': (end_time - start_time) * 1000,
                'error': str(e)
            }
    
    def make_health_request(self):
        """Make a single health check request"""
        start_time = time.time()
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            end_time = time.time()
            return {
                'success': response.status_code == 200,
                'response_time': (end_time - start_time) * 1000,
                'status_code': response.status_code
            }
        except Exception as e:
            end_time = time.time()
            return {
                'success': False,
                'response_time': (end_time - start_time) * 1000,
                'error': str(e)
            }
    
    @pytest.mark.parametrize("concurrent_users", [1, 5, 10])
    def test_concurrent_health_requests(self, concurrent_users):
        """Test concurrent health check requests"""
        results = []
        
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            # Submit concurrent requests
            futures = [executor.submit(self.make_health_request) 
                      for _ in range(concurrent_users * 5)]
            
            # Collect results
            for future in as_completed(futures):
                results.append(future.result())
        
        # Analyze results
        response_times = [r['response_time'] for r in results]
        success_rate = sum(1 for r in results if r['success']) / len(results) * 100
        
        print(f"\n=== Concurrent Health Requests ({concurrent_users} users) ===")
        print(f"Total requests: {len(results)}")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Average response time: {statistics.mean(response_times):.2f}ms")
        print(f"95th percentile: {np.percentile(response_times, 95):.2f}ms")
        
        assert success_rate >= 95  # At least 95% success rate
    
    @pytest.mark.parametrize("concurrent_users", [1, 3, 5])
    def test_concurrent_recognition_requests(self, concurrent_users):
        """Test concurrent recognition requests"""
        results = []
        
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            # Submit concurrent requests (fewer for recognition due to resource intensity)
            futures = [executor.submit(self.make_recognition_request) 
                      for _ in range(concurrent_users * 2)]
            
            # Collect results
            for future in as_completed(futures):
                results.append(future.result())
        
        # Analyze results
        response_times = [r['response_time'] for r in results]
        success_rate = sum(1 for r in results if r['success']) / len(results) * 100
        
        print(f"\n=== Concurrent Recognition Requests ({concurrent_users} users) ===")
        print(f"Total requests: {len(results)}")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Average response time: {statistics.mean(response_times):.2f}ms")
        print(f"95th percentile: {np.percentile(response_times, 95):.2f}ms")
        
        assert success_rate >= 80  # At least 80% success rate for recognition


class TestPerformanceLoad:
    """Load testing scenarios"""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.config = PerformanceTestConfig()
        self.base_url = self.config.BASE_URL
    
    def test_sustained_load_health(self):
        """Test sustained load on health endpoint"""
        duration_seconds = 30
        requests_per_second = 10
        
        results = []
        start_time = time.time()
        
        def make_request():
            return requests.get(f"{self.base_url}/health", timeout=5)
        
        while time.time() - start_time < duration_seconds:
            batch_start = time.time()
            
            # Make batch of requests
            with ThreadPoolExecutor(max_workers=requests_per_second) as executor:
                futures = [executor.submit(make_request) for _ in range(requests_per_second)]
                
                for future in as_completed(futures):
                    try:
                        response = future.result()
                        results.append({
                            'success': response.status_code == 200,
                            'timestamp': time.time()
                        })
                    except Exception as e:
                        results.append({
                            'success': False,
                            'error': str(e),
                            'timestamp': time.time()
                        })
            
            # Wait for next second
            elapsed = time.time() - batch_start
            if elapsed < 1.0:
                time.sleep(1.0 - elapsed)
        
        # Analyze results
        total_requests = len(results)
        successful_requests = sum(1 for r in results if r['success'])
        success_rate = successful_requests / total_requests * 100
        actual_rps = total_requests / duration_seconds
        
        print(f"\n=== Sustained Load Test Results ===")
        print(f"Duration: {duration_seconds}s")
        print(f"Target RPS: {requests_per_second}")
        print(f"Actual RPS: {actual_rps:.2f}")
        print(f"Total requests: {total_requests}")
        print(f"Success rate: {success_rate:.1f}%")
        
        assert success_rate >= 95
        assert actual_rps >= requests_per_second * 0.5  # Within 10% of target


class TestPerformanceMemory:
    """Memory and resource usage tests"""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.config = PerformanceTestConfig()
        self.base_url = self.config.BASE_URL
    
    def test_memory_usage_under_load(self):
        """Monitor memory usage under load"""
        initial_memory = psutil.virtual_memory().percent
        memory_samples = [initial_memory]
        
        def monitor_memory(stop_event):
            while not stop_event.is_set():
                memory_samples.append(psutil.virtual_memory().percent)
                time.sleep(0.5)
        
        # Start memory monitoring
        stop_event = threading.Event()
        monitor_thread = threading.Thread(target=monitor_memory, args=(stop_event,))
        monitor_thread.start()
        
        try:
            # Generate load
            results = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = []
                
                for _ in range(20):
                    test_image = TestImageGenerator.create_test_image(640, 480)
                    future = executor.submit(
                        requests.post,
                        f"{self.base_url}/recognition",
                        files={'image': ('test.jpg', test_image, 'image/jpeg')},
                        timeout=self.config.TIMEOUT
                    )
                    futures.append(future)
                
                for future in as_completed(futures):
                    try:
                        response = future.result()
                        results.append(response.status_code == 200)
                    except:
                        results.append(False)
        
        finally:
            # Stop monitoring
            stop_event.set()
            monitor_thread.join()
        
        # Analyze memory usage
        max_memory = max(memory_samples)
        avg_memory = statistics.mean(memory_samples)
        memory_increase = max_memory - initial_memory
        
        print(f"\n=== Memory Usage Analysis ===")
        print(f"Initial memory usage: {initial_memory:.1f}%")
        print(f"Maximum memory usage: {max_memory:.1f}%")
        print(f"Average memory usage: {avg_memory:.1f}%")
        print(f"Memory increase: {memory_increase:.1f}%")
        print(f"Success rate: {sum(results)/len(results)*100:.1f}%")
        
        # Assert memory doesn't grow excessively
        assert memory_increase < 20  # Less than 20% memory increase


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

