
"""
Locust load testing script for Face Recognition Service
Run with: locust -f locust_load_test.py --host=http://localhost:8000
"""
from locust import HttpUser, task, between
from io import BytesIO
from PIL import Image
import numpy as np
import random


class TestImageGenerator:
    """Generate test images for load testing"""
    
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
    def get_random_image_size():
        """Get random image dimensions for varied testing"""
        sizes = [
            (320, 240),   # Small
            (640, 480),   # Medium
            (800, 600),   # Large
            (1024, 768),  # Extra Large
        ]
        return random.choice(sizes)


class FaceRecognitionUser(HttpUser):
    """User behavior for face recognition service"""
    
    # Wait between 1 and 3 seconds between requests
    wait_time = between(1, 3)
    
    def on_start(self):
        """Called when a user starts"""
        # Check if service is healthy
        response = self.client.get("/health")
        if response.status_code != 200:
            print("Service health check failed!")
    
    @task(10)
    def health_check(self):
        """Health check endpoint - high frequency"""
        self.client.get("/health")
    
    @task(5)
    def list_faces(self):
        """List registered faces"""
        self.client.get("/faces")
    
    @task(3)
    def recognize_face_small(self):
        """Recognize face with small image"""
        width, height = 320, 240
        test_image = TestImageGenerator.create_test_image(width, height)
        
        with self.client.post(
            "/recognition",
            files={'image': ('test.jpg', test_image, 'image/jpeg')},
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Recognition failed with status {response.status_code}")
    
    @task(2)
    def recognize_face_medium(self):
        """Recognize face with medium image"""
        width, height = 640, 480
        test_image = TestImageGenerator.create_test_image(width, height)
        
        with self.client.post(
            "/recognition",
            files={'image': ('test.jpg', test_image, 'image/jpeg')},
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Recognition failed with status {response.status_code}")
    
    @task(1)
    def recognize_face_large(self):
        """Recognize face with large image"""
        width, height = 1024, 768
        test_image = TestImageGenerator.create_test_image(width, height)
        
        with self.client.post(
            "/recognition",
            files={'image': ('test.jpg', test_image, 'image/jpeg')},
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Recognition failed with status {response.status_code}")


class HeavyUser(HttpUser):
    """Heavy user that performs resource-intensive operations"""
    
    wait_time = between(2, 5)
    
    @task(1)
    def register_new_face(self):
        """Register a new face - resource intensive"""
        # Create multiple images for registration
        test_images = []
        for i in range(3):
            width, height = TestImageGenerator.get_random_image_size()
            img = TestImageGenerator.create_test_image(width, height)
            test_images.append(('images', (f'test_{i}.jpg', img, 'image/jpeg')))
        
        user_id = random.randint(1000, 9999)
        
        with self.client.post(
            "/register",
            data={'name': f'load_test_user_{user_id}'},
            files=test_images,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Registration failed with status {response.status_code}")
    
    @task(2)
    def batch_recognition(self):
        """Perform multiple recognitions in sequence"""
        for _ in range(3):
            width, height = TestImageGenerator.get_random_image_size()
            test_image = TestImageGenerator.create_test_image(width, height)
            
            self.client.post(
                "/recognition",
                files={'image': ('test.jpg', test_image, 'image/jpeg')}
            )


class APIOnlyUser(HttpUser):
    """User that only uses API endpoints (no recognition)"""
    
    wait_time = between(0.5, 2)
    
    @task(5)
    def health_check(self):
        """Frequent health checks"""
        self.client.get("/health")
    
    @task(3)
    def list_faces(self):
        """List registered faces"""
        self.client.get("/faces")


# Custom Locust test scenarios
class StressTestUser(HttpUser):
    """User for stress testing - rapid requests"""
    
    wait_time = between(0.1, 0.5)  # Very short wait times
    
    @task
    def rapid_health_checks(self):
        """Rapid health check requests"""
        self.client.get("/health")
    
    @task
    def rapid_recognition(self):
        """Rapid recognition requests with small images"""
        test_image = TestImageGenerator.create_test_image(320, 240)
        
        self.client.post(
            "/recognition",
            files={'image': ('test.jpg', test_image, 'image/jpeg')}
        )


# Example custom test shapes for different scenarios
class SpikeMixin:
    """Mixin for spike testing behavior"""
    
    def spike_behavior(self):
        """Simulate traffic spikes"""
        # Normal operation
        for _ in range(5):
            self.client.get("/health")
            self.wait()
        
        # Sudden spike
        for _ in range(20):
            self.client.get("/health")
        
        # Back to normal
        for _ in range(5):
            self.client.get("/health")
            self.wait()


# Usage examples and comments
"""
Different ways to run this load test:

1. Basic load test:
   locust -f locust_load_test.py --host=http://localhost:8000

2. Headless mode with specific parameters:
   locust -f locust_load_test.py --host=http://localhost:8000 --users 10 --spawn-rate 2 --run-time 60s --headless

3. Test specific user class:
   locust -f locust_load_test.py --host=http://localhost:8000 FaceRecognitionUser

4. Multiple user types:
   locust -f locust_load_test.py --host=http://localhost:8000 FaceRecognitionUser HeavyUser

5. Stress test:
   locust -f locust_load_test.py --host=http://localhost:8000 StressTestUser --users 50 --spawn-rate 10

6. CSV output for analysis:
   locust -f locust_load_test.py --host=http://localhost:8000 --csv=results --headless --users 20 --spawn-rate 5 --run-time 120s

Test scenarios:
- FaceRecognitionUser: Normal user behavior with mixed requests
- HeavyUser: Resource-intensive operations (registration, batch processing)
- APIOnlyUser: Light API usage without image processing
- StressTestUser: High-frequency requests for stress testing
"""

