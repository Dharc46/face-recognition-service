import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import Security, HTTPException
from src.application import app, get_db
import os  # Thêm import


class TestAPIAuth(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

        # Create API key patch
        self.api_key_patcher = patch('os.getenv')
        self.mock_getenv = self.api_key_patcher.start()
        self.mock_getenv.return_value = "test_api_key"  # Trả về giá trị mock

        # Create a mock dependency to simulate API key verification
        async def verify_api_key(api_key: str = Security(...)):
            if api_key != "test_api_key":
                raise HTTPException(status_code=403, detail="Invalid API key")
            return api_key

        # Mock the get_db dependency to avoid actual DB connections
        def override_get_db():
            db = MagicMock()
            try:
                yield db
            finally:
                pass

        # Apply dependency overrides
        app.dependency_overrides[get_db] = override_get_db

    def tearDown(self):
        self.api_key_patcher.stop()
        app.dependency_overrides = {}

    def test_auth_required_for_protected_endpoints(self):
        # Gửi request không có API key nhưng có đủ các tham số bắt buộc
        dummy_file = ("test.jpg", b"dummy", "image/jpeg")
        
        # Test recognition endpoint
        response = self.client.post(
            "/recognition",
            files={"image": dummy_file}
        )
        self.assertEqual(response.status_code, 403)  # 👈 Giờ sẽ pass

    def test_auth_with_invalid_api_key(self):
        # Mock service xử lý ảnh để tập trung test API key
        with patch('src.application.FaceRecognitionService.detect_faces') as mock_detect:
            mock_detect.return_value = []  # Giả lập xử lý thành công
            
            headers = {"X-API-Key": "invalid_key"}
            response = self.client.post(
                "/recognition",
                headers=headers,
                files={"image": ("test.jpg", b"dummy", "image/jpeg")}
            )
            self.assertEqual(response.status_code, 403)  # 👈 Giờ sẽ pass

    def test_auth_with_invalid_api_key(self):
        # Mock FaceRecognitionService để tránh lỗi xử lý ảnh
        with patch('src.application.FaceRecognitionService.detect_faces') as mock_detect:
            mock_detect.return_value = []  # Giả lập xử lý thành công

            headers = {"X-API-Key": "invalid_key"}
            response = self.client.post(
                "/recognition",
                headers=headers,
                files={"image": ("test.jpg", b"test image content", "image/jpeg")}
            )
            self.assertEqual(response.status_code, 403)  # Giờ sẽ kiểm tra API key

    def test_health_endpoint_no_auth(self):
        # Health endpoint should be accessible without auth
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})