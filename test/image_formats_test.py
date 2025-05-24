import unittest
import os
import tempfile
import cv2
import numpy as np
from unittest.mock import patch, MagicMock
import shutil
from PIL import Image
from fastapi.testclient import TestClient
from src.application import app, FaceRecognitionService

class TestImageFormats(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.jpg_path = os.path.join(self.test_dir, "test.jpg")
        self.png_path = os.path.join(self.test_dir, "test.png")
        self.bmp_path = os.path.join(self.test_dir, "test.bmp")
        self.webp_path = os.path.join(self.test_dir, "test.webp")
        self.tiff_path = os.path.join(self.test_dir, "test.tiff")

        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 127
        cv2.imwrite(self.jpg_path, test_image)
        cv2.imwrite(self.png_path, test_image)
        cv2.imwrite(self.bmp_path, test_image)
        Image.fromarray(test_image).save(self.webp_path, format="WEBP")
        Image.fromarray(test_image).save(self.tiff_path, format="TIFF")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch('src.application.align.detect_face.detect_face')
    def test_face_detection_with_different_formats(self, mock_detect_face):
        face_box = np.array([[20, 20, 80, 80, 0.99]])
        mock_detect_face.return_value = (face_box, None)

        mock_pnet = MagicMock()
        mock_rnet = MagicMock()
        mock_onet = MagicMock()
        
        with patch('tensorflow.Graph'), \
             patch('tensorflow.compat.v1.Session'), \
             patch('src.application.align.detect_face.create_mtcnn') as mock_create_mtcnn, \
             patch('pickle.load') as mock_pickle:
            
            mock_create_mtcnn.return_value = (mock_pnet, mock_rnet, mock_onet)
            
            model_mock = MagicMock()
            model_mock.predict_proba.return_value = np.array([[0.2, 0.8]])
            mock_pickle.return_value = (model_mock, ["person1", "person2"])

            service = FaceRecognitionService()

            for fmt in ["jpg", "png", "bmp", "webp", "tiff"]:
                result = service.detect_faces(getattr(self, f"{fmt}_path"))
                self.assertEqual(len(result), 1)
                self.assertEqual(result[0]["name"], "person2")

    def test_api_endpoint_with_different_formats(self):
        client = TestClient(app)
        headers = {"X-API-Key": "test_api_key"}

        # Mock API key validation
        with patch('os.getenv', return_value="test_api_key"):
            # Test với các định dạng ảnh hợp lệ, mock detect_faces
            with patch('src.application.FaceRecognitionService.detect_faces') as mock_detect:
                mock_detect.return_value = [{"name": "test_person", "confidence": 0.95, "bbox": [10, 10, 50, 50]}]

                for fmt in ["jpg", "png"]:
                    with open(getattr(self, f"{fmt}_path"), "rb") as f:
                        response = client.post(
                            "/recognition",
                            headers=headers,
                            files={"image": (f"test.{fmt}", f, f"image/{fmt}")}
                        )
                        # In nội dung JSON của phản hồi để gỡ lỗi
                        print(f"Phản hồi JSON cho {fmt}: {response.json()}") 
                        self.assertEqual(response.status_code, 200)
                        self.assertTrue(isinstance(response.json(), dict)) # Kiểm tra rằng nó là một từ điển
                        self.assertIn("best_match", response.json()) # Kiểm tra có khóa 'best_match'
                        self.assertEqual(response.json()["best_match"]["name"], "test_person")
                        self.assertEqual(response.json()["best_match"]["confidence"], 0.95)

            # Test với file text, KHÔNG mock detect_faces để logic ứng dụng thực tế chạy
            # Khối 'with patch' cho detect_faces đã đóng, vì vậy nó không bị mock cho phần này.
            text_path = os.path.join(self.test_dir, "test.txt")
            with open(text_path, "w") as f:
                f.write("This is not an image")
            
            with open(text_path, "rb") as f:
                response = client.post(
                    "/recognition",
                    headers=headers,
                    files={"image": ("test.txt", f, "text/plain")}
                )
                print(f"Phản hồi JSON cho text/plain: {response.json()}") # Thêm print để gỡ lỗi
                self.assertEqual(response.status_code, 400)