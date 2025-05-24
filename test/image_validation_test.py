import unittest
import os
import tempfile
import cv2
import numpy as np
from datetime import datetime  # Thêm dòng này
from unittest.mock import patch, MagicMock
import shutil
from src.application import validate_image_quality
from fastapi.testclient import TestClient
from src.application import app, FaceRecognitionService

class TestImageValidation(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for test images
        self.test_dir = tempfile.mkdtemp()

        # Tạo ảnh để test độ mờ (sharp và blurry)
        self.sharp_img_path = os.path.join(self.test_dir, "sharp.jpg")
        # Bắt đầu với một ảnh xám trung bình hoặc sáng hơn, sau đó thêm chi tiết
        sharp_img = np.ones((100, 100, 3), dtype=np.uint8) * 150 # Nền sáng hơn
        cv2.putText(sharp_img, "Sharp", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2) # Text đen trên nền sáng
        cv2.imwrite(self.sharp_img_path, sharp_img)

        self.blurry_img_path = os.path.join(self.test_dir, "blurry.jpg")
        # Tạo ảnh với ít chi tiết, dễ bị mờ
        blurry_img_data = np.zeros((100, 100, 3), dtype=np.uint8)
        blurry_img = cv2.GaussianBlur(blurry_img_data, (99, 99), 0) # Sử dụng kernel lớn để đảm bảo mờ
        cv2.imwrite(self.blurry_img_path, blurry_img)

        # Tạo ảnh để test độ sáng (normal, dark, bright)
        self.normal_img_path = os.path.join(self.test_dir, "normal.jpg")
        normal_img = np.ones((100, 100, 3), dtype=np.uint8) * 127 # Ảnh xám trung bình
        # Thêm một chút nhiễu hoặc một hình tròn/vuông nhỏ để tăng độ tương phản
        cv2.circle(normal_img, (50, 50), 20, (0, 0, 255), -1) # Thêm một hình tròn màu xanh
        cv2.imwrite(self.normal_img_path, normal_img)

        self.dark_img_path = os.path.join(self.test_dir, "dark.jpg")
        dark_img = np.ones((100, 100, 3), dtype=np.uint8) * 20
        cv2.imwrite(self.dark_img_path, dark_img)

        self.bright_img_path = os.path.join(self.test_dir, "bright.jpg")
        bright_img = np.ones((100, 100, 3), dtype=np.uint8) * 240
        cv2.imwrite(self.bright_img_path, bright_img)

        # Tạo ảnh không đọc được
        self.invalid_img_path = os.path.join(self.test_dir, "invalid.txt")
        with open(self.invalid_img_path, "w") as f:
            f.write("This is not an image file.")

    def tearDown(self):
        # Clean up
        shutil.rmtree(self.test_dir)

    def test_validate_image_quality_sharp_image(self):
        # Test với ảnh sắc nét
        result = validate_image_quality(self.sharp_img_path)
        self.assertTrue(result["valid"], f"Expected sharp image to be valid, but got: {result['errors']}")
        self.assertEqual(len(result["errors"]), 0)

    def test_validate_image_quality_blurry_image(self):
        # Test với ảnh mờ
        result = validate_image_quality(self.blurry_img_path)
        self.assertFalse(result["valid"])
        # Cập nhật chuỗi lỗi khớp với hàm validate_image_quality
        self.assertIn("Ảnh bị mờ (độ tương phản thấp)", result["errors"])

    def test_validate_image_quality_normal_brightness_image(self):
        # Test với ảnh độ sáng bình thường
        result = validate_image_quality(self.normal_img_path)
        self.assertTrue(result["valid"], f"Expected normal brightness image to be valid, but got: {result['errors']}")
        self.assertEqual(len(result["errors"]), 0)

    def test_validate_image_quality_dark_image(self):
        # Test với ảnh thiếu sáng
        result = validate_image_quality(self.dark_img_path)
        self.assertFalse(result["valid"])
        self.assertIn("Ảnh thiếu sáng", result["errors"])

    def test_validate_image_quality_bright_image(self):
        # Test với ảnh dư sáng
        result = validate_image_quality(self.bright_img_path)
        self.assertFalse(result["valid"])
        self.assertIn("Ảnh dư sáng", result["errors"])

    def test_validate_image_quality_invalid_file(self):
        # Test với file không phải ảnh (không đọc được)
        result = validate_image_quality(self.invalid_img_path)
        self.assertFalse(result["valid"])
        self.assertIn("Không đọc được file ảnh", result["errors"])

    @patch('src.application.validate_image_quality')
    @patch('src.application.FaceRecognitionService.align_faces')
    @patch('src.application.FaceRecognitionService.train_classifier')
    @patch('src.application.SessionLocal')
    def test_validation_in_registration_flow(self, mock_db_session, mock_train_classifier, mock_align_faces, mock_validate_image_quality):
        client = TestClient(app)

        # Setup mock cho SessionLocal
        mock_db_session.return_value.query.return_value.filter.return_value.first.return_value = None
        mock_db_session.return_value.add.return_value = None
        mock_db_session.return_value.commit.return_value = None

        mock_face_data_instance = MagicMock()
        mock_face_data_instance.person_name = "test_person"
        mock_face_data_instance.num_images = 3
        mock_face_data_instance.registration_date = datetime.now()
        mock_face_data_instance.embedding = "some_mocked_embedding"
        mock_face_data_instance.id = 1

        mock_db_session.return_value.refresh.side_effect = lambda obj: obj.__dict__.update(mock_face_data_instance.__dict__)

        test_files_data = [
            ("images", ("img1.jpg", b"dummy content", "image/jpeg")),
            ("images", ("img2.jpg", b"dummy content", "image/jpeg")),
            ("images", ("img3.jpg", b"dummy content", "image/jpeg"))
        ]

        with patch('os.getenv', return_value="test_api_key"):
            # Case 1: Tất cả ảnh đều hợp lệ
            # Sửa mock align_faces và train_classifier để trả về giá trị phù hợp
            mock_align_faces.return_value = [MagicMock()]  # Trả về danh sách các face đã căn chỉnh
            mock_train_classifier.return_value = MagicMock()  # Trả về đối tượng classifier giả lập
            mock_validate_image_quality.return_value = {"valid": True, "errors": []}

            # Case 1: Tất cả ảnh đều hợp lệ
            response = client.post(
                "/register",
                data={"name": "test_person"},
                files=test_files_data,
                headers={"X-API-Key": "test_api_key"}
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "success")  # Kiểm tra key "status"

            # Case 2: Ảnh bị mờ (do validate_image_quality mock trả về lỗi mờ)
            mock_validate_image_quality.return_value = {"valid": False, "errors": ["Ảnh bị mờ (độ tương phản thấp)"]} # Cập nhật chuỗi lỗi
            response = client.post(
                "/register",
                data={"name": "test_person"},
                files=test_files_data,
                headers={"X-API-Key": "test_api_key"}
            )
            self.assertEqual(response.status_code, 400)
            self.assertIn("mờ", response.json()["details"][0]["errors"][0])

            # Case 3: Ảnh thiếu sáng (do validate_image_quality mock trả về lỗi thiếu sáng)
            mock_validate_image_quality.return_value = {"valid": False, "errors": ["Ảnh thiếu sáng"]}
            response = client.post(
                "/register",
                data={"name": "test_person"},
                files=test_files_data,
                headers={"X-API-Key": "test_api_key"}
            )
            self.assertEqual(response.status_code, 400)
            self.assertIn("thiếu sáng", response.json()["details"][0]["errors"][0])

            # Case 4: Ảnh dư sáng (do validate_image_quality mock trả về lỗi dư sáng)
            mock_validate_image_quality.return_value = {"valid": False, "errors": ["Ảnh dư sáng"]}
            response = client.post(
                "/register",
                data={"name": "test_person"},
                files=test_files_data,
                headers={"X-API-Key": "test_api_key"}
            )
            self.assertEqual(response.status_code, 400)
            self.assertIn("dư sáng", response.json()["details"][0]["errors"][0])

            # Case 5: Không đọc được file ảnh (do validate_image_quality mock trả về lỗi không đọc được)
            mock_validate_image_quality.return_value = {"valid": False, "errors": ["Không đọc được file ảnh"]}
            response = client.post(
                "/register",
                data={"name": "test_person"},
                files=test_files_data,
                headers={"X-API-Key": "test_api_key"}
            )
            self.assertEqual(response.status_code, 400)
            self.assertIn("Không đọc được", response.json()["details"][0]["errors"][0])