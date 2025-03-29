import os
import shutil
import sys
import uuid
import subprocess
from datetime import datetime
from typing import List, Optional
from sklearn.svm import SVC
import pickle
import tensorflow as tf
# Disable eager execution for TensorFlow 2.x compatibility
tf.compat.v1.disable_eager_execution()

import numpy as np
import cv2
import align.detect_face
from face_recognition_process import facenet
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, String, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import logging
from typing import List, Optional
from fastapi import Path, Query, HTTPException, Depends
from sqlalchemy.orm import Session

# Configure logging
logging.basicConfig(
    filename="face_service.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("face_recognition_service")

# Database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./face_recognition.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define the FaceData model for database
class FaceData(Base):
    __tablename__ = "faces"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, index=True)
    registered_at = Column(DateTime, default=datetime.now)
    embedding = Column(String)  # Store the face embedding as a serialized string


# Create the database tables
Base.metadata.create_all(bind=engine)

# Set up paths and constants
# Get the actual project base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logger.info(f"Base directory: {BASE_DIR}")

# Define paths based on project structure
RAW_DATASET_DIR = os.path.join(BASE_DIR, "Dataset", "FaceData", "raw")
PROCESSED_DATASET_DIR = os.path.join(BASE_DIR, "Dataset", "FaceData", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "Models")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "facemodel.pkl")
FACENET_MODEL_PATH = os.path.join(MODEL_DIR, "20180402-114759.pb")

# Log all path information for debugging
logger.info(f"Raw dataset directory: {RAW_DATASET_DIR}")
logger.info(f"Processed dataset directory: {PROCESSED_DATASET_DIR}")
logger.info(f"Model directory: {MODEL_DIR}")
logger.info(f"Classifier path: {CLASSIFIER_PATH}")
logger.info(f"FaceNet model path: {FACENET_MODEL_PATH}")

# Ensure directories exist
os.makedirs(RAW_DATASET_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# Pydantic models for API requests/responses
class RegisterRequest(BaseModel):
    name: str

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class RecognitionResponse(BaseModel):
    id: str
    name: str
    confidence: float
    registered_at: datetime

# Add these Pydantic models near the existing ones
class FaceListResponse(BaseModel):
    id: str
    name: str
    registered_at: datetime


class FaceRecognitionService:
    def __init__(self):
        self.graph = tf.Graph()
        self.sess = None
        self.pnet = None
        self.rnet = None
        self.onet = None
        self.embeddings = None
        self.images_placeholder = None
        self.phase_train_placeholder = None

        self.init_face_recognition()

    def init_face_recognition(self):
        """Initialize the face recognition model"""
        logger.info("Initializing face recognition model")
        with self.graph.as_default():
            self.sess = tf.compat.v1.Session()

            # Load MTCNN
            # Fix path handling for cross-platform compatibility
            align_path = os.path.join(BASE_DIR, "src", "align")
            if not os.path.exists(os.path.join(align_path, "det1.npy")):
                logger.error(f"MTCNN model files not found in {align_path}")
                raise FileNotFoundError(f"MTCNN model files not found in {align_path}")
            self.pnet, self.rnet, self.onet = align.detect_face.create_mtcnn(self.sess, align_path)

            # Load FaceNet model
            facenet.load_model(FACENET_MODEL_PATH)

            # Get input and output tensors
            self.images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            self.embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            self.phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")

    def align_faces(self, person_name):
        """
        Align faces using the original GitHub code approach.
        This produces the same results as the original project.
        """
        logger.info(f"Aligning faces for {person_name}")

        input_dir = os.path.join(RAW_DATASET_DIR, person_name)
        output_dir = PROCESSED_DATASET_DIR

        if not os.path.exists(input_dir):
            logger.error(f"Input directory does not exist: {input_dir}")
            return False

        os.makedirs(output_dir, exist_ok=True)

        # Import necessary modules that the original code uses
        import random
        from PIL import Image
        import math

        # MTCNN parameters - same as original code
        minsize = 20
        threshold = [0.6, 0.7, 0.7]
        factor = 0.709
        image_size = 160
        margin = 44

        with self.graph.as_default():
            with self.sess.as_default():
                logger.info("Running MTCNN face detection")

                # Get the dataset for this person
                dataset = []
                dataset.append(facenet.ImageClass(person_name, [p for p in os.listdir(input_dir)
                                                                if os.path.isfile(os.path.join(input_dir, p))]))

                # Create output directory
                person_output_dir = os.path.join(output_dir, person_name)
                os.makedirs(person_output_dir, exist_ok=True)

                # Process each class (just one in this case)
                nrof_images_total = 0
                nrof_successfully_aligned = 0

                for cls in dataset:
                    output_class_dir = os.path.join(output_dir, cls.name)
                    if not os.path.exists(output_class_dir):
                        os.makedirs(output_class_dir)

                    # Process each image
                    for image_path in cls.image_paths:
                        full_image_path = os.path.join(input_dir, image_path)
                        nrof_images_total += 1

                        filename = os.path.splitext(os.path.basename(image_path))[0]
                        output_filename = os.path.join(output_class_dir, filename + '.png')

                        logger.info(f"Processing {full_image_path}")

                        if not os.path.exists(output_filename):
                            try:
                                # Read the image
                                img = cv2.imread(full_image_path)
                                if img is None:
                                    logger.error(f"Could not read {full_image_path}")
                                    continue

                                # Convert BGR to RGB for MTCNN
                                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                                # Detect faces
                                logger.info(f"Detecting faces in {image_path}")
                                bounding_boxes, _ = align.detect_face.detect_face(
                                    img_rgb, minsize, self.pnet, self.rnet, self.onet, threshold, factor
                                )

                                logger.info(f"Detected {bounding_boxes.shape[0]} faces in {image_path}")

                                # If faces detected, process them
                                if bounding_boxes.shape[0] > 0:
                                    # Use bounding box for largest face
                                    det_areas = (bounding_boxes[:, 2] - bounding_boxes[:, 0]) * (
                                                bounding_boxes[:, 3] - bounding_boxes[:, 1])
                                    sorted_idx = np.argsort(det_areas)[::-1]  # Sort by area, largest first
                                    det = bounding_boxes[sorted_idx[0], 0:4]

                                    # Calculate the bounding box
                                    bb = np.zeros(4, dtype=np.int32)
                                    bb[0] = np.maximum(det[0] - margin / 2, 0)
                                    bb[1] = np.maximum(det[1] - margin / 2, 0)
                                    bb[2] = np.minimum(det[2] + margin / 2, img.shape[1])
                                    bb[3] = np.minimum(det[3] + margin / 2, img.shape[0])

                                    # Crop and align the face
                                    cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]

                                    # Convert to PIL and resize
                                    cropped_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                                    scaled = cropped_pil.resize((image_size, image_size), Image.BICUBIC)

                                    # Save the image
                                    scaled.save(output_filename, format="PNG")
                                    logger.info(f"Saved aligned face to {output_filename}")
                                    nrof_successfully_aligned += 1
                                else:
                                    # If no face detected, try using Haar cascade as fallback
                                    try:
                                        # Load Haar cascade
                                        cascade_path = os.path.join(cv2.__path__[0], 'data',
                                                                    'haarcascade_frontalface_default.xml')
                                        if not os.path.exists(cascade_path):
                                            cascade_path = os.path.join(BASE_DIR, "src",
                                                                        "haarcascade_frontalface_default.xml")

                                        if os.path.exists(cascade_path):
                                            face_cascade = cv2.CascadeClassifier(cascade_path)
                                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                                            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

                                            if len(faces) > 0:
                                                # Use the largest detected face
                                                if len(faces) > 1:
                                                    face_areas = [(w * h, (x, y, w, h)) for (x, y, w, h) in faces]
                                                    face_areas.sort(reverse=True)
                                                    _, largest_face = face_areas[0]
                                                    x, y, w, h = largest_face
                                                else:
                                                    x, y, w, h = faces[0]

                                                # Add margin (40%)
                                                margin_percent = 0.4
                                                center_x = x + w / 2
                                                center_y = y + h / 2

                                                # Calculate new size with margin
                                                new_size = max(w, h) * (1 + margin_percent)

                                                # Calculate bounding box
                                                bb = np.zeros(4, dtype=np.int32)
                                                bb[0] = max(0, int(center_x - new_size / 2))
                                                bb[1] = max(0, int(center_y - new_size / 2))
                                                bb[2] = min(img.shape[1], int(center_x + new_size / 2))
                                                bb[3] = min(img.shape[0], int(center_y + new_size / 2))

                                                # Crop and align the face
                                                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]

                                                # Convert to PIL and resize
                                                cropped_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                                                scaled = cropped_pil.resize((image_size, image_size), Image.BICUBIC)

                                                # Save the image
                                                scaled.save(output_filename, format="PNG")
                                                logger.info(
                                                    f"Saved face detected with Haar cascade to {output_filename}")
                                                nrof_successfully_aligned += 1
                                                continue
                                    except Exception as e:
                                        logger.error(f"Haar cascade failed: {str(e)}")

                                    # If all detection fails, use center crop
                                    logger.warning(f"No face detected in {image_path}, using center crop")
                                    height, width = img.shape[:2]
                                    center_x = width // 2
                                    center_y = height // 2

                                    # If portrait, shift up
                                    if height > width:
                                        center_y = int(height * 0.4)

                                    crop_size = min(width, height)
                                    bb = np.zeros(4, dtype=np.int32)
                                    bb[0] = max(0, center_x - crop_size // 2)
                                    bb[1] = max(0, center_y - crop_size // 2)
                                    bb[2] = min(width, center_x + crop_size // 2)
                                    bb[3] = min(height, center_y + crop_size // 2)

                                    # Crop and resize
                                    cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                                    cropped_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                                    scaled = cropped_pil.resize((image_size, image_size), Image.BICUBIC)

                                    # Save the image
                                    scaled.save(output_filename, format="PNG")
                                    logger.info(f"Saved center-cropped image to {output_filename}")
                                    nrof_successfully_aligned += 1
                            except Exception as e:
                                logger.error(f"Error processing {image_path}: {str(e)}")
                                import traceback
                                logger.error(traceback.format_exc())

                logger.info(f"Total images: {nrof_images_total}, Successfully aligned: {nrof_successfully_aligned}")
                return nrof_successfully_aligned > 0

    def train_classifier(self):
        """Train the SVM classifier on the processed dataset using the same parameters as the GitHub project"""
        logger.info("Training SVM classifier")

        # Get the correct path to the classifier.py script
        classifier_script_path = os.path.join(BASE_DIR, "src/face_recognition_process", "classifier.py")
        logger.info(f"Classifier script path: {classifier_script_path}")

        if not os.path.exists(classifier_script_path):
            logger.error(f"Classifier script not found at: {classifier_script_path}")
            return False

        # Check if the processed dataset is not empty
        if not os.path.exists(PROCESSED_DATASET_DIR):
            logger.error(f"Processed dataset directory not found: {PROCESSED_DATASET_DIR}")
            return False

        # Count classes and images
        class_count = 0
        image_count = 0
        for class_name in os.listdir(PROCESSED_DATASET_DIR):
            class_dir = os.path.join(PROCESSED_DATASET_DIR, class_name)
            if os.path.isdir(class_dir):
                class_count += 1
                for f in os.listdir(class_dir):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_count += 1

        logger.info(f"Found {class_count} classes with {image_count} images in processed dataset")

        if class_count == 0 or image_count == 0:
            logger.error("No classes or images found in processed dataset")
            return False

        # Try to use the original classifier.py script via subprocess
        try:
            # Use the original classifier.py script via subprocess
            cmd = [
                sys.executable,  # Use the current Python interpreter
                classifier_script_path,
                "TRAIN",
                PROCESSED_DATASET_DIR,
                FACENET_MODEL_PATH,
                CLASSIFIER_PATH,
                "--batch_size", "1000"
            ]

            logger.info(f"Running classifier training command: {' '.join(cmd)}")

            # Run the subprocess and capture output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate()

            # Log the output
            if stdout:
                logger.info(f"Classifier training stdout: {stdout}")
            if stderr:
                if "error" in stderr.lower() or "exception" in stderr.lower():
                    logger.error(f"Classifier training stderr: {stderr}")
                else:
                    logger.info(f"Classifier training stderr: {stderr}")

            if process.returncode != 0:
                logger.error(f"Classifier training process exited with code {process.returncode}")
                return False

            # Verify the classifier was created
            if not os.path.exists(CLASSIFIER_PATH):
                logger.error(f"Classifier file not created: {CLASSIFIER_PATH}")
                return False

            # Try to load the classifier to verify it's valid
            try:
                with open(CLASSIFIER_PATH, 'rb') as file:
                    model, class_names = pickle.load(file)
                logger.info(f"Successfully trained classifier with {len(class_names)} classes: {class_names}")
                return True
            except Exception as e:
                logger.error(f"Failed to load the trained classifier: {str(e)}")
                return False

        except Exception as e:
            logger.error(f"Classifier training failed with exception: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

            # If the subprocess approach fails, try direct training as fallback
            logger.warning("Subprocess training failed, attempting direct training...")

            try:
                # Get dataset
                dataset = facenet.get_dataset(PROCESSED_DATASET_DIR)

                # Check that there's at least one image per class
                for cls in dataset:
                    if len(cls.image_paths) == 0:
                        logger.error(f"No images for class {cls.name}")
                        return False

                logger.info(f"Training on dataset with {len(dataset)} classes")

                # Extract features using FaceNet
                with self.graph.as_default():
                    with self.sess.as_default():
                        # Get input and output tensors
                        images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
                        embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
                        phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")

                        # Get image paths and labels
                        paths, labels = facenet.get_image_paths_and_labels(dataset)

                        logger.info(f"Found {len(paths)} images and {len(set(labels))} unique labels")

                        # Calculate embeddings
                        batch_size = 100
                        nrof_images = len(paths)
                        nrof_batches_per_epoch = int(np.ceil(1.0 * nrof_images / batch_size))
                        emb_array = np.zeros((nrof_images, embeddings.get_shape()[1]))

                        for i in range(nrof_batches_per_epoch):
                            start_index = i * batch_size
                            end_index = min((i + 1) * batch_size, nrof_images)
                            paths_batch = paths[start_index:end_index]
                            images = facenet.load_data(paths_batch, False, False, 160)
                            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                            emb_array[start_index:end_index, :] = self.sess.run(embeddings, feed_dict=feed_dict)

                        # Train classifier
                        logger.info("Training SVM classifier")
                        model = SVC(kernel='linear', probability=True)
                        model.fit(emb_array, labels)

                        # Create a list of class names
                        class_names = [cls.name.replace('_', ' ') for cls in dataset]

                        # Save classifier model
                        with open(CLASSIFIER_PATH, 'wb') as outfile:
                            pickle.dump((model, class_names), outfile)

                        logger.info(f"Saved classifier to {CLASSIFIER_PATH}")
                        return True

            except Exception as e2:
                logger.error(f"Direct training also failed: {str(e2)}")
                import traceback
                logger.error(traceback.format_exc())
                return False
    def detect_faces(self, image_path):
        """Detect faces in an image and return the face data"""
        logger.info(f"Detecting faces in {image_path}")

        MINSIZE = 20
        THRESHOLD = [0.6, 0.7, 0.7]  # Same thresholds as original GitHub project
        FACTOR = 0.709
        INPUT_IMAGE_SIZE = 160

        with self.graph.as_default():
            with self.sess.as_default():
                # Check if classifier exists
                if not os.path.exists(CLASSIFIER_PATH):
                    logger.error(f"Classifier not found at path: {CLASSIFIER_PATH}")
                    return {"error": "Classifier model not found", "path": CLASSIFIER_PATH}

                # Load classifier model
                try:
                    with open(CLASSIFIER_PATH, 'rb') as file:
                        model, class_names = pickle.load(file)
                    logger.info(f"Classifier loaded successfully with {len(class_names)} classes: {class_names}")
                except Exception as e:
                    logger.error(f"Error loading classifier: {str(e)}")
                    return {"error": f"Failed to load classifier: {str(e)}"}

                # Load and preprocess image
                if not os.path.exists(image_path):
                    logger.error(f"Image not found at path: {image_path}")
                    return {"error": "Image file not found"}

                frame = cv2.imread(image_path)
                if frame is None:
                    logger.error(f"Failed to read image from {image_path}")
                    return {"error": "Failed to read image"}

                logger.info(f"Image loaded, shape: {frame.shape}")

                # Don't resize - process at original resolution for better accuracy
                # This matches the approach in the original GitHub project

                # Detect faces using MTCNN
                try:
                    logger.info("Running MTCNN face detection")
                    bounding_boxes, _ = align.detect_face.detect_face(
                        frame, MINSIZE, self.pnet, self.rnet, self.onet, THRESHOLD, FACTOR
                    )
                    logger.info(f"MTCNN detected {bounding_boxes.shape[0]} faces")
                except Exception as e:
                    logger.error(f"Face detection error: {str(e)}")
                    return {"error": f"Face detection failed: {str(e)}"}

                faces_data = []

                if bounding_boxes.shape[0] > 0:
                    for i in range(bounding_boxes.shape[0]):
                        try:
                            det = bounding_boxes[i, 0:4]
                            bb = np.zeros(4, dtype=np.int32)
                            bb[0] = max(det[0], 0)
                            bb[1] = max(det[1], 0)
                            bb[2] = min(det[2], frame.shape[1])
                            bb[3] = min(det[3], frame.shape[0])

                            if bb[2] <= bb[0] or bb[3] <= bb[1]:
                                logger.warning(f"Invalid bounding box: {bb}")
                                continue

                            logger.info(f"Processing face {i + 1}, bounding box: {bb}")

                            # Extract and process face using same method as original project
                            margin = 44  # Same as in align_faces
                            bb_margin = np.zeros(4, dtype=np.int32)
                            bb_margin[0] = np.maximum(det[0] - margin / 2, 0)
                            bb_margin[1] = np.maximum(det[1] - margin / 2, 0)
                            bb_margin[2] = np.minimum(det[2] + margin / 2, frame.shape[1])
                            bb_margin[3] = np.minimum(det[3] + margin / 2, frame.shape[0])

                            cropped = frame[bb_margin[1]:bb_margin[3], bb_margin[0]:bb_margin[2], :]

                            # Use exact same resizing as in alignment
                            from PIL import Image
                            cropped_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                            scaled = cropped_pil.resize((INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), Image.BICUBIC)
                            scaled_np = np.array(scaled)

                            # Convert back to BGR for prewhiten
                            scaled_bgr = cv2.cvtColor(scaled_np, cv2.COLOR_RGB2BGR)
                            scaled_prewhite = facenet.prewhiten(scaled_bgr)

                            scaled_reshape = scaled_prewhite.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)

                            # Get face embedding
                            feed_dict = {
                                self.images_placeholder: scaled_reshape,
                                self.phase_train_placeholder: False
                            }
                            emb_array = self.sess.run(self.embeddings, feed_dict=feed_dict)
                            logger.info(f"Generated embedding vector of shape {emb_array.shape}")

                            # Predict identity
                            predictions = model.predict_proba(emb_array)
                            best_class_indices = np.argmax(predictions, axis=1)
                            best_class_probabilities = predictions[
                                np.arange(len(best_class_indices)), best_class_indices]

                            best_name = class_names[best_class_indices[0]]
                            best_prob = float(best_class_probabilities[0])

                            logger.info(f"Face {i + 1} recognized as '{best_name}' with confidence {best_prob:.4f}")

                            face_data = {
                                "name": best_name,
                                "confidence": best_prob,
                                "bbox": bb.tolist()
                            }
                            faces_data.append(face_data)
                        except Exception as e:
                            logger.error(f"Error processing face {i + 1}: {str(e)}")
                            import traceback
                            logger.error(traceback.format_exc())
                else:
                    logger.warning("No faces detected in the image")

                return faces_data

# Create FastAPI app
app = FastAPI(title="Face Recognition Service", version="1.0")

# Add middleware for CORS
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize face recognition service
face_service = FaceRecognitionService()


@app.post("/register")
async def register_face(
        name: str = Form(...),
        images: List[UploadFile] = File(...),  # This expects multiple files with the same field name 'images'
):
    """
    Register a new person with multiple face images.

    - name: Name of the person to register
    - images: Multiple images of the person's face
    """
    try:
        logger.info(f"Registering new face: {name}")

        # Generate a unique ID
        face_id = str(uuid.uuid4())

        # Create directory for this person if it doesn't exist
        person_dir = os.path.join(RAW_DATASET_DIR, name)
        os.makedirs(person_dir, exist_ok=True)

        # Save all uploaded images
        saved_paths = []
        for i, image in enumerate(images):
            image_path = os.path.join(person_dir, f"{face_id}_{i}.jpg")
            with open(image_path, "wb") as f:
                shutil.copyfileobj(image.file, f)
            saved_paths.append(image_path)

        logger.info(f"Saved {len(saved_paths)} images for {name}")

        # Use the single alignment method
        if not face_service.align_faces(name):
            raise HTTPException(status_code=500, detail="Face alignment failed")

        # Train classifier
        if not face_service.train_classifier():
            raise HTTPException(status_code=500, detail="Classifier training failed")

        # Save to database
        db = SessionLocal()
        try:
            db_face = FaceData(
                id=face_id,
                name=name,
                registered_at=datetime.now()
            )
            db.add(db_face)
            db.commit()
            logger.info(f"Successfully registered {name} with ID {face_id}")
        finally:
            db.close()

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": f"Registered {name} with {len(images)} images",
                "id": face_id
            }
        )

    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


@app.post("/recognition")
async def recognize_face(image: UploadFile = File(...)):
    """
    Recognize faces in an uploaded image.

    Returns:
    - id: The unique ID of the recognized person
    - name: The name of the recognized person
    - confidence: The confidence score (0-1)
    - registered_at: When the person was registered
    """
    try:
        logger.info("Processing recognition request")
        logger.info(f"Uploaded image: {image.filename}, size: {image.size} bytes")

        # Save uploaded image temporarily
        temp_dir = os.path.join(BASE_DIR, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f"temp_recognition_{uuid.uuid4()}.jpg")

        logger.info(f"Saving uploaded image to {temp_path}")
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        # Detect faces
        faces_data = face_service.detect_faces(temp_path)

        # Check if faces_data is an error dictionary
        if isinstance(faces_data, dict) and "error" in faces_data:
            logger.error(f"Face detection returned error: {faces_data}")
            return JSONResponse(
                status_code=400,
                content={"error": faces_data["error"], "details": faces_data}
            )

        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

        # If no faces found, return detailed message instead of empty array
        if not faces_data:
            logger.warning("No faces found or recognized in the image")
            return JSONResponse(
                status_code=200,
                content={
                    "message": "No faces found or recognized in the image",
                    "suggestions": [
                        "Make sure the image contains clearly visible faces",
                        "Check that the person is registered in the system",
                        "Ensure good lighting conditions in the image",
                        "Try with a different image or angle"
                    ]
                }
            )

        # Get registration data from database
        db = SessionLocal()
        try:
            results = []
            for face in faces_data:
                # Find the person in the database
                db_face = db.query(FaceData).filter(FaceData.name == face["name"]).first()

                logger.info(f"Found match: {face['name']} with confidence {face['confidence']}")
                logger.info(f"Database record found: {db_face is not None}")

                # Lowered threshold to 0.4 for testing
                if db_face and face["confidence"] > 0.4:
                    results.append({
                        "id": db_face.id,
                        "name": db_face.name,
                        "confidence": face["confidence"],
                        "registered_at": db_face.registered_at.isoformat()
                    })

            if not results:
                return JSONResponse(
                    status_code=200,
                    content={
                        "message": "Faces were detected but did not match any registered person with sufficient confidence",
                        "detected_faces": len(faces_data),
                        "best_match": {
                            "name": faces_data[0]["name"],
                            "confidence": faces_data[0]["confidence"]
                        } if faces_data else None
                    }
                )

            return results
        finally:
            db.close()

    except Exception as e:
        logger.error(f"Recognition error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Recognition failed: {str(e)}")



@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

@app.get("/faces", response_model=List[FaceListResponse])
async def list_faces(db: Session = Depends(get_db)):
    """
    List all registered faces in the database.
    
    Returns:
    - List of all registered faces with their ID, name, and registration timestamp
    """
    try:
        logger.info("Listing all registered faces")
        
        # Query all faces from the database
        faces = db.query(FaceData).all()
        
        # Convert to response model
        response = [
            FaceListResponse(
                id=face.id,
                name=face.name,
                registered_at=face.registered_at
            ) for face in faces
        ]
        
        logger.info(f"Returning {len(response)} registered faces")
        return response
        
    except Exception as e:
        logger.error(f"Error listing faces: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to list faces: {str(e)}")

@app.delete("/faces/{face_id}")
async def delete_face(face_id: str = Path(..., description="The ID of the face to delete"), 
                      db: Session = Depends(get_db)):
    """
    Delete a registered face by ID.
    
    Parameters:
    - face_id: The unique ID of the face to delete
    
    Returns:
    - Success message if deletion was successful
    """
    try:
        logger.info(f"Deleting face with ID: {face_id}")
        
        # Get the face from the database
        face = db.query(FaceData).filter(FaceData.id == face_id).first()
        
        if not face:
            logger.warning(f"Face with ID {face_id} not found in database")
            raise HTTPException(status_code=404, detail=f"Face with ID {face_id} not found")
        
        # Remember the person's name before deletion
        person_name = face.name
        logger.info(f"Found face: {person_name} with ID {face_id}")
        
        # Delete from database
        db.delete(face)
        db.commit()
        logger.info(f"Deleted face record from database")
        
        # Check if there are any more faces for this person
        remaining_faces = db.query(FaceData).filter(FaceData.name == person_name).count()
        logger.info(f"Remaining faces for {person_name}: {remaining_faces}")
        
        # If no more faces for this person, delete their images from filesystem
        if remaining_faces == 0:
            # Delete from raw dataset
            raw_person_dir = os.path.join(RAW_DATASET_DIR, person_name)
            if os.path.exists(raw_person_dir):
                shutil.rmtree(raw_person_dir)
                logger.info(f"Deleted raw dataset directory for {person_name}")
            
            # Delete from processed dataset
            processed_person_dir = os.path.join(PROCESSED_DATASET_DIR, person_name)
            if os.path.exists(processed_person_dir):
                shutil.rmtree(processed_person_dir)
                logger.info(f"Deleted processed dataset directory for {person_name}")
            
            # Retrain the classifier if the person was deleted
            if os.path.exists(CLASSIFIER_PATH):
                try:
                    # Only retrain if there are still other people in the dataset
                    if len(os.listdir(PROCESSED_DATASET_DIR)) > 0:
                        logger.info("Retraining classifier after deleting person")
                        face_service.train_classifier()
                    else:
                        # If no people left, delete the classifier
                        os.remove(CLASSIFIER_PATH)
                        logger.info("Deleted classifier as no people remain in dataset")
                except Exception as e:
                    logger.error(f"Error retraining classifier: {str(e)}")
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": f"Successfully deleted face with ID {face_id}"
            }
        )
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
        
    except Exception as e:
        logger.error(f"Error deleting face: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to delete face: {str(e)}")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000 )