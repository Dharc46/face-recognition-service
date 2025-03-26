# Face Recognition with MTCNN and Pretrained Facenet.

This project allows accurate face recognition using MTCNN and Facenet, built on TensorFlow 2.x.

## Credit to MìAI

[MìAI Face Recognition 2.0](http://miai.vn/2019/09/11/face-recog-2-0-nhan-dien-khuon-mat-trong-video-bang-mtcnn-va-facenet/)

## Installation & Run

### 1. Clone the repository and install dependencies

```bash
git clone https://github.com/Dharc46/facenet-ft-mtcnn-for-exam-web.git
```

```bash
pip install -r requirements.txt
```

### 2. Preprocess Data to Extract Face from Original Images

Firstly, you need to create a folder in Dataset/FaceData/raw, then put your images (that you want model to recognize) into it.

Run the following command to preprocess and extract faces from your dataset. It will create a processed folder that contains only the cropped faces.

```bash
python src/align_dataset_mtcnn.py Dataset/FaceData/raw Dataset/FaceData/processed --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.25
```

Once the process completes, you should see the message "Total number of images: ..." in the terminal, indicating the preprocessing is successful.

Create Models folder and download Facenet pretrained models (https://bit.ly/3ixQH7o) then put them in the folder.

### 3. Train the last SVM layer to classify data for Face Recognition Model

Train the model to recognize faces using the following command. It will create a model file called facemodel.pkl.

```bash
python src/classifier.py TRAIN Dataset/FaceData/processed Models/20180402-114759.pb Models/facemodel.pkl --batch_size 1000
```

### 4. Results

- To recognize face using camera:

```bash
python src/face_rec_cam.py
```

- To recognize a face from an image:

```bash
python src/face_rec_image.py --path 'Faces/your_image.jpg'
```

_Note: Replace `your_image.jpg` with the name of the image you want to recognize._


#Face Recognition Microservice
This is a FastAPI-based microservice for face recognition, built on top of the existing MTCNN and FaceNet code. It provides endpoints for registering new faces and recognizing faces in images.

##Architecture
The service follows the original project's workflow:

Face Detection & Alignment - Using MTCNN
Feature Extraction - Using FaceNet
Classification - Using SVM
##API Endpoints
1. Register a New Face
Endpoint: POST /register

Description: Register a new person with multiple face images.

Request Format:

name (form field): Name of the person to register
images (form files): Multiple image files of the person's face
Process:

Saves images to Dataset/FaceData/raw/{name}/
Aligns and crops faces using MTCNN
Trains a new SVM classifier
Stores metadata in the database
Response:

json

Copy
{
  "status": "success",
  "message": "Registered {name} with {n} images",
  "id": "unique-uuid"
}
2. Recognize Faces
Endpoint: POST /recognition

Description: Detect and recognize faces in an uploaded image.

Request Format:

image (form file): An image file containing one or more faces
Process:

Detects faces using MTCNN
Generates embeddings using FaceNet
Classifies faces using the trained SVM
Retrieves registration data from the database
Response:

json

Copy
[
  {
    "id": "unique-uuid",
    "name": "person name",
    "confidence": 0.95,
    "registered_at": "2023-11-01T12:00:00"
  },
  // Additional faces if multiple are detected
]
3. Health Check
Endpoint: GET /health

Description: Simple health check endpoint to verify the service is running.

Response:

json

Copy
{
  "status": "ok"
}
##Database Schema
The service uses SQLite to store face registration data:

Table: faces

id (UUID, primary key): Unique identifier
name (string): Person's name
registered_at (datetime): Registration timestamp
embedding (string): Serialized face embedding data
##Docker Deployment
The service is containerized with Docker and can be run using:

bash

Copy
docker-compose up -d
The Docker setup includes:

TensorFlow 1.15.5 with GPU support
All necessary Python dependencies
Volume mounts for dataset and model persistence
##Notes for Integration
Model Files: Make sure you have the FaceNet model (20180402-114759.pb) in the Models directory.
GPU Support: The service is configured to use GPU if available.
TensorFlow Version: Uses TensorFlow 1.15.5 for compatibility with the existing codebase.
Database: Uses SQLite by default but can be switched to another database by changing the connection string.
##Error Handling
The service includes robust error handling and logging:

All errors are logged to face_service.log
HTTP exceptions with appropriate status codes are returned for client errors
Detailed error messages help with debugging
