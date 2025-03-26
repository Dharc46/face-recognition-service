# Face Recognition with MTCNN and FaceNet

This project implements a robust face recognition system using MTCNN for face detection and alignment, combined with FaceNet for feature extraction and SVM for classification.

## Overview

The system performs face recognition in three key stages:
1. **Face Detection & Alignment** - Using MTCNN to locate and align facial images
2. **Feature Extraction** - Using FaceNet to generate embeddings (feature vectors)
3. **Classification** - Using SVM to identify individuals based on their facial embeddings

## Installation

### Prerequisites

- Python 3.6+
- TensorFlow 1.15.5 (for compatibility with the existing codebase)
- GPU support recommended but not required

### Setup

1. Clone the repository:
```bash
git clone https://github.com/KienNL1927/face-recognition-mtcnn-facenet.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the pre-trained models:
Create a `Models` folder and download the FaceNet pre-trained model (20180402-114759.pb) - you can find it at: https://bit.ly/3ixQH7o

## Usage

### Dataset Preparation

1. Create a dataset structure in the following format:
```
Dataset/FaceData/raw/
    person1/
        image1.jpg
        image2.jpg
        ...
    person2/
        image1.jpg
        ...
```

2. Preprocess data to extract faces from original images:
```bash
python src/align_dataset_mtcnn.py Dataset/FaceData/raw Dataset/FaceData/processed --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.25
```

### Training

Train the SVM classifier:
```bash
python src/classifier.py TRAIN Dataset/FaceData/processed Models/20180402-114759.pb Models/facemodel.pkl --batch_size 1000
```

### Recognition

#### From Camera Feed
```bash
python src/face_rec.py
```

#### From an Image
```bash
python src/face_rec_image.py --path 'path/to/your_image.jpg'
```

## REST API Service

The project also provides a FastAPI-based microservice for face recognition with the following endpoints:

### API Endpoints

1. **Register a New Face**
   - `POST /register`
   - Register a new person with multiple face images

2. **Recognize Faces**
   - `POST /recognition`
   - Detect and recognize faces in an uploaded image

3. **Health Check**
   - `GET /health`
   - Simple health check endpoint

### Running the API Service

#### Using Uvicorn (Local Development)
Navigate to the src directory and run the service with Uvicorn:
```bash
cd src
uvicorn face_service:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at http://localhost:8000 with interactive documentation at http://localhost:8000/docs

#### Using Docker
The service can also be deployed using Docker:
```bash
docker-compose up -d
```

## Project Structure

```
├── src/
│   ├── align/                  # Face alignment code
│   ├── models/                 # Neural network model definitions
│   ├── classifier.py           # SVM classifier training
│   ├── facenet.py              # Main FaceNet implementation
│   ├── face_rec.py             # Real-time recognition from camera
│   ├── face_rec_image.py       # Recognition from image files
│   └── align_dataset_mtcnn.py  # Dataset preprocessing
├── Models/                     # Pre-trained models
├── Dataset/                    # Training data
└── requirements.txt            # Python dependencies
```

## Notes

- The system works best with well-lit, front-facing images
- For optimal accuracy, provide at least 5-10 different images per person during training
- Performance depends on the quality of the input images and the diversity of the training dataset
- GPU acceleration is recommended for processing speed, especially for real-time applications

## Credits

This implementation is based on:
- MTCNN paper: "Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks"
- FaceNet paper: "FaceNet: A Unified Embedding for Face Recognition and Clustering"
- Original implementation by David Sandberg and MìAI