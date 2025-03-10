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
