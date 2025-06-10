# Face Recognition with MTCNN and Pretrained Facenet.

This project allows accurate face recognition using MTCNN and Facenet, built on TensorFlow 2.x.

### 1. Clone the repository and install dependencies

```bash
git clone https://github.com/Dharc46/facenet-ft-mtcnn-for-exam-web.git
```

```bash
pip install -r requirements.txt
```

### 2. Download the model

Download Facenet pretrained models (https://bit.ly/3ixQH7o) then put them in the Models folder.

### 3. Results

Redirect to src folder

```bash
cd src
```

You needs to set the API key, for example:

```bash
$env:API_KEY="your-secret-api-key"
```

Then run the server:

```bash
python application.py
```

Go to http://127.0.0.1:8000/docs/ to see the result.


