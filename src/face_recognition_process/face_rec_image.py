import tensorflow as tf
from imutils.video import VideoStream
import argparse
import facenet
import imutils
import os
import sys
import math
import pickle
import align.detect_face
import numpy as np
import cv2
import collections
from sklearn.svm import SVC

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Path to the image you want to test on.', required=True)
    args = parser.parse_args()

    MINSIZE = 20
    THRESHOLD = [0.6, 0.7, 0.7]
    FACTOR = 0.709
    IMAGE_SIZE = 182
    INPUT_IMAGE_SIZE = 160
    CLASSIFIER_PATH = 'Models/facemodel.pkl'
    IMAGE_PATH = args.path  # Path to input image
    FACENET_MODEL_PATH = 'Models/20180402-114759.pb'

    # Load The Custom Classifier
    with open(CLASSIFIER_PATH, 'rb') as file:
        model, class_names = pickle.load(file)
    print("Custom Classifier, Successfully loaded")

    # Use TensorFlow 2.x features
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.set_logical_device_configuration(physical_devices[0], [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])

    # Create the session and load the model
    with tf.Graph().as_default():
        with tf.compat.v1.Session() as sess:
            # Load the facenet model
            print('Loading feature extraction model')
            facenet.load_model(FACENET_MODEL_PATH)

            # Get input and output tensors
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.shape[1]

            # Initialize MTCNN
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "src/align")

            # Read image file instead of using webcam
            frame = cv2.imread(IMAGE_PATH)
            frame = imutils.resize(frame, width=600)

            # Detect faces
            bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

            faces_found = bounding_boxes.shape[0]
            try:
                if faces_found > 1:
                    print("Only one face allowed, more than one detected.")
                elif faces_found > 0:
                    det = bounding_boxes[:, 0:4]
                    bb = np.zeros((faces_found, 4), dtype=np.int32)
                    for i in range(faces_found):
                        bb[i][0] = det[i][0]
                        bb[i][1] = det[i][1]
                        bb[i][2] = det[i][2]
                        bb[i][3] = det[i][3]
                        if (bb[i][3] - bb[i][1]) / frame.shape[0] > 0.25:
                            cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                            scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
                            scaled = facenet.prewhiten(scaled)
                            scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)

                            # Run the model to get embeddings
                            feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                            emb_array = sess.run(embeddings, feed_dict=feed_dict)

                            # Predict the class of the detected face
                            predictions = model.predict_proba(emb_array)
                            best_class_indices = np.argmax(predictions, axis=1)
                            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                            best_name = class_names[best_class_indices[0]]
                            print(f"Name: {best_name}, Probability: {best_class_probabilities}")

                            # If confidence is high enough, print the result
                            if best_class_probabilities > 0.7:
                                print(f"Detected face: {best_name} with confidence: {round(best_class_probabilities[0], 3)}")
                            else:
                                print("Unknown face detected")

            except Exception as e:
                print(e)
                pass

if __name__ == '__main__':
    main()
