import os, sys
import warnings
# Hide deprecation warnings from numpy and tensorflow
warnings.filterwarnings('ignore',category=FutureWarning)
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

import face_recognition
import cv2
import dlib
import cvlib
sys.stderr = stderr

class Detectors:
  def __init__(self, filename):
    self.filename = filename

  def detectFaceViaFaceRecognition(self):
    faces = face_recognition.face_locations(
        face_recognition.load_image_file(self.filename))
    return len(faces) > 0

  def detectFaceViaHaarCascadeFaceDetector(self):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    img = cv2.imread(self.filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

  def detectFaceViaHoGFaceDetector(self):
    hog_face_detector = dlib.get_frontal_face_detector()
    faces = hog_face_detector(dlib.load_rgb_image(
        self.filename), 1)  # 1 - unsampled count
    return len(faces) > 0

  def detectFaceViaCNNFaceDetector(self):
    cnn_face_detector = dlib.cnn_face_detection_model_v1(
        "mmod_human_face_detector.dat")
    faces = cnn_face_detector(dlib.load_rgb_image(
        self.filename), 1)  # 1 - unsampled count
    return len(faces) > 0

  def detectFaceViaCVLIBFaceDetector(self):
    img = cv2.imread(self.filename)
    faces, _ = cvlib.detect_face(img) 
    return len(faces) > 0
