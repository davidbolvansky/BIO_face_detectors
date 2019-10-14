import glob
import os
import sys
import pathlib
import shutil
import face_recognition

from Detectors import Detectors

def filter_images_by_face_width(img_dir, face_width, img_limit, out_dir):
    images = list(pathlib.Path(img_dir).rglob("*.jpg"))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    c = 0
    for img in images:
        face_locs = face_recognition.face_locations(face_recognition.load_image_file(img))
        if len(face_locs) == 1:
            current_image = face_locs[0]
            # (top, right, bottom, left)
            face_width_px = current_image[1] - current_image[3]
            if face_width_px == face_width:
                shutil.copy(img, out_dir)
                c += 1
                if c == img_limit: return

def do_analysis(base_dir):
    inner_dirs = [x[0] for x in os.walk(base_dir) if x[0] is not base_dir]
    det1 = 0
    det2 = 0
    det3 = 0
    det4 = 0
    det5 = 0
    for dir in inner_dirs:
        #print("Dir ", dir)
        for file in os.listdir(dir):
            #print("File: " , file)
            D = Detectors(os.path.join(dir, file))
            if D.detectFaceViaFaceRecognition():
                det1 += 1
            if D.detectFaceViaHaarCascadeFaceDetector():
                det2 += 1
            if D.detectFaceViaHoGFaceDetector():
                det3 += 1
            if D.detectFaceViaCNNFaceDetector():
                det4 += 1
            if D.detectFaceViaCVLIBFaceDetector():
                det5 += 1
        
        print("Rasa: ", os.path.basename(dir))
        print("Face recognization: ", det1)
        print("Haar: ", det2)
        print("HoG: ", det3)
        print("CNN: ", det4)
        print("DNN: ", det5)
        det1 = det2 = det3 = det4 = det5 = 0

                    

do_analysis("./Dataset")
#filter_images_by_face_width("/home/xbolva00/face_detect/2003/01/19/big/", 90, 110, "./ou2t")
