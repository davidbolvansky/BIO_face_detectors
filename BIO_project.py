from skimage import img_as_ubyte
import pandas as pd
import numpy as np
import glob
import os
import sys
import pathlib
import shutil
import face_recognition
import matplotlib
import matplotlib.pyplot as plt

from skimage import io
from skimage.transform import resize
from PIL import Image, ImageEnhance
from Detectors import Detectors

brightness_factors = [0.1, 0.3, 0.6, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
downscale_factors = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

def filter_images_by_face_width(img_dir, face_width, img_limit, out_dir):
    images = list(pathlib.Path(img_dir).rglob("*.jpg"))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    c = 0
    for img in images:
        face_locs = face_recognition.face_locations(
            face_recognition.load_image_file(img))
        if len(face_locs) == 1:
            current_image = face_locs[0]
            # (top, right, bottom, left)
            face_width_px = current_image[1] - current_image[3]
            if face_width_px == face_width:
                shutil.copy(img, out_dir)
                c += 1
                if c == img_limit:
                    return


def do_analysis_brightness(base_dir):
    inner_dirs = [x[0] for x in os.walk(base_dir) if x[0] is not base_dir]
    r1 = []
    r2 = []
    r3 = []
    r4 = []
    r5 = []
    inner_dirs = sorted(inner_dirs)
    for dir in inner_dirs:
            det1 = 0
            det2 = 0
            det3 = 0
            det4 = 0
            det5 = 0
            all_files = len(os.listdir(dir))
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

            r1.append((det1 * 100.0) / all_files)
            r2.append((det2 * 100.0) / all_files)
            r3.append((det3 * 100.0) / all_files)
            r4.append((det4 * 100.0) / all_files)
            r5.append((det5 * 100.0) / all_files)
            print("Rasa: ", os.path.basename(dir))
            print("Face recognition: ", det1)
            print("Haar: ", det2)
            print("HoG: ", det3)
            print("CNN: ", det4)
            print("DNN: ", det5)

    #df = pd.DataFrame({'x': np.array(brightness_factors), 'd1': np.array(r1), 'd2': np.array(
    #    r2), 'd3': np.array(r3), 'd4': np.array(r4), 'd5': np.array(r5)})
    #df['x'] = df['x'].astype(str)


    plt.bar(np.array(brightness_factors) - 0.08, r1, color = 'blue', width = 0.03, label="Face recognition")
    plt.bar(np.array(brightness_factors) - 0.04, r2, color = 'green', width = 0.03, label="Haar (opencv)")
    plt.bar(np.array(brightness_factors) , r3, color = 'red', width = 0.03, label="HoG (dlib)")
    plt.bar(np.array(brightness_factors) + 0.04, r4, color = 'black', width = 0.03, label="CNN (dlib)")
    plt.bar(np.array(brightness_factors) + 0.08, r5, color = 'yellow', width = 0.03, label="DNN (cvlib)")
    plt.xticks(np.array(brightness_factors))
    plt.ylabel('úspešnosť detekcie (%)')
    plt.xlabel('faktor jasu')
    plt.title('Vplyv zmeny jasu na úspešnosť detekcie tváre')
    plt.legend()
    plt.show()

def do_analysis_scale(base_dir):
    inner_dirs = [x[0] for x in os.walk(base_dir) if x[0] is not base_dir]
    r1 = []
    r2 = []
    r3 = []
    r4 = []
    r5 = []
    inner_dirs = sorted(inner_dirs)
    for dir in reversed(inner_dirs):
            det1 = 0
            det2 = 0
            det3 = 0
            det4 = 0
            det5 = 0
            all_files = len(os.listdir(dir))
            for file in os.listdir(dir):
                #print("File: " , file)
                D = Detectors(os.path.join(dir, file))
                if D.detectFaceViaFaceRecognition():
                    det1 += 1
                if D.detectFaceViaHaarCascadeFaceDetector():
                    det2 += 1
                if D.detectFaceViaHoGFaceDetector():
                    det3 += 1
                print(file)
                if D.detectFaceViaCNNFaceDetector():
                    det4 += 1
                print(file)
                if D.detectFaceViaCVLIBFaceDetector():
                    det5 += 1

            r1.append((det1 * 100.0) / all_files)
            r2.append((det2 * 100.0) / all_files)
            r3.append((det3 * 100.0) / all_files)
            r4.append((det4 * 100.0) / all_files)
            r5.append((det5 * 100.0) / all_files)
            print("Rasa: ", os.path.basename(dir))
            print("Face recognition: ", det1)
            print("Haar: ", det2)
            print("HoG: ", det3)
            print("CNN: ", det4)
            print("DNN: ", det5)

    #df = pd.DataFrame({'x': np.array(brightness_factors), 'd1': np.array(r1), 'd2': np.array(
    #    r2), 'd3': np.array(r3), 'd4': np.array(r4), 'd5': np.array(r5)})
    #df['x'] = df['x'].astype(str)


    plt.bar(np.array(downscale_factors) - 0.025, r1, color = 'blue', width = 0.015, label="Face recognition")
    plt.bar(np.array(downscale_factors) - 0.01, r2, color = 'green', width = 0.015, label="Haar (opencv)")
    plt.bar(np.array(downscale_factors) , r3, color = 'red', width = 0.015, label="HoG (dlib)")
    plt.bar(np.array(downscale_factors) + 0.01, r4, color = 'black', width = 0.015, label="CNN (dlib)")
    plt.bar(np.array(downscale_factors) + 0.025, r5, color = 'yellow', width = 0.015, label="DNN (cvlib)")
    plt.xticks(np.array(downscale_factors))
    plt.ylabel('úspešnosť detekcie (%)')
    plt.xlabel('downscale faktor')
    plt.title('Vplyv zmeny rozlíšenia na úspešnosť detekcie tváre')
    plt.legend()
    plt.show()

#do_analysis_all("/home/xbolva00/face_detect/BIO_face_detectors/Dataset/tt/")
#filter_images_by_face_width("/home/xbolva00/face_detect/2003/01/19/big/", 90, 110, "./ou2t")


def downscale_images_in_folder(img_dir, downscale_factor):
    if downscale_factor > 1.0:
        print("Error: downscale factor must be < 1")
        return

    images = list(pathlib.Path(img_dir).rglob("*.jpg"))
    out_dir = os.path.join(pathlib.Path(img_dir).parents[0], os.path.dirname(img_dir) + "_downscale_" + str(downscale_factor))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for img in images:
        image = io.imread(img)
        new_x = (int)(image.shape[0] * downscale_factor)
        new_y = (int)(image.shape[1] * downscale_factor)
        image_downscaled = resize(image, (new_x, new_y),
                                  anti_aliasing=True)
        io.imsave(os.path.join(out_dir, os.path.basename(img)),
                  img_as_ubyte(image_downscaled))

def change_brightness_for_images_in_folder(img_dir, brightness_factor):
    images = list(pathlib.Path(img_dir).rglob("*.jpg"))
    out_dir = os.path.join(pathlib.Path(img_dir).parents[0], os.path.dirname(img_dir) + "_brightness_" + str(brightness_factor))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for img in images:
        im = Image.open(img)
        enhancer = ImageEnhance.Brightness(im)
        newimg = enhancer.enhance(brightness_factor) # set FACTOR > 1 to enhance contrast, < 1 to decrease
        newimg.save(os.path.join(out_dir, os.path.basename(img)))

def change_brightness(strain):
    for i in brightness_factors:
        change_brightness_for_images_in_folder("/home/xbolva00/BIO_face_detectors/Dataset/rozslisenie/" + strain + "/", i)

def change_scale(strain):
    for i in downscale_factors:
        downscale_images_in_folder("/home/xbolva00/BIO_face_detectors/Dataset/rozslisenie/" + strain + "/", i)

#change_brightness("fake")
do_analysis_brightness("/home/xbolva00/BIO_face_detectors/Dataset/rozslisenie/res/")