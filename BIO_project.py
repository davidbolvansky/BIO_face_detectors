import pandas as pd
import numpy as np
import glob
import os
import sys
import argparse
import pathlib
import shutil
import face_recognition
import matplotlib
import matplotlib.pyplot as plt

from skimage import io, img_as_ubyte
from skimage.transform import resize
from PIL import Image, ImageEnhance
from Detectors import Detectors

brightness_factors = [0.1, 0.3, 0.6, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
downscale_factors = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]


def filter_images_by_face_width(img_dir, face_width, img_limit, out_dir):
    images = list(pathlib.Path(img_dir).rglob("*.*"))

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
                    return img_limit
    return c


def do_analysis_brightness_effect(base_dir, brightness_factors):
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

    plt.bar(np.array(brightness_factors) - 0.08, r1,
            color='blue', width=0.03, label="Face recognition")
    plt.bar(np.array(brightness_factors) - 0.04, r2,
            color='green', width=0.03, label="Haar (opencv)")
    plt.bar(np.array(brightness_factors), r3,
            color='red', width=0.03, label="HoG (dlib)")
    plt.bar(np.array(brightness_factors) + 0.04, r4,
            color='black', width=0.03, label="CNN (dlib)")
    plt.bar(np.array(brightness_factors) + 0.08, r5,
            color='yellow', width=0.03, label="DNN (cvlib)")
    plt.xticks(np.array(brightness_factors))
    plt.ylabel('úspešnosť detekcie (%)')
    plt.xlabel('faktor jasu')
    plt.title('Vplyv zmeny jasu na úspešnosť detekcie tváre')
    plt.legend()
    plt.show()


def do_analysis_downscale_effect(base_dir, downscale_factors):
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

    plt.bar(np.array(downscale_factors) - 0.025, r1,
            color='blue', width=0.015, label="Face recognition")
    plt.bar(np.array(downscale_factors) - 0.01, r2,
            color='green', width=0.015, label="Haar (opencv)")
    plt.bar(np.array(downscale_factors), r3,
            color='red', width=0.015, label="HoG (dlib)")
    plt.bar(np.array(downscale_factors) + 0.01, r4,
            color='black', width=0.015, label="CNN (dlib)")
    plt.bar(np.array(downscale_factors) + 0.025, r5,
            color='yellow', width=0.015, label="DNN (cvlib)")
    plt.xticks(np.array(downscale_factors))
    plt.ylabel('úspešnosť detekcie (%)')
    plt.xlabel('downscale faktor')
    plt.title('Vplyv zmeny rozlíšenia na úspešnosť detekcie tváre')
    plt.legend()
    plt.show()


def downscale_images(img_dir, downscale_factor, base_out_dir):
    images = list(pathlib.Path(img_dir).rglob("*.*"))
    out_dir = os.path.join(base_out_dir, "downscale_" + str(downscale_factor))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for img in images:
        image = io.imread(img)
        new_x = (int)(image.shape[0] * downscale_factor)
        new_y = (int)(image.shape[1] * downscale_factor)
        image_downscaled = resize(image, (new_x, new_y), anti_aliasing=True)
        io.imsave(os.path.join(out_dir, os.path.basename(img)),
                  img_as_ubyte(image_downscaled))


def change_brightness_of_images(img_dir, brightness_factor, base_out_dir):
    images = list(pathlib.Path(img_dir).rglob("*.*"))
    out_dir = os.path.join(
        base_out_dir, "brightness_" + str(brightness_factor))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for img in images:
        im = Image.open(img)
        enhancer = ImageEnhance.Brightness(im)
        newimg = enhancer.enhance(brightness_factor)
        newimg.save(os.path.join(out_dir, os.path.basename(img)))


def main():
    global brightness_factors
    global downscale_factors
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", help="working directory",
                        default=os.getcwd(), nargs='?')
    parser.add_argument("--filter-face-width",
                        help="filter faces by certain face width, copy them to output dir", type=int, const=90, nargs='?')
    parser.add_argument("--filter-face-limit",
                        help="maximal number of faces filtered to output dir", type=int, default=100, nargs='?')
    parser.add_argument("--output-dir", help="working directory",
                        default=os.getcwd() + "/out", nargs='?')
    parser.add_argument("--change-brightness",
                        help="change brightness factor of images; factor > 1 to enhance brightness, factor < 1 to decrease", type=float)
    parser.add_argument("--change-brightness-default-factors",
                        help="change brightness of images with default factors: " + str(brightness_factors), type=float)
    parser.add_argument(
        "--downscale", help="downscale images by downscale factor; factor must be in range (0, 1>", type=float)
    parser.add_argument("--downscale-default-factors",
                        help="downscale with default factors: " + str(downscale_factors), type=float)
    parser.add_argument("--brightness-effect-analysis-default-factors",
                        help="try various face detection tools on lightened/dimmed face images and show results as a graph")
    parser.add_argument("--downscale-effect-analysis-default-factors",
                        help="try various face detection tools on downscalled face images and show results as a graph")
    args = parser.parse_args()
    print(args)
    if args.filter_face_width:
        count = filter_images_by_face_width(
            args.input_dir, args.filter_face_width, args.filter_face_limit, args.output_dir)
        print("Found %d images. Images were saved to: %s" %
              (count, args.output_dir))
    elif args.change_brightness:
        change_brightness_of_images(
            args.input_dir, args.change_brightness, args.output_dir)
        print("Images were saved to:", args.output_dir)
    elif args.downscale:
        if args.downscale > 1.0:
            print("Error: downscale factor must be < 1", file=sys.stderr)
        else:
            downscale_images(args.input_dir, args.downscale, args.output_dir)
            print("Images were saved to:", args.output_dir)
    elif args.change_brightness_default_factors:
        for f in brightness_factors:
            change_brightness_of_images(args.input_dir, f, args.output_dir)
        print("Images were saved to:", args.output_dir)
    elif args.downscale_default_factors:
        for f in downscale_factors:
            downscale_images(args.input_dir, f, args.output_dir)
        print("Images were saved to:", args.output_dir)
    elif args.brightness_effect_analysis_default_factors:
        do_analysis_brightness_effect(args.input_dir, brightness_factors)
    elif args.downscale_effect_analysis_default_factors:
        do_analysis_downscale_effect(args.input_dir, downscale_factors)


if __name__ == "__main__":
    main()
