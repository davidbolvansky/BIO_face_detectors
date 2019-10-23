import glob
import pathlib
import filecmp
import os
import face_recognition
from skimage import io, img_as_ubyte

DIR_ANNOTATIONS = "/home/xbolva00/Stiahnuté/FDDB-folds"
FACES_DIR = "/home/xbolva00/BIO_face_detectors/All"

txt = list(pathlib.Path(DIR_ANNOTATIONS).rglob("*-ellipseList.txt"))
jpg = list(pathlib.Path(FACES_DIR).rglob("*.jpg"))
#print(jpg)
f = 0
for i in jpg:
    name = i.stem
    bo = "_33." in name
    for j in txt:
        with open(j) as fp:
            while True:
                line = fp.readline()
                if not line: break
                if name in line:
                    s = "/home/xbolva00/Stiahnuté/" + line.strip()+ ".jpg"
                    one = io.imread(i)
                    two = io.imread(s)
                    if one.shape[0] == two.shape[0] and one.shape[1] == two.shape[1]:
                    #print(s)
                    #face_locs = face_recognition.face_locations(face_recognition.load_image_file(s))
                    #if len(face_locs) >= 1:
                        #current_image = face_locs[0]
                        # (top, right, bottom, left)
                        #face_width_px = current_image[1] - current_image[3]
                        #if face_width_px == 90:
                        #print("Line at {}: {}".format(j, line.strip()))
                        sa = line
                        line = fp.readline()
                        print("Line at {}: {}".format(j, sa))
                        print("--Num faces: {}".format(line.strip()))
                        f += 1

print(f)