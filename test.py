import Detectors
import os

for i in os.listdir("/home/xbolva00/BIO_face_detectors/euro_b/downscale_0.1/"):
    print(i)
    D = Detectors.FaceDetectors("/home/xbolva00/BIO_face_detectors/euro_b/downscale_0.1/" + i, False)
    print(D.detectFaceViaMTCNNFaceDetector())