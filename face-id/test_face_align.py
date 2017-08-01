#! /usr/local/bin/python3

# E. Culurciello, July 2017
# face demo

import sys
import os
import cv2 # install cv3, python3: brew install opencv3 --with-contrib --with-python3 --without-python
# add to profile: export PYTHONPATH=$PYTHONPATH:/usr/local/Cellar/opencv3/3.2.0/lib/python3.6/site-packages/
import numpy as np
import argparse
# from model_spatial import ModelDef # contains def of spatial model
import dlib
from facealigner import FaceAligner # http://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/

def define_and_parse_args():
    # argument Checking
    parser = argparse.ArgumentParser(description="Face test alignment")
    parser.add_argument('examplefile', help='example of face alignment file')
    parser.add_argument('-i', '--input', default='0', help='camera device index or file name, default 0')
    parser.add_argument('--fsize', type=int, default=128, help='network input size')
    return parser.parse_args()


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


print("FWDNXT test face aligner")
args = define_and_parse_args()


# setup camera input:
xres=640
yres=480
if args.input[0] >= '0' and args.input[0] <= '9':
    cam = cv2.VideoCapture(int(args.input))
    cam.set(3, xres)
    cam.set(4, yres)
    usecam = True
else:
    image = cv2.imread(args.input)
    xres = image.shape[1]
    yres = image.shape[0]
    usecam = False

print('camera width, height:', xres, ' x ', yres)

# dlib face detector:
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
fa = FaceAligner(predictor, desiredFaceWidth=args.fsize, desiredLeftEye=(0.35, 0.35))



# load example image of aligned image from a dataset:
# such as the: The aligned LFW images are uploaded on Baidu Yun.
# from : https://github.com/AlfredXiangWu/LightCNN
exampleim = cv2.imread(args.examplefile)
grayim = cv2.cvtColor(exampleim, cv2.COLOR_BGR2GRAY) # frame converted to grayscale
dets = detector(exampleim, 1)
print('Test dets:', dets)
# print(" Number of faces detected: {} ".format(len(dets)))
for i, rect in enumerate(dets):
    faceAligned, data = fa.align(exampleim, grayim, rect)
    print(data)
    # print(" Detection {}: Left: {} Top: {} Right: {} Bottom: {} ".format(
        # i, d.left(), d.top(), d.right(), d.bottom()))


# running a second time will tell us if we need to align more or not and if our alignment is not the same as the data:
dets = detector(faceAligned, 1)
grayim = cv2.cvtColor(faceAligned, cv2.COLOR_BGR2GRAY) # frame converted to grayscale
print('Test dets:', dets)
# print(" Number of faces detected: {} ".format(len(dets)))
for i, rect in enumerate(dets):
    faceAligned2, data2 = fa.align(faceAligned, grayim, rect)
    print(data2)
    # print(" Detection {}: Left: {} Top: {} Right: {} Bottom: {} ".format(
        # i, d.left(), d.top(), d.right(), d.bottom()))



# while True:
#     ret, frame = cam.read()
#     if not ret:
#         print('no camera input!')
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # frame converted to grayscale

#     # detect faces:
#     dets = detector(frame, 1)
#     # print('Cam Dets:', dets)
#     # print(" Number of faces detected: {} ".format(len(dets)))
#     # for i, d in enumerate(dets):
#         # print(" Detection {}: Left: {} Top: {} Right: {} Bottom: {} ".format(
#             # i, d.left(), d.top(), d.right(), d.bottom()))

#     # loop over the face detections:
#     for rect in dets:
#         # align the face using facial landmarks
#         (x, y, w, h) = rect_to_bb(rect)
#         faceAligned,_ = fa.align(frame, gray, rect)
#         frame[0:faceAligned.shape[1],0:faceAligned.shape[1],:] = faceAligned # show aligned face on frame
#         pgray = cv2.cvtColor(faceAligned, cv2.COLOR_BGR2GRAY)
#         pgray = np.reshape(pgray, (128, 128, 1))


#     # show GUI:
#     frame[0:exampleim.shape[1],128:128+exampleim.shape[1],:] = exampleim
#     cv2.imshow('win1', frame)

#     sys.stdout.flush()

#     if cv2.waitKey(1) == 27: # ESC to stop
#         break

# # end program:
# cam.release()
cv2.destroyAllWindows()
