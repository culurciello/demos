#! /usr/local/bin/python3

# E. Culurciello, July 2017
# face demo

import sys
import os
import time
import glob
import cv2 # install cv3, python3: brew install opencv3 --with-contrib --with-python3 --without-python
# add to profile: export PYTHONPATH=$PYTHONPATH:/usr/local/Cellar/opencv3/3.2.0/lib/python3.6/site-packages/
import numpy as np
from scipy.spatial import distance
import argparse
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
# from model_spatial import ModelDef # contains def of spatial model
import dlib
from facealigner import FaceAligner # http://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/

def define_and_parse_args():
    # argument Checking
    parser = argparse.ArgumentParser(description="Face Identification Demo")
    parser.add_argument('--fid_db_dir', type=str, default='./face-db/', help='face database directory')
    parser.add_argument('--fid_num_face_db', type=int, default=10, help='number of faces to save in database per id')
    parser.add_argument('-i', '--input', default='0', help='camera device index or file name, default 0')
    parser.add_argument('--model', default='LightCNN_model_cpu.pth', type=str, help='path to trained model')
    parser.add_argument('--fsize', type=int, default=128, help='network input size')
    parser.add_argument('--fid_features_size', type=int, default=256, help='size of face features from neural net')
    parser.add_argument('--num_classes', default=79077, type=int, help='number of classes (default: 79077)')
    parser.add_argument('--extract', type=bool, default=False, help='extract face features')
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


def match_face_to_db(feats_in, fid_db):
    # find minimum distance of input features to database entries:
    min_dist = 1e12 # a huge number 
    for name_db, f_db in fid_db.items():
        for i in range(f_db.shape[0]):
            dist = distance.cosine( feats_in, f_db[i,:] )
            # print(dist, name_db) # to debug
            if dist < min_dist:
                min_dist = dist
                matched_id = name_db

    return matched_id


def load_fid_db():
    # face database is a dictionary of names and features (numpy array)
    fid_db = {}
    for fn in glob.glob(args.fid_db_dir+'/*.npy'):
        base = os.path.basename(fn)
        # print(os.path.splitext(base)[0]) # gives you the id name
        fid_db[os.path.splitext(base)[0]] = np.load(fn)

    return fid_db


demo_title = 'FWDNXT face demo'
print(demo_title)
args = define_and_parse_args()
font = cv2.FONT_HERSHEY_SIMPLEX

# if we want to extract faces examples for an id, we ask the name:
if args.extract: 
    print('Extracting face features for local face database. Please face the camera.')
    fid_name = input('Input your name: ')
    fid_name_dir =  args.fid_db_dir + fid_name
    # create face db features entry
    fid_features = np.zeros((args.fid_num_face_db, args.fid_features_size))
else:
    face_id_database = load_fid_db()
    print('>>> Loaded face identities database: ', list(face_id_database.keys()))

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

# xres = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
# yres = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('>>> Using camera with width, height:', xres, ' x ', yres)

# dlib face detector:
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
fa = FaceAligner(predictor, desiredFaceWidth=args.fsize)

netfile = args.model
print('Importing PyTorch model from:', netfile)
model_dict = torch.load(netfile)
model = model_dict['model_def']
model.load_state_dict( model_dict['weights'] )

transform = transforms.Compose([transforms.ToTensor()])

if args.extract: fid_counter = 0 # face id counter for taking examples

while True:
    startt = time.time()
    ret, frame = cam.read()
    if not ret:
        print('no camera input!')
        break

    gray = frame[:,:,1] # just take green channel instead of converting to grayscale!
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # frame converted to grayscale
    
    # detect faces:
    dets = detector(frame, 1)
    # print(" Number of faces detected: {} ".format(len(dets)))
    # for i, d in enumerate(dets):
        # print(" Detection {}: Left: {} Top: {} Right: {} Bottom: {} ".format(
            # i, d.left(), d.top(), d.right(), d.bottom()))

    # loop over face detections:
    matched_id = ''
    for rect in dets:
        # align the face using facial landmarks
        (x, y, w, h) = rect_to_bb(rect)
        faceAligned,_ = fa.align(frame, gray, rect)
        # overlay aligned face on frame bottom-left corner:
        frame[yres-faceAligned.shape[1]:yres, 0:faceAligned.shape[1], :] = faceAligned
        pgray = faceAligned[:,:,1] # just take green channel instead of converting to grayscale!
        # pgray = cv2.cvtColor(faceAligned, cv2.COLOR_BGR2GRAY)
        pgray = np.reshape(pgray, (128, 128, 1))

        # extract face features:
        pgray = transform(pgray)
        face_in = pgray.unsqueeze(0)
        face_in_var = torch.autograd.Variable(face_in, volatile=True)
        _, features = model(face_in_var) # [1,256] features
        features = features.data.numpy()[0]
        # print(features)

        if args.extract:
            if fid_counter < args.fid_num_face_db:
                fid_features[fid_counter] = features
                fid_counter += 1 # increment face id counter
        else:
            matched_id = match_face_to_db(features, face_id_database) # match faces to database:
            # print('>>> Identified: ', matched_id, '\n\n')

    # terminate program if we collected enough faces
    if args.extract and fid_counter >= args.fid_num_face_db:
        outfile = fid_name_dir + '.npy'
        np.save(outfile, fid_features)
        print('Collected ', args.fid_num_face_db, ' faces for id ', fid_name)
        print('Data saved to: ', outfile) 
        print('>>> Ending face collection.')
        break

    # show GUI:
    textsize = cv2.getTextSize(str(matched_id), font, 1, 2)[0] # (textsize[0], textsize[1]) are sizes X,Y
    cv2.putText(frame, str(matched_id), (int(xres-textsize[0]-30), yres-30), font, 1, (255,0,0), 2)
    cv2.imshow(demo_title, frame)

    # timings, etc:
    endt = time.time()

    # sys.stdout.write("\rfps: "+'{0:.2f}'.format(1/(endt-startt)))
    # sys.stdout.write("fps: "+'{0:.2f}'.format(1/(endt-startt)))
    sys.stdout.flush()

    if cv2.waitKey(1) == 27: # ESC to stop
        break

# end program:
cam.release()
cv2.destroyAllWindows()
