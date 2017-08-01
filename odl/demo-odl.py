#! /usr/local/bin/python3

# E. Culurciello, July 2017
# This is a PyTorch demo for generic object detection and localization with bounding boxes
# it uses standard neural nets map thresholding and blob detections to find bounding boxes

import sys
import os
import time
import cv2 # install cv3, python3: brew install opencv3 --with-contrib --with-python3 --without-python
# add to profile: export PYTHONPATH=$PYTHONPATH:/usr/local/Cellar/opencv3/3.2.0/lib/python3.6/site-packages/
import numpy as np
import argparse
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from model_spatial import ModelDef # contains def of spatial model


def define_and_parse_args():
    # argument Checking
    parser = argparse.ArgumentParser(description="Object Detection Demo")
    parser.add_argument('model', help='model directory')
    parser.add_argument('-i', '--input', default='0', help='camera device index or file name, default 0')
    parser.add_argument('-s', '--size', type=int, default=224, help='network input size')
    parser.add_argument('-hd', type=bool, default=False, help='process full frame or resize to net eye size only')
    parser.add_argument('-tt', '--target', type=int, default=27, help='target category number from category file for detection in HD mode (27=person)')
    parser.add_argument('-th', '--threshold', type=float, default=0.5, help='detection threshold')
    return parser.parse_args()

def cat_file():
    # load classes file
    categories = []
    try:
        f = open(args.model + '/categories.txt', 'r')
        for line in f:
            cat = line.split(',')[0].split('\n')[0]
            if cat != 'classes':
                categories.append(cat)
        f.close()
        print('Number of categories:', len(categories))
    except:
        print('Error opening file ' + args.model + '/categories.txt')
        quit()
    return categories

def patch(m):
    s = str(type(m))
    s = s[str.rfind(s, '.')+1:-2]
    if s == 'Padding' and hasattr(m, 'nInputDim') and m.nInputDim == 3:
        m.dim = m.dim + 1
    if s == 'View' and len(m.size) == 1:
        m.size = torch.Size([1,m.size[0]])
    if hasattr(m, 'modules'):
        for m in m.modules:
            patch(m)


def spatialize(pretrained_model, new_model):
    # copy the weights and bias of each layer from loaded model to new model
    # convert Linear layer into convolution layer
    for i, j in zip(pretrained_model.modules(), new_model.modules()):
        if not list(i.children()):
            if isinstance(i, nn.Linear):
                j.weight.data = i.weight.data.view(j.weight.size())
                j.bias.data = i.bias.data
            else:
                if len(i.state_dict()) > 0:
                    j.weight.data = i.weight.data
                    j.bias.data = i.bias.data


# this is not used yet but would make a more general converter:
def spatialize_ec(model):
    # copy the weights and bias of each layer from loaded model to new model
    # convert Linear layer into convolution layer
    idx = 0
    for i in model.modules():
        if not list(i.children()):
            if isinstance(i, nn.Conv2d):
                llf = i.weight.size(0) # get last features size
                # print('llf',llf)
            if isinstance(i, nn.Linear):
                # print('w size:',i.weight.size())
                if idx == 0:
                    ks = int(np.sqrt(i.weight.size(1)/llf))
                    j = nn.Conv2d(llf, i.weight.size(0), kernel_size=ks)
                    idx = idx + 1
                else:
                    j = nn.Conv2d(i.weight.size(1), i.weight.size(0), kernel_size=1)
                # print(j)



print("FWDNXT PyTorch Object detector and localization")
args = define_and_parse_args()
categories = cat_file() # load category file
font = cv2.FONT_HERSHEY_SIMPLEX

# setup camera input:
if args.input[0] >= '0' and args.input[0] <= '9':
    cam = cv2.VideoCapture(int(args.input))
    # cam.set(3, xres)
    # cam.set(4, yres)
    usecam = True
else:
    image = cv2.imread(args.input)
    xres = image.shape[1]
    yres = image.shape[0]
    usecam = False

xres = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
yres = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('camera width, height:', xres, ' x ', yres)

# load pyTorch CNN moodels:
netfile = args.model + '/model.pth'
print('Importing PyTorch model from:', netfile)
model_dict = torch.load(netfile, map_location=lambda storage, loc: storage)
model = model_dict['model_def']
model.load_state_dict( model_dict['weights'] )

if args.hd: 
    print('Localizing category:', categories[args.target])
    new_model = ModelDef() # load spatial model definition with random weights
    spatialize(model, new_model)
    model = new_model
softMax = nn.Softmax() # to get probabilities out of CNN

model.eval()
print(model)

# image pre-processing functions:
transformsImage = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.Scale(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # needed for pytorch ZOO models on ImageNet (stats)
    ])

while True:
    startt = time.time()
    ret, frame = cam.read()
    if not ret:
        print('no camera input!')
        break

    if args.hd:
        pframe = cv2.resize(frame, dsize=(xres, yres))
    else:
        if xres > yres:
            frame = frame[:,int((xres - yres)/2):int((xres+yres)/2),:]
        else:
            frame = frame[int((yres - xres)/2):int((yres+xres)/2),:,:]

        pframe = cv2.resize(frame, dsize=(args.size, args.size))


    # prepare and normalize frame for processing:
    # if args.torch7:
    #     pframe = pframe[...,[2,1,0]]
    #     pframe = transformsImage(pframe)
    #     pframe = pframe.view(1, pframe.size(0), pframe.size(1), pframe.size(2))
    #     # process via CNN model:
    #     output = model.forward(pframe)
    #     output = softMax(output) # convert CNN output to probabilities
    #     output = output.numpy()[0]

    
    pframe = pframe[...,[2,1,0]]
    pframe = transformsImage(pframe)
    pframe = torch.autograd.Variable(pframe) # turn Tensor to variable required for pytorch processing
    pframe = pframe.unsqueeze(0)
    # process via CNN model:
    output = model.forward(pframe)

    if args.hd:
        # chose target category:
        targetp = output.squeeze(0)[args.target] # target probability map
        targetp = targetp.data.numpy()#[0] # get data from pytorch Variable, [0] = get vector from array
    else:
        output = softMax(output) # convert CNN output to probabilities

    output = output.data.numpy()[0] # get data from pytorch Variable, [0] = get vector from array

    if output is None:
        print('no output from CNN model file')
        break

    if args.hd:
        # print(output)
        # print('output shape:', output.shape)
        # print('target shape', targetp.shape)
        # print(targetp)
        # print(targetp.min(), targetp.max())
        targetp = (targetp-targetp.min())/targetp.max() # normalize

        # opencv connected components to get bounding boxes:
        # https://stackoverflow.com/questions/35854197/how-to-use-opencvs-connected-components-with-stats-in-python
        # Threshold it so it becomes binary
        ret, thresh = cv2.threshold(targetp, args.threshold, 1, cv2.THRESH_BINARY)
        # You need to choose 4 or 8 for connectivity type
        connectivity = 8  
        # Perform the operation
        targetcc = cv2.connectedComponentsWithStats(np.uint8(thresh), connectivity, cv2.CV_32S)
        # print(targetcc) # targetcc[2] has info on the bounding box [left, top, width, height, area]
        
        # GUI:
        scaling = int(xres/output.shape[2])
        # bounding boxes:
        for i in targetcc[2]:
            i = i*scaling # scaling to input image
            x1 = i[0]
            y1 = i[1]
            x2 = i[0]+i[2]
            y2 = i[1]+i[4]
            if x2-x1>xres/10 and y2-y1>yres/10 and x2-x1<xres*3/4: # only show large but not too large blobs
                cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 10)
                cv2.putText(frame, categories[args.target], (x1+10, y1+30), font, 1, (255,0,0), 2)

        cv2.imshow('win1', frame)

        # debug thresholded image:
        # pthresh = cv2.resize(thresh, dsize=(xres, yres)) # resize to full frame size:
        # cv2.imshow('win2', pthresh)

    else:
        # process output and print results:
        order = output.argsort()
        last = len(order)-1
        text = ''
        for i in range(min(5, last+1)):
            text += categories[order[last-i]] + ' (' + '{0:.2f}'.format(output[order[last-i]]*100) + '%) '

        # # overlay on GUI frame
        cv2.putText(frame, text, (10, yres-20), font, 0.5, (255, 255, 255), 1)
        cv2.imshow('win', frame)

    endt = time.time()

    sys.stdout.write("\rfps: "+'{0:.2f}'.format(1/(endt-startt)))
    sys.stdout.flush()

    if cv2.waitKey(1) == 27: # ESC to stop
        break

# end program:
cam.release()
cv2.destroyAllWindows()
