# classifier.py: trains a classifier on face database
# e. culurciello, August 4th 2017

import os
import sys
import glob
from itertools import count
import random
import numpy as np
import torch
import torch.autograd
import torch.nn.functional as F
from torch.autograd import Variable


def load_fid_db():
    # face database is a dictionary of names and features (numpy array)
    fid_db = {}
    for fn in glob.glob(sys.argv[1]+'/*.npy'):
        base = os.path.basename(fn)
        # print(os.path.splitext(base)[0]) # gives you the id name
        fid_db[os.path.splitext(base)[0]] = np.load(fn)

    return fid_db


def get_batch(db, batch_size=32, feat_vec_size=256):
    """Builds a batch i.e. (x, f(x)) pair."""

    x = torch.zeros(batch_size, feat_vec_size)
    y = torch.LongTensor(batch_size).zero_()

    for i in range(batch_size):
       y[i] = np.random.randint( len(face_id_database.keys()) )
       iy = list(face_id_database.keys())[int(y[i])]
       ix = face_id_database[iy][np.random.randint( face_id_database[iy].shape[0] )]
       x[i] = torch.from_numpy(ix)

    return Variable(x), Variable(y)


# load dataset / database:
face_id_database = load_fid_db()
print('>>> Loaded face identities database: ', list(face_id_database.keys()))


# Define model
feat_vec_size = 256
model = torch.nn.Linear(feat_vec_size, len(face_id_database.keys()))

for batch_idx in count(1):
    # Get data
    batch_x, batch_y = get_batch(db=face_id_database, batch_size=32, feat_vec_size=256)

    # Reset gradients
    model.zero_grad()

    # Forward pass
    output = F.nll_loss(F.log_softmax( model(batch_x) ), batch_y)
    loss = output.data[0]

    # Backward pass
    output.backward()

    # Apply gradients
    for param in model.parameters():
        param.data.add_(-0.01 * param.grad.data)

    # Stop criterion
    if loss < 1e-3:
        break
    print('Loss: {:.6f} after {} batches'.format(loss, batch_idx))

# saving model:                                                                                                             
model_dict={}                                                                                                               
model_dict['model_def'] = model                                                                                          
model_dict['weights'] = model.state_dict()                                                                                    
torch.save(model_dict, sys.argv[1] + '/classifier.pth') 
