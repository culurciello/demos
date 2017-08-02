# convert LightCNN model to CPU
# E. Culurciello, August 2nd 2017

import torch
import torch.nn.parallel
from light_cnn import LightCNN # https://github.com/AlfredXiangWu/LightCNN/

model = LightCNN(pretrained=True, num_classes=79077)
model.eval()
#print('this is the model definition:')
#print(model)
print('These are the model state dict keys:')
print(model.state_dict)

checkpoint = torch.load('LightCNN_checkpoint.pth.tar')
state_dict = checkpoint['state_dict']
print('loaded state dict:')
print(state_dict.keys())

print('\nIn state dict keys there is an extra word inserted by model parallel: "module.". We remove it here:')
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    #if name[0:7] == 'features':
    new_state_dict[name] = v
    #else:
        #new_state_dict[k] = v

print('Now see converted state dict:')
print(new_state_dict.keys())
model.load_state_dict(new_state_dict)
model.cpu()

# saving model:
model_dict={}
model_dict['model_def']=model
model_dict['weights']=model.state_dict()
torch.save(model_dict, 'LightCNN_model_cpu.pth')
