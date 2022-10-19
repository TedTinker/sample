#%%
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--image_size", type = int, default = 128)

try:    args = parser.parse_args()
except: args, _ = parser.parse_known_args()

image_size = args.image_size



import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConstrainedConv2d(nn.Conv2d):
    def forward(self, input):
        return nn.functional.conv2d(input, self.weight.clamp(min=-1.0, max=1.0), self.bias, self.stride,
                                    self.padding, self.dilation, self.groups)
    
def init_weights(m):
    try:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)
    except: pass
    


import pybullet as p
import pybullet_data

def get_physics(GUI, w, h):
    if(GUI):
        physicsClient = p.connect(p.GUI)
        p.resetDebugVisualizerCamera(1,90,-89,(w/2,h/2,w), physicsClientId = physicsClient)
    else:   
        physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    return(physicsClient)
# %%
