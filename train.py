from torch.autograd import Variable
import torch
from models.unet import UNetRes
from torch.optim import Adam
from torch.nn import L1Loss
from utils import util, datasets
import argparse
import random
import numpy as np
from torch.utils.data import DataLoader


# get config parameters
parser = argparse.ArgumentParser()
parser.add_argument('--opt', type=str, required=True, help='Path to option JSON file.')
opt = util.parse(parser.parse_args().opt)

# config seed
seed = opt['training']['seed']
if seed is None:
    seed = random.randint(1, 10000)
print('Random seed: {}'.format(seed))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Instantiate a neural network model
in_nc  = opt['model']['in_numChannels']   # number of input channels
out_nc = opt['model']['out_numChannels']  # number of output channels
nc     = opt['model']['nc']               # feature map dims for each subsequent resblock set (match the dim of nb)
nb     = opt['model']['numResBlocks']     # number of res blocks in each downsample stage

upsample_mode   = opt['model']['upsample_mode']
downsample_mode = opt['model']['downsample_mode']
act_mode        = opt['model']['act_mode']

model = UNetRes(in_nc=in_nc, out_nc=1, nc=nc, nb=nb, act_mode=act_mode, downsample_mode=downsample_mode, upsample_mode=upsample_mode)

# Define L1 Loss Function
loss_fn = L1Loss()

# Define Adam Optimizer
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0)

# Create Datasets
train_set = datasets.DatasetPatchNoise(opt)

# Define Training Dataloader
train_loader = DataLoader(train_set,
                          batch_size=opt['training']['batch_size'],
                          num_workers=opt['training']['num_workers'],
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)

'''
# ----------------------------------------
# Step--4 (main training)
# ----------------------------------------
'''
current_step = 0
for epoch in range(1000000):  # keep running
    for i, train_data in enumerate(train_loader):


        # -------------------------------
        # 1) update learning rate
        # -------------------------------


        # -------------------------------
        # 2) feed patch pairs
        # -------------------------------


        # -------------------------------
        # 3) optimize parameters
        # -------------------------------


        # -------------------------------
        # 4) training information
        # -------------------------------


        # -------------------------------
        # 5) save model
        # -------------------------------


        # -------------------------------
        # 6) testing
        # -------------------------------

# Function to save the model
def saveModel():
    path = "./myFirstModel.pth"
    torch.save(model.state_dict(), path)

