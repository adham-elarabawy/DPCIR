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
from torch.optim.lr_scheduler import StepLR
import os


# get config parameters
parser = argparse.ArgumentParser()
parser.add_argument('--opt', type=str, required=True, help='Path to option JSON file.')
opt_filename = os.path.basename(parser.parse_args().opt).split('.')[0] # get options filename
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

# Define learning rate scheduler
lr_period = opt['training']['lr_period']
lr_gamma = opt['training']['lr_gamma']
scheduler = StepLR(optimizer, step_size=lr_period, gamma=lr_gamma)

# Create Datasets
train_set = datasets.DatasetPatchNoise(opt)

# Define Training Dataloader
train_loader = DataLoader(train_set,
                          batch_size=opt['training']['batch_size'],
                          num_workers=opt['training']['num_workers'],
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)


# move model to GPU
if 'gpu' in opt and torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"[Logging] Using GPU {str(device)}")
else:
    device = torch.device("cpu")
    print("[Logging] Using CPU")

'''
# ----------------------------------------
# Step--4 (main training)
# ----------------------------------------
'''
currStep = 0
for epoch in range(1000000):  # keep running
    for i, train_data in enumerate(train_loader):
        print("in loop")

        # -------------------------------
        # 1) get patch pairs
        # -------------------------------
        noisy        = train_data['L'].to(device) # noisy image [augmented]
        ground_truth = train_data['H'].to(device) # clean image [ground truth]

        # -------------------------------
        # 2) optimize parameters
        # -------------------------------
        optimizer.zero_grad()
        output = model.forward(noisy)
        loss = loss_fn(output, ground_truth)
        loss.backward()

        optimizer.step()

        # -------------------------------
        # 3) update learning rate
        # -------------------------------
        scheduler.step()

        # -------------------------------
        # 4) training information
        # -------------------------------
        if currStep % opt['training']['checkpoint_print'] == 0:
            log = f'epoch: {epoch}, step: {currStep}, lr: {scheduler.get_lr()}, loss: {loss}'
            print(log)

        # -------------------------------
        # 5) save model
        # -------------------------------
        if currStep % opt['training']['checkpoint_save'] == 0:
            print('Saving the model.')
            path_to_save_checkpoint = opt['training']['checkpoint_path'] + str(currStep) + '.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, path_to_save_checkpoint)

        # -------------------------------
        # 6) testing
        # -------------------------------

# # Function to save the model
# def saveModel():
#     path = "./myFirstModel.pth"
#     torch.save(model.state_dict(), path)