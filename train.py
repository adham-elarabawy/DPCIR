from torch.autograd import Variable
import torch
from models.unet import UNetRes
from torch.optim import Adam
from torch.nn import L1Loss
from utils import util, datasets
from collections import OrderedDict
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
train_set = datasets.DatasetPatchNoise(opt, train=True)
test_set = datasets.DatasetPatchNoise(opt, train=False)

# Define Training Dataloader
train_loader = DataLoader(train_set,
                          batch_size=opt['training']['batch_size'],
                          num_workers=opt['training']['num_workers'],
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)

# Define Testing Dataloader
test_loader = DataLoader(test_set,
                         batch_size=1,
                         num_workers=1,
                         shuffle=False,
                         drop_last=False,
                         pin_memory=True)


# move model to GPU
if 'gpu_ids' in opt and torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"[Logging] Using GPU {str(device)}")
else:
    device = torch.device("cpu")
    print("[Logging] Using CPU")

model = model.to(device)

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
            log = f'epoch: {epoch}, step: {currStep}, lr: {scheduler.get_last_lr()}, loss: {loss}'
            print(log)

        # -------------------------------
        # 5) save model
        # -------------------------------
        if currStep % opt['training']['checkpoint_save'] == 0:
            print('Saving the model.')
            path_to_save_checkpoint = opt['training']['checkpoint_save_path'] + str(currStep) + '.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, path_to_save_checkpoint)

        # -------------------------------
        # 6) testing
        # -------------------------------

        if currStep % opt['training']['checkpoint_test'] == 0:

            avg_psnr = 0.0
            idx = 0

            for test_data in test_loader:
                idx += 1
                image_name_ext = os.path.basename(test_data['L_path'][0])
                img_name, ext = os.path.splitext(image_name_ext)

                img_dir = os.path.join(opt['training']['checkpoint_test_path'], img_name)
                util.mkdir(img_dir)

                # feed data
                noisy        = test_data['L'].to(device) # noisy image [augmented]
                ground_truth = test_data['H'].to(device) # clean image [ground truth]

                # test model
                model.eval()
                with torch.no_grad():
                    output = model.forward(noisy)
                model.train()

                # get the visuals
                visuals = OrderedDict()
                visuals['L'] = noisy.detach()[0].float().cpu()
                visuals['E'] = output.detach()[0].float().cpu()
                visuals['H'] = ground_truth.detach()[0].float().cpu()

                visuals = model.current_visuals()
                E_img = util.tensor2uint(visuals['E'])
                H_img = util.tensor2uint(visuals['H'])

                # -----------------------
                # save estimated image E
                # -----------------------
                save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, currStep))
                util.imsave(E_img, save_img_path)

                # -----------------------
                # calculate PSNR
                # -----------------------
                current_psnr = util.calculate_psnr(E_img, H_img, border=1)

                avg_psnr += current_psnr

            avg_psnr = avg_psnr / idx

            with open(opt['training']['checkpoint_save_path'] + "test_metrics.txt", 'a') as f: f.write(f"step:{currStep},avg_psnr:{avg_psnr}\n")

            # testing log
            log = f'epoch: {epoch}, step: {currStep}, Average PSNR: {avg_psnr}'
            print(log)

        currStep += 1
