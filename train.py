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

# Instantiate a neural network model
model = UNetRes()

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

        current_step += 1

        # -------------------------------
        # 1) update learning rate
        # -------------------------------
        model.update_learning_rate(current_step)

        # -------------------------------
        # 2) feed patch pairs
        # -------------------------------
        model.feed_data(train_data)

        # -------------------------------
        # 3) optimize parameters
        # -------------------------------
        model.optimize_parameters(current_step)

        # -------------------------------
        # 4) training information
        # -------------------------------
        if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
            logs = model.current_log()  # such as loss
            message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
            for k, v in logs.items():  # merge log information into message
                message += '{:s}: {:.3e} '.format(k, v)
            logger.info(message)

        # -------------------------------
        # 5) save model
        # -------------------------------
        if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
            logger.info('Saving the model.')
            model.save(current_step)

        # -------------------------------
        # 6) testing
        # -------------------------------
        if current_step % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:

            avg_psnr = 0.0
            idx = 0

            for test_data in test_loader:
                idx += 1
                image_name_ext = os.path.basename(test_data['L_path'][0])
                img_name, ext = os.path.splitext(image_name_ext)

                img_dir = os.path.join(opt['path']['images'], img_name)
                util.mkdir(img_dir)

                model.feed_data(test_data)
                model.test()

                visuals = model.current_visuals()
                E_img = util.tensor2uint(visuals['E'])
                H_img = util.tensor2uint(visuals['H'])

                # -----------------------
                # save estimated image E
                # -----------------------
                save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
                util.imsave(E_img, save_img_path)

                # -----------------------
                # calculate PSNR
                # -----------------------
                current_psnr = util.calculate_psnr(E_img, H_img, border=border)

                logger.info('{:->4d}--> {:>10s} | {:<4.2f}dB'.format(idx, image_name_ext, current_psnr))

                avg_psnr += current_psnr

            avg_psnr = avg_psnr / idx

            # testing log
            logger.info('<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB\n'.format(epoch, current_step, avg_psnr))


# Function to save the model
def saveModel():
    path = "./myFirstModel.pth"
    torch.save(model.state_dict(), path)

