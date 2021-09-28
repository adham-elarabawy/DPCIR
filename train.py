from torch.autograd import Variable
import torch
from models.unet import UNetRes
from torch.optim import Adam
from torch.nn import L1Loss
import torch.nn as nn
from utils import util, datasets
from collections import OrderedDict
import argparse
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
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

# Instantiate Tensorboard instance
log_dir = os.path.join(opt['logging']['tb_logdir'], opt_filename)

util.mkdir(log_dir)

tb_train = SummaryWriter(os.path.join(log_dir, 'train'))
tb_eval  = SummaryWriter(os.path.join(log_dir, 'eval'))

# Instantiate a neural network model
numChannels = opt['numChannels']
in_nc       = opt['model']['in_numChannels']   # number of input channels
out_nc      = opt['model']['out_numChannels']  # number of output channels
nc          = opt['model']['nc']               # feature map dims for each subsequent resblock set (match the dim of nb)
nb          = opt['model']['numResBlocks']     # number of res blocks in each downsample stage

upsample_mode   = opt['model']['upsample_mode']
downsample_mode = opt['model']['downsample_mode']
act_mode        = opt['model']['act_mode']

assert (in_nc == numChannels + 1)
assert (out_nc == numChannels)

# Define model
model = UNetRes(in_nc=in_nc, out_nc=out_nc, nc=nc, nb=nb, act_mode=act_mode, downsample_mode=downsample_mode, upsample_mode=upsample_mode)

# Define L1 Loss Function
loss_fn = L1Loss()

# Define Adam Optimizer
optimizer = Adam(model.parameters(), lr=opt['training']['learning_rate'], weight_decay=0)

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
    device = torch.device("cuda:" + str(opt['gpu_ids'][0]))
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
    total_loss = 0
    for i, train_data in enumerate(train_loader):
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
        if currStep % opt['training']['checkpoint_log_loss'] == 0:
            train_metrics_path = os.path.join(opt['training']['checkpoint_save_path'], "train_metrics.txt")
            util.mkdir(opt['training']['checkpoint_save_path'])
            with open(train_metrics_path, 'a') as f: f.write(f"step:{currStep},loss:{loss}\n")


        # -------------------------------
        # 5) save model
        # -------------------------------
        if currStep % opt['training']['checkpoint_save'] == 0:
            print('Saving the model.')
            path_to_save_checkpoint = os.path.join(opt['training']['checkpoint_save_path'], str(currStep) + '.pt')
            util.mkdir(opt['training']['checkpoint_save_path'])
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
            eval_loss = 0.0
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

                # ensure that our dimensions are divisible by 8 (to make sure our conv + deconv operations are valid)
                h, w = ground_truth.size()[-2:]
                paddingBottom = int(np.ceil(h/8)*8-h)
                paddingRight = int(np.ceil(w/8)*8-w)
                ground_truth = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(ground_truth)

                # test model
                model.eval()
                with torch.no_grad():
                    output = model.forward(noisy, isTest=True)
                    eval_loss += loss_fn(output, ground_truth)
                model.train()

                # get the visuals
                visuals = OrderedDict()
                visuals['L'] = noisy.detach()[0].float().cpu()
                visuals['E'] = output.detach()[0].float().cpu()
                visuals['H'] = ground_truth.detach()[0].float().cpu()

                L_img = util.tensor2uint(visuals['L'])
                E_img = util.tensor2uint(visuals['E'])
                H_img = util.tensor2uint(visuals['H'])

                x_diff = abs(H_img.shape[0] - E_img.shape[0]) // 2
                y_diff = abs(H_img.shape[1] - E_img.shape[1]) // 2

                if x_diff:
                    E_img = E_img[x_diff:-x_diff, :]
                if y_diff:
                    E_img = E_img[:, y_diff:-y_diff]

                # -----------------------
                # save estimated image E and noisy version
                # -----------------------
                # util.imsave(E_img, os.path.join(img_dir, '{:s}_{:d}_E.png'.format(img_name, currStep)))

                # -----------------------
                # calculate PSNR
                # -----------------------
                current_psnr = util.calculate_psnr(E_img, H_img, border=1)

                avg_psnr += current_psnr

            avg_psnr = avg_psnr / idx
            eval_loss = eval_loss / idx
            tb_eval.add_scalar("[Testing] Average PSNR", avg_psnr, epoch)
            tb_eval.add_scalar("Loss", eval_loss, epoch)
            test_metrics_path = os.path.join(opt['training']['checkpoint_save_path'], "test_metrics.txt")
            util.mkdir(opt['training']['checkpoint_save_path'])
            with open(test_metrics_path, 'a') as f: f.write(f"step:{currStep},avg_psnr:{avg_psnr}\n")

            # testing log
            log = f'epoch: {epoch}, step: {currStep}, Average PSNR: {avg_psnr}'
            print(log)
        total_loss += loss
        currStep += 1

    tb_train.add_scalar("Loss", total_loss, epoch)
tb_train.close()
tb_eval.close()

