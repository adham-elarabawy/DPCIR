import os
from tqdm import tqdm
import wandb
import torch
import argparse
import random
import numpy as np
import torch.nn as nn
from torch.nn import L1Loss
from torch.optim import Adam
from models.unet import UNetRes
from utils import util, datasets
from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR



def train(config, model):
    # set model weights to training mode
    model.train()

    # Define L1 Loss Function
    loss_fn = L1Loss()

    # Define Adam Optimizer
    optimizer = Adam(model.parameters(), lr=config['hyperparams']['learning_rate'])

    # Define learning rate scheduler
    lr_period = config['hyperparams']['lr_period']
    lr_gamma = config['hyperparams']['lr_gamma']
    scheduler = StepLR(optimizer, step_size=lr_period, gamma=lr_gamma)

    # Create training/testing dataloaders
    train_loader, test_loader = create_loaders(config)

    # Set up cpu/gpu torch device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if config['logging']['verbose']:
        print(f'[Log] Using {device}')

    # Move model to cpu/gpu
    model = model.to(device)

    # Keep track of model with wandb
    wandb.watch(model, loss_fn, log="all", log_freq=10)

    currStep = 0
    for epoch in range(config['hyperparams']['epochs']):
        totalLoss = 0

        with tqdm(train_loader, desc=f'Epoch {epoch}', unit="batch") as steps:
            for step, train_data in enumerate(steps):
                # Get patch pairs
                noisy        = train_data['L'].to(device) # noisy image [augmented]
                ground_truth = train_data['H'].to(device) # clean image [ground truth]

                # Forward pass through model
                optimizer.zero_grad()
                output = model.forward(noisy)

                # Compute loss + backwards pass
                loss = loss_fn(output, ground_truth)
                loss.backward()
                optimizer.step()

                # updating total loss
                totalLoss += loss.item()

                # logging and saving
                log(config, epoch, currStep, scheduler, loss.item())
                save(config, model, epoch, currStep, loss, optimizer)

                # evaluating on test dataset
                test(config, model, device, loss_fn, epoch, currStep, test_loader)

                # set model back into train mode
                model.train()

                # update progress bar
                steps.set_postfix(loss=totalLoss / (step+1))

                # update current step count
                currStep += 1

def log(config, epoch, currStep, scheduler, loss):
    if currStep % config['logging']['period'] == 0:
        wandb.log({"epoch": epoch, "step": currStep, "loss": loss, "lr": scheduler.get_last_lr()})

def save(config, model, epoch, currStep, loss, optimizer):
    if currStep % config['save']['period'] == 0:
        path_to_save_checkpoint = os.path.join('checkpoints', config['name'], 'models')
        util.mkdir(path_to_save_checkpoint)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, os.path.join(path_to_save_checkpoint, str(currStep) + '.pth'))

def get_model(config):
    in_nc       = config['model']['in_numChannels']   # number of input channels
    out_nc      = config['model']['out_numChannels']  # number of output channels
    nc          = config['model']['nc']               # feature map dims for each subsequent resblock set (match the dim of nb)
    nb          = config['model']['numResBlocks']     # number of res blocks in each downsample stage

    model = UNetRes(in_nc=in_nc, out_nc=out_nc, nc=nc, nb=nb, act_mode="R", downsample_mode="strideconv", upsample_mode="convtranspose")
    return model

def create_loaders(config):
    # Create Datasets
    train_set = datasets.DatasetPatchNoise(config, train=True, addNoiseLevelMap=False)
    test_set = datasets.DatasetPatchNoise(config, train=False, addNoiseLevelMap=False)

    # Define Training Dataloader
    train_loader = DataLoader(train_set,
                              batch_size=config['hyperparams']['batch_size'],
                              num_workers=1,
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

    return train_loader, test_loader

def test(config, model, device, loss_fn, epoch, currStep, test_loader):
    if currStep % config['test']['period'] == 0:

        # test model
        model.eval()

        avg_psnr = 0.0
        eval_loss = 0.0
        idx = 0

        for test_data in test_loader:
            idx += 1
            image_name_ext = os.path.basename(test_data['L_path'][0])
            img_name, ext = os.path.splitext(image_name_ext)

            img_dir = os.path.join('checkpoints', config['name'], 'testing', img_name)
            util.mkdir(img_dir)

            # feed data
            noisy        = test_data['L'].to(device) # noisy image [augmented]
            ground_truth = test_data['H'].to(device) # clean image [ground truth]

            # ensure that our dimensions are divisible by 8 (to make sure our conv + deconv operations are valid)
            h, w = ground_truth.size()[-2:]
            paddingBottom = int(np.ceil(h/8)*8-h)
            paddingRight = int(np.ceil(w/8)*8-w)
            ground_truth = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(ground_truth)


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
            # calculate PSNR
            # -----------------------
            current_psnr = util.calculate_psnr(E_img, H_img, border=1)

            avg_psnr += current_psnr

        avg_psnr = avg_psnr / idx
        eval_loss = eval_loss / idx
        wandb.log({"epoch": epoch, "step": currStep, "eval_loss": eval_loss, "avg_psnr": avg_psnr})

if __name__ == '__main__':
    ### CONFIG INIT ###

    # get model configuration from cli
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, required=True, help='Path to option JSON file.')
    model_name = os.path.basename(parser.parse_args().opt).split('.')[0] # get options filename
    config = util.parse(parser.parse_args().opt)

    config['name'] = model_name

    # initialize project tracking
    wandb.init(project="DPCIR", entity=config['logging']['username'])

    wandb.config = {
        "learning_rate": config['hyperparams']['learning_rate'],
        "batch_size": config['hyperparams']['batch_size'],
        "num_channels": config['model']['numChannels'],
        "sigma": config['hyperparams']['sigma'],
        "sigma_test": config['hyperparams']['sigma_test'],
        "noise_level_map": 0
    }

    # get model
    model = get_model(config)

    # train model
    train(config, model)