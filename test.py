import os.path
import logging

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch

from utils import utils_logger
from utils import utils_model
from utils import utils_image as util

def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------
    n_channels = 2
    noise_level_img = 25                 # set AWGN noise level for noisy image
    noise_level_model = noise_level_img  # set noise level for model
    model_name = "model3_35000"
    model_path = 'checkpoints/model3/35000.pt'           # set denoiser model, 'drunet_gray' | 'drunet_color'
    testset_name = 'regular'             # set test set,  'bsd68' | 'cbsd68' | 'set12'
    x8 = False                           # default: False, x8 to boost performance
    show_img = True                      # default: False
    border = 0                           # shave border to calculate PSNR and SSIM

    testsets = 'testsets'                # fixed
    results = 'results'                  # fixed
    task_current = 'dn'                  # 'dn' for denoising
    result_name = testset_name + '_' + task_current + '_' + model_name

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    # torch.cuda.empty_cache()

    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------

    L_path = os.path.join(testsets, testset_name) # L_path, for Low-quality images
    E_path = os.path.join(results, result_name)   # E_path, for Estimated images
    util.mkdir(E_path)

    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    # ----------------------------------------
    # load model
    # ----------------------------------------

    from models.unet import UNetRes as net
    model = net(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model_state_dict'], strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    logger.info('Model path: {:s}'.format(model_path))
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('Params number: {}'.format(number_parameters))

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []

    logger.info('model_name:{}, model sigma:{}, image sigma:{}'.format(model_name, noise_level_img, noise_level_model))
    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)

    for idx, img in enumerate(L_paths):

        # ------------------------------------
        # (1) img_L
        # ------------------------------------

        img_name, ext = os.path.splitext(os.path.basename(img))

        # get the n_channels image
        img_H = util.imread_uint(img, n_channels=n_channels)
        img_L = util.uint2single(img_H)

        # Add noise without clipping
        np.random.seed(seed=0)  # for reproducibility
        img_L += np.random.normal(0, noise_level_img/255., img_L.shape)

        noisy_im = util.single2uint(img_L)
        # util.imshow(util.fill_in(util.single2uint(img_L)), title='Noisy image with noise level {}'.format(noise_level_img)) if show_img else None
        # util.imsave(util.single2uint(img_L), os.path.join(L_path, img_name + f'_noisy{noise_level_img}_' + ext))

        # reshapes the input as (1, 1, sizeX, sizeY)
        img_L = util.single2tensor4(img_L)
        # adds the noise level map to the image as an extra channel
        img_L = torch.cat((img_L, torch.FloatTensor([noise_level_model/255.]).repeat(1, 1, img_L.shape[2], img_L.shape[3])), dim=1)
        # move the image to the gpu memory
        img_L = img_L.to(device)


        # ------------------------------------
        # (2) img_E
        # ------------------------------------


        if not x8 and img_L.size(2)//8==0 and img_L.size(3)//8==0:
            img_E = model(img_L)
        elif not x8 and (img_L.size(2)//8!=0 or img_L.size(3)//8!=0):
            img_E = utils_model.test_mode(model, img_L, refield=64, mode=5)
        elif x8:
            img_E = utils_model.test_mode(model, img_L, mode=3)

        img_E = util.tensor2uint(img_E)
        # ax[0].imshow(noisy_im)
        # util.imsave(util.single2uint(img_E), os.path.join(E_path, img_name + f'_denoised{noise_level_img}_' + ext))
        # util.imshow(util.fill_in(img_E), title='De-noised image with noise level {}'.format(noise_level_img)) if show_img else None


    # --------------------------------
        # PSNR and SSIM
        # --------------------------------

        if n_channels == 1:
            img_H = img_H.squeeze() 
        psnr = util.calculate_psnr(img_E, img_H, border=border)
        ssim = util.calculate_ssim(img_E, img_H, border=border)
        test_results['psnr'].append(psnr)
        test_results['ssim'].append(ssim)
        # logger.info('{:s} - PSNR: {:.2f} dB; SSIM: {:.4f}.'.format(img_name+ext, psnr, ssim))

        # ------------------------------------
        # display /  results
        # ------------------------------------
        fig, big_axes = plt.subplots( figsize=(12.0, 12.0) , nrows=4, ncols=1, sharey=True)
        fig.suptitle(f'\'{model_name}\' Evaluation (PSNR: {psnr})', fontsize=16)
        for row, big_ax in enumerate(big_axes, start=1):
            # Turn off axis lines and ticks of the big subplot
            # obs alpha is 0 in RGBA string!
            big_ax.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
            # removes the white frame
            big_ax._frameon = False

        big_axes[0].set_title("Noisy\n", fontsize=14)
        big_axes[1].set_title("Denoised\n", fontsize=14)
        big_axes[2].set_title("Ground Truth\n", fontsize=14)
        big_axes[3].set_title("Denoised - Ground Truth\n", fontsize=14)

        # NOISY
        ax = fig.add_subplot(4,3,1, aspect=1)
        ax.imshow(noisy_im[:,:,0])
        ax.set_title('Channel 1')

        ax = fig.add_subplot(4,3,2, aspect=1)
        ax.imshow(noisy_im[:,:,1])
        ax.set_title('Channel 2')

        ax = fig.add_subplot(4,3,3, aspect=1)
        ax.imshow(util.fill_in(noisy_im))
        ax.set_title('Combined (zeroed ch3)')

        # DENOISED
        ax = fig.add_subplot(4,3,4, aspect=1)
        ax.imshow(img_E[:,:,0])
        ax.set_title('Channel 1')

        ax = fig.add_subplot(4,3,5, aspect=1)
        ax.imshow(img_E[:,:,1])
        ax.set_title('Channel 2')

        ax = fig.add_subplot(4,3,6, aspect=1)
        ax.imshow(util.fill_in(img_E))
        ax.set_title('Combined (zeroed ch3)')

        # GROUND TRUTH
        ax = fig.add_subplot(4,3,7, aspect=1)
        ax.imshow(img_H[:,:,0])
        ax.set_title('Channel 1')

        ax = fig.add_subplot(4,3,8, aspect=1)
        ax.imshow(img_H[:,:,1])
        ax.set_title('Channel 2')

        ax = fig.add_subplot(4,3,9, aspect=1)
        ax.imshow(util.fill_in(img_H))
        ax.set_title('Combined (zeroed ch3)')

        # DENOISED - GROUND TRUTH
        ax = fig.add_subplot(4,3,10, aspect=1)
        ax.imshow(img_E[:,:,0] - img_H[:,:,0], cmap = 'jet')
        ax.set_title('Channel 1')

        ax = fig.add_subplot(4,3,11, aspect=1)
        ax.imshow(img_E[:,:,1] - img_H[:,:,1], cmap = 'jet')
        ax.set_title('Channel 2')

        ax = fig.add_subplot(4,3,12, aspect=1)
        ax.imshow(util.fill_in(img_E) - util.fill_in(img_H), cmap = 'jet')
        ax.set_title('Combined (zeroed ch3)')

        fig.set_facecolor('w')
        plt.tight_layout(pad=0.5)
        plt.show()

        # util.imsave(img_E, os.path.join(E_path, img_name + f'_denoised{noise_level_img}_' + ext))

    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
    logger.info('Average PSNR/SSIM(RGB) - {} - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(result_name, ave_psnr, ave_ssim))


if __name__ == '__main__':

    main()
