import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util


class DatasetPatchNoise(data.Dataset):
    """
    # -----------------------------------------
    # Get L/H/M for denosing on AWGN with a range of sigma.
    # Only dataroot_H is needed.
    # -----------------------------------------
    # e.g., FDnCNN, H = f(cat(L, M)), M is noise level map
    # -----------------------------------------
    """

    def __init__(self, opt, train=True):

        super(DatasetPatchNoise, self).__init__()
        self.opt = opt
        self.n_channels = opt['numChannels']
        self.patch_size = self.opt['training']['patch_res']
        self.sigma = opt['sigma']
        self.sigma_min, self.sigma_max = self.sigma[0], self.sigma[1]
        self.sigma_test = opt['sigma_test']
        self.train = train
        if self.train:
            self.im_path = opt['datasets']['train']
        else:
            self.im_path = opt['datasets']['test']

        # -------------------------------------
        # get the path of H, return None if input is None
        # -------------------------------------
        self.paths_H = util.get_image_paths(self.im_path)

    def __getitem__(self, index):
        # -------------------------------------
        # get H image
        # -------------------------------------
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)

        L_path = H_path

        if self.train:
            """
            # --------------------------------
            # get L/H/M patch pairs
            # --------------------------------
            """
            H, W = img_H.shape[:2]

            # ---------------------------------
            # randomly crop the patch
            # ---------------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # ---------------------------------
            # augmentation - flip, rotate
            # ---------------------------------
            mode = random.randint(0, 7)
            patch_H = util.augment_img(patch_H, mode=mode)

            # ---------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # ---------------------------------
            img_H = util.uint2tensor3(patch_H)
            img_L = img_H.clone()

            # ---------------------------------
            # get noise level
            # ---------------------------------
            # noise_level = torch.FloatTensor([np.random.randint(self.sigma_min, self.sigma_max)])/255.0
            noise_level = torch.FloatTensor([np.random.uniform(self.sigma_min, self.sigma_max)])/255.0

            noise_level_map = torch.ones((1, img_L.size(1), img_L.size(2))).mul_(noise_level).float()  # torch.full((1, img_L.size(1), img_L.size(2)), noise_level)

            # ---------------------------------
            # add noise
            # ---------------------------------
            noise = torch.randn(img_L.size()).mul_(noise_level).float()
            img_L.add_(noise)

        else:
            """
            # --------------------------------
            # get L/H/M image pairs
            # --------------------------------
            """
            img_H = util.uint2single(img_H)
            img_L = np.copy(img_H)
            np.random.seed(seed=0)
            img_L += np.random.normal(0, self.sigma_test/255.0, img_L.shape)
            noise_level_map = torch.ones((1, img_L.shape[0], img_L.shape[1])).mul_(self.sigma_test/255.0).float()  # torch.full((1, img_L.size(1), img_L.size(2)), noise_level)

            # ---------------------------------
            # L/H image pairs
            # ---------------------------------
            img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)

        """
        # -------------------------------------
        # concat L and noise level map M
        # -------------------------------------
        """
        img_L = torch.cat((img_L, noise_level_map), 0)

        return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        return len(self.paths_H)