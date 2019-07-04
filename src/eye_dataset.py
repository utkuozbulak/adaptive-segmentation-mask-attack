"""
@author: Utku Ozbulak
@repository: github.com/utkuozbulak
@article: Impact of Adversarial Examples on Deep Learning Models for Biomedical Segmentation
@conference: MICCAI-19
"""
import numpy as np
import glob
from PIL import Image

import torch
from torch.utils.data.dataset import Dataset


class EyeDatasetTest(Dataset):
    def __init__(self, image_path, mask_path, in_size=572, out_size=388):
        # paths to all images and masks
        self.mask_path = mask_path
        self.image_list = glob.glob(str(image_path) + str("/*"))
        self.data_len = len(self.image_list)
        self.in_size, self.out_size = in_size, out_size
        print('Dataset size:', self.data_len)

    def __getitem__(self, index):
        # --- Image operations --- #
        image_path = self.image_list[index]
        image_name = image_path[image_path.rfind('/')+1:]
        # Read image
        im_as_im = Image.open(image_path)
        im_as_np = np.asarray(im_as_im)
        im_as_np = im_as_np.transpose(2, 0, 1)
        # Crop image
        im_as_np = im_as_np[:, 70:70+408, 10:-10]  # 428 x 428
        # Pad image
        pad_size = 82
        im_as_np = np.asarray([np.pad(single_slice, pad_size, mode='edge')
                              for single_slice in im_as_np])
        """
        # Sanity check
        img1 = Image.fromarray(im_as_np.transpose(1, 2, 0))
        img1.show()
        """
        # Normalize image
        im_as_np = im_as_np/255
        # Convert numpy array to tensor
        im_as_tensor = torch.from_numpy(im_as_np).float()

        # --- Mask operations --- #
        # Read mask
        msk_as_im = Image.open(self.mask_path + '/' + image_name)
        msk_as_np = np.asarray(msk_as_im)
        # Crop mask
        msk_as_np = msk_as_np[70:70+388, 20:-20]
        msk_as_np.setflags(write=1)
        # Just in case if there are some gray-ish artifacts left in the image
        msk_as_np[msk_as_np > 20] = 255

        """
        # Sanity check
        img2 = Image.fromarray(msk_as_np)
        img2.show()
        """

        # Normalize mask to only 0 and 1
        msk_as_np = msk_as_np/255
        msk_as_tensor = torch.from_numpy(msk_as_np).long()  # Convert numpy array to tensor
        return (image_name, im_as_tensor, msk_as_tensor)

    def __len__(self):
        return self.data_len
