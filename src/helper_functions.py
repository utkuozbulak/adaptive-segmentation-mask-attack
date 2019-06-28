import numpy as np
from PIL import Image
import os
import copy

import torch


def save_input_image(modified_im, im_name, folder_name='result_images', save_flag=True):
    """
    Discretizes 0-255 (real) image from 0-1 normalized image
    """
    modified_copy = copy.deepcopy(modified_im)[0]
    modified_copy = modified_copy * 255
    # Box constraint
    modified_copy[modified_copy > 255] = 255
    modified_copy[modified_copy < 0] = 0
    modified_copy = modified_copy.transpose(1, 2, 0)
    modified_copy = modified_copy.astype('uint8')
    if save_flag:
        save_image(modified_copy, im_name, folder_name)
    return modified_copy


def save_prediction_image(pred_out, im_name, folder_name='result_images'):
    """
    Saves the prediction of a segmentation model as a real image
    """
    # Disc. pred image
    pred_img = copy.deepcopy(pred_out)
    pred_img = pred_img * 255
    pred_img[pred_img > 127] = 255
    pred_img[pred_img <= 127] = 0
    pred_img = pred_img.astype('uint8')
    save_image(pred_img, im_name, folder_name)


def save_image_difference(org_image, perturbed_image, im_name, folder_name='result_images'):
    """
    Finds the absolute difference between two images in terms of grayscale plaette
    """
    # Process images
    im1 = save_input_image(org_image, '', '', save_flag=False)
    im2 = save_input_image(perturbed_image, '', '', save_flag=False)
    # Find difference
    diff = np.abs(im1 - im2)
    # Sum over channel
    diff = np.sum(diff, axis=2)
    # Normalize
    diff_max = np.max(diff)
    diff_min = np.min(diff)
    diff = np.clip((diff - diff_min) / (diff_max - diff_min), 0, 1)
    # Enhance x 120, modify this according to your needs
    diff = diff*120
    diff = diff.astype('uint8')
    save_image(diff, im_name, folder_name)


def save_image(im_as_arr, im_name, folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    image_name_with_path = folder_name + '/' + str(im_name) + '.png'
    pred_img = Image.fromarray(im_as_arr)
    pred_img.save(image_name_with_path)


def load_model(path_to_model):
    """
    Loads pytorch model from disk
    """
    model = torch.load(path_to_model)
    return model


def calculate_mask_similarity(mask1, mask2):
    """
    Calculates IOU and pixel accuracy between two masks
    """
    # Calculate IoU
    intersection = mask1 * mask2
    union = mask1 + mask2
    # Update intersection 2s to 1
    union[union > 1] = 1
    iou = np.sum(intersection) / np.sum(union)

    # Calculate pixel accuracy
    correct_pixels = mask1 == mask2
    pixel_acc = np.sum(correct_pixels) / (correct_pixels.shape[0]*correct_pixels.shape[1])
    return (iou, pixel_acc)


def calculate_image_distance(im1, im2):
    """
    Calculates L2 and L_inf distance between two images
    """
    # Calculate L2 distance
    l2_dist = torch.dist(im1, im2, p=2).item()

    # Calculate Linf distance
    diff = torch.abs(im1 - im2)
    diff = torch.max(diff, dim=2)[0]  # 0-> item, 1-> pos
    linf_dist = torch.max(diff).item()
    return l2_dist, linf_dist
