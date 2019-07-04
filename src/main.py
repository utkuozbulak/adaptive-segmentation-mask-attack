"""
@author: Utku Ozbulak
@repository: github.com/utkuozbulak
@article: Impact of Adversarial Examples on Deep Learning Models for Biomedical Segmentation
@conference: MICCAI-19
"""
# In-repo imports
from eye_dataset import EyeDatasetTest
from helper_functions import load_model
from adaptive_attack import AdaptiveSegmentationMaskAttack


if __name__ == '__main__':
    # Glaucoma dataset
    eye_dataset = EyeDatasetTest('../data/image_samples',
                                 '../data/mask_samples')
    # GPU parameters
    DEVICE_ID = 0

    # Load model, change it to where you download the model to
    model = load_model('../models/eye_pretrained_model.pt')
    model.eval()
    model.cpu()
    model.cuda(DEVICE_ID)

    # Attack parameters
    tau = 1e-7
    beta = 1e-6

    # Read images
    im_name1, im12, mask12 = eye_dataset[0]
    im_name2, im22, mask22 = eye_dataset[1]

    # Perform attack
    adaptive_attack = AdaptiveSegmentationMaskAttack(DEVICE_ID, model, tau, beta)
    adaptive_attack.perform_attack(im22, mask22, mask12, [0, 1])
