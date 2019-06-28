# Adaptive Segmentation- Mask Attack

This repository contains the implementation of _Adaptive Segmentation Mask Attack (ASMA)_, a targeted adversarial example generation method for deep learning segmentation models. This attack was proposed in  "_Impact of Adversarial Examples on Deep Learning Models for Biomedical Segmentation. U. Ozbulak et al._" and published in the 22nd International Conference on Medical Image Computing and Computer Assisted Intervention, MICCAI-2019.

<img src="https://raw.githubusercontent.com/utkuozbulak/adaptive-segmentation-mask-attack/master/data/repository_examples/adversarial_optimization.gif?token=AESS2FAUP4VXC6HJGUYD7Z25D6J2Q">

## General Information
This repository is organized as follows:
* **src/** contains following python files:

  __adaptive_attack.py__ - This file contains the implementation of the main algorithm. Although we have tested the attack in binary segmentation problems, it is designed to work for multi-class problems as well.

  __eye_dataset.py__ - This file contains the pytorch dataset for the Glaucoma dataset[1] used in this study.
  
  __helper_functions.py__ - There a number of functions used in the respository such as calculating L2 distance, IOU etc. This file contains these functions.
  
  __unet_model.py__ - Unet model[2] used in this study is here.
  
  __main.py__ - An example way to run the attack on samples taken from the aforementioned dataset.
  
* **data/** 
  
  
  
  
  
  
  
  
  
  
  ## References
  [1] 
