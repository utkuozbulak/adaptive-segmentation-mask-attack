# Adaptive Segmentation Mask Attack

This repository contains the implementation of _Adaptive Segmentation Mask Attack (ASMA)_, a targeted adversarial example generation method for deep learning segmentation models. This attack was proposed in  "_Impact of Adversarial Examples on Deep Learning Models for Biomedical Segmentation. U. Ozbulak et al._" and published in the 22nd International Conference on Medical Image Computing and Computer Assisted Intervention, MICCAI-2019.

<br /> 
<img src="https://raw.githubusercontent.com/utkuozbulak/adaptive-segmentation-mask-attack/master/data/repository_examples/adversarial_optimization.gif?token=AESS2FAUP4VXC6HJGUYD7Z25D6J2Q">

## Citation
If you find the code in this repository relevant to your research, consider citing our paper. Also, feel free to use any visuals available.

	@article{,
	  author = {},
	  title = {},
	  year = {},
	  publisher = {},
	  journal = {}
	}

## General Information
This repository is organized as follows:
* **Code** - *src/* folder contains following python files:

  __adaptive_attack.py__ - This file contains the implementation of the main algorithm. Although we have tested the attack in binary segmentation problems, it is designed to work for multi-class problems as well.

  __eye_dataset.py__ - This file contains the pytorch dataset for the Glaucoma dataset[1] used in this study.
  
  __helper_functions.py__ - There a number of functions used in the respository such as calculating L2 distance, IOU etc. This file contains these functions.
  
  __unet_model.py__ - Unet model[2] used in this study is here.
  
  __main.py__ - An example way to run the attack on samples taken from the aforementioned dataset.
  
* **Data** - *data/* folder contains a couple of examples for testing purposes. The data we used in this study can be taken from [1].
  
* **Model** - Example model used in this repository can be downloaded from https://www.dropbox.com/s/6ziz7s070kkaexp/eye_pretrained_model.pt . _helper_functions.py_ contains a function to load this file and _main.py_ contains an exaple that uses this model.

## Requirements:
```
python > 3.5
torch >= 0.4.0
torchvision >= 0.1.9
numpy >= 1.13.0
PIL >= 1.1.7
```
  
  
## References
[1]  Pena-Betancor C., Gonzalez-Hernandez M., Fumero-Batista F., Sigut J., Medina-Mesa E., Alayon S., Gonzalez M. _Estimation of the relative amount of hemoglobin in the cup and neuroretinal rim using stereoscopic color fundus images._

[2] Ronneberger, O., Fischer, P., Brox, T. _U-Net: Convolutional networks for biomedical image segmentation._
