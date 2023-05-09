# Top-label_Temperature_scaling
An efficient confidence calibration method for multi-classification neural networks

# code structure
The method is written in a framework called CustomKing, which is a framework similar to [Detectron2](https://github.com/facebookresearch/detectron2) written by [Fvcore](https://github.com/facebookresearch/fvcore). The differences between CustomKing and Detectron2 are as follows:

(1) CustomKing is used for classification or calibration tasks, while Detectron2 is used for detection or segmentation tasks.

(2) The code does not need to be compiled when installing CustomKing, while Detectron2 needs to be compiled. Therefore, CustomKing can easily be compatible with the Linux and Windows systems.

# Tutorials
## Step1: Installing
  pip install -r requirements.txt  
## Step2：Set and check configuration
  See customKing/config/defaults.py


