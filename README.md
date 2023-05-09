# Top-label_Temperature_scaling
An efficient confidence calibration method for multi-classification neural networks

# code structure
The method is written in a framework called CustomKing, which is a framework similar to [Detectron2](https://github.com/facebookresearch/detectron2) written by [Fvcore](https://github.com/facebookresearch/fvcore). The differences between CustomKing and Detectron2 are as follows:

(1) CustomKing is used for classification or calibration tasks, while Detectron2 is used for detection or segmentation tasks.

(2) The code does not need to be compiled when installing CustomKing, while Detectron2 needs to be compiled. Therefore, CustomKing can easily be compatible with the Linux and Windows systems.

# Tutorials
## Step1: Installing
  pip install -r requirements.txt  
## Step2：Set and check configuration of classification model
  See customKing/config/defaults.py
## Step3：Train the classification model and save the classification model output data (to be used for calibration).
  Python tools/Image_Classification/main.py
## Step4：Set and check configuration of calibration method
  See customKing/config/defaults.py
## Step5：Train the calibration method
  Python tools/Calibration/main.py
## Step6：View the performance of the calibration method
  The calibration performance of the training process is saved in a json file, and the file path can be set in the configuration file.
  


