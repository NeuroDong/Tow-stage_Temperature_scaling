# Tow-stage_Temperature_scaling
Two-stage temperature scaling: a balancing method between sample-specific and class-specific temperature scaling

# code structure
The method is written in a framework called [CustomKing](https://github.com/NeuroDong/CustomKing), which is a framework similar to [Detectron2](https://github.com/facebookresearch/detectron2) written by [Fvcore](https://github.com/facebookresearch/fvcore).

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
  
# Calibration method code
  See customKing/modeling/meta_arch/MultiClassification_calibration/Top_label_temperature.py
  
# Visualization
## Visualize the gap in different top-labels and components after temperature scaling
  python tools/Visualization/Calibration/plot_gap.py
## Visualize the interpretability of Coarse Scaling Vectors
  python tools/Visualization/Calibration/Interpretability.py
  


