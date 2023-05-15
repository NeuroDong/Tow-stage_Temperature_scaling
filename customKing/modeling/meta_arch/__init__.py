# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

from turtle import forward
from .build import META_ARCH_REGISTRY, build_model  # isort:skip

#Classfication model
from .Image_classification.Resnext import Resnet20,Resnet110,Resnet18,Resnet34,Resnet50,Resnet101,Resnet152,ResNeXt29_8x64d,ResNeXt29_16x64d,ResNeXt50,ResNeXt101,Wide_resnet50_2,Wide_resnet101_2

#MultiClassification calibration model
from .MultiClassification_calibration.Temperature_scale import temperature_scale
from .MultiClassification_calibration.Dirichlet_calibration import dirichlet_calibration
from .MultiClassification_calibration.Top_label_temperature import top_label_temperature_scale
from .MultiClassification_calibration.Matrix_scale import matrix_scale
from .MultiClassification_calibration.Mix_n_match import mix_n_match
from .MultiClassification_calibration.Intra_order_preserving import intra_order_preserving_model
from .MultiClassification_calibration.Parameterized_temperature import parameterized_temperature_scale
from .MultiClassification_calibration.Adaptive_temperature import adaptive_temperature_scale

#Calibration evaluate method
from .Calibration_evaluate.Confidence_ECE import confidence_ece_with_equal_interval,confidence_ece_with_equal_sample
from .Calibration_evaluate.Classwise_ECE import classwise_ece_with_equal_interval,classwise_ece_with_equal_sample
from .Calibration_evaluate.Top_label_ECE import top_label_ece_with_equal_interval,top_label_ece_with_equal_sample
from .Calibration_evaluate.Thresholded_adaptive_calibration_error import tace_ece

__all__ = list(globals().keys())
