'''
Refer to the paper "Beyond temperature scaling: Obtaining well-calibrated multi-class probabilities with dirichlet calibration"
The file is used for computing Classwise-ECE, refer to official code: https://github.com/dirichletcal/experiments_neurips/blob/master/calib/utils/functions.py
Include equal interval binning and equal sample binning
'''

import torch
from ..build import META_ARCH_REGISTRY
from torch.nn.functional import softmax
import numpy as np
from sklearn.preprocessing import label_binarize
from torch.utils.data import Dataset

class Classwise_ECE():
    def __init__(self,cfg,Dataset,mode = "equal_interval", softmaxed = None):
        self.mode = mode
        self.Dataset = Dataset
        self.softmaxed = softmaxed
        self.z_list, self.label_list = self.get_z_and_label()
        self.class_num = cfg.CALIBRATION_MODEL.NUM_CLASS
        self.cfg = cfg

    def get_z_and_label(self):
        label_list = []
        if isinstance(self.Dataset,Dataset):
            z_list = []
            for z,label in self.Dataset:
                if self.softmaxed == None:
                    z = softmax(z,dim=0,dtype=torch.float64)
                z = z.tolist()
                label = label.item()
                label_list.append(label)
                z_list.append(z)
        else:
            z,label = self.Dataset
            if self.softmaxed == None:
                z = softmax(z,dim=1,dtype=torch.float64)
            z_list = z.tolist()
            label_list = label.tolist()
        return z_list,label_list


    def binary_ECE(self,probs, y_true, power = 1, bins = 15):

        if self.mode == "equal_interval":
            idx = np.digitize(probs, np.linspace(0, 1, bins)) - 1
            
        if self.mode == "equal_sample":
            sort_x = np.sort(probs)
            bin_boundaries = []

            sample_num = self.cfg.CALIBRATION_EVALUATE.SAMPLE_NUM[0]
            if len(probs) < sample_num:
                    sample_num = len(probs)/2

            for i in range(len(probs)):
                if i%sample_num== 0:
                    bin_boundaries.append(sort_x[i])
            idx = [0]*len(probs)
            for j in range(len(bin_boundaries)-1):
                for k in range(len(probs)):
                    if probs[k] >= bin_boundaries[j] and probs[k] < bin_boundaries[j+1]:
                        idx[k]=j
                    if probs[k] == bin_boundaries[-1]:
                        idx[k] = len(bin_boundaries)-1
    
        bin_func = lambda p, y, idx: (np.abs(np.mean(p[idx]) - np.mean(y[idx])) ** power) * np.sum(idx) / len(probs)
        ece = 0
        for i in np.unique(idx):
            ece += bin_func(probs, y_true, idx == i)
        return ece

    def classwise_ECE(self,probs, y_true, power = 1, bins = 15):

        probs = np.array(probs)
        y_true = np.array(y_true)
        if not np.array_equal(probs.shape, y_true.shape):
            y_true = label_binarize(np.array(y_true), classes=range(probs.shape[1]))

        n_classes = probs.shape[1]

        return np.mean(
            [
                self.binary_ECE(
                    probs[:, c], y_true[:, c].astype(float), power = power, bins = bins
                ) for c in range(n_classes)
            ]
        )

    def compute_ECE(self):
        ece = self.classwise_ECE(self.z_list,self.label_list,power=1,bins = self.cfg.CALIBRATION_EVALUATE.INTERVAL_NUM[0])
        return ece
    
@META_ARCH_REGISTRY.register()
def classwise_ece_with_equal_interval(cfg,Dataset,softmaxed):
    return Classwise_ECE(cfg,Dataset,softmaxed=softmaxed)

@META_ARCH_REGISTRY.register()
def classwise_ece_with_equal_sample(cfg,Dataset,softmaxed):
    return Classwise_ECE(cfg,Dataset,mode = "equal_sample",softmaxed=softmaxed)