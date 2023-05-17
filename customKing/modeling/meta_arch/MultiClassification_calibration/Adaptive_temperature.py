'''
参考论文：《Adaptive temperature scaling for robust calibration of deep neural networks》
'''

import torch
from torch import nn
from ..build import META_ARCH_REGISTRY
from customKing.solver.build import build_Calibration_lossFun
import math


class Adaptive_temperature_scale(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.w_L = nn.Parameter(torch.ones(cfg.CALIBRATION_MODEL.NUM_CLASS,dtype=torch.float64))
        self.w_H = nn.Parameter(torch.tensor([1.],dtype=torch.float64))
        self.b = nn.Parameter(torch.tensor([1.],dtype=torch.float64))
        self.cfg = cfg
        self.softmax = nn.Softmax(dim=1)
        self.lossfun = build_Calibration_lossFun(cfg)
        self.stage_num = 1
        self.stage = None

    def forward(self,Simple_vector,label_list):
        
        LTS = torch.matmul(Simple_vector,self.w_L)
        x = self.softmax(Simple_vector)
        H_hat_list = []
        for xi in x:
            H_hat = 0
            for i in xi:
                H_hat = H_hat + i* torch.log(i)
            H_hat_list.append(H_hat)
        H_hat_list = torch.stack(H_hat_list)
        HTS = self.w_H * (H_hat_list/math.log(self.cfg.CALIBRATION_MODEL.NUM_CLASS)) + self.b
        a = LTS + HTS
        T = torch.log(1+torch.exp(a)).unsqueeze(1).expand(len(a),self.cfg.CALIBRATION_MODEL.NUM_CLASS)
        T = T.clamp(torch.finfo(torch.float64).eps,torch.finfo(torch.float64).max)

        Simple_vector = Simple_vector/T

        #计算损失
        loss = self.lossfun(Simple_vector,label_list)
        
        softmaxed = None
        return Simple_vector,loss,softmaxed
    
@META_ARCH_REGISTRY.register()
def adaptive_temperature_scale(cfg):
    return Adaptive_temperature_scale(cfg)