'''
Reference paper:《Mix-n-Match: Ensemble and Compositional Methods for Uncertainty Calibration in Deep Learning》
Official code: https://github.com/zhang64-llnl/Mix-n-Match-Calibration
'''

import torch
from torch import nn
from ..build import META_ARCH_REGISTRY
from torch.nn.functional import softmax
import json
from customKing.solver.build import build_Calibration_lossFun

class Mix_n_Match(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.h = 0
        self.cfg = cfg
        self.w = nn.Parameter(torch.ones(2)*0.333)
        torch.clamp(self.w,0,1)
        #初始化损失函数
        self.lossfun = torch.nn.NLLLoss()
        self.stage_num = 1
        self.stage = None    
    
    def get_T(self,file_path):
        data = []
        with open(file_path) as f:
            for line in f:
                data.append(json.loads(line))
        t = 1.5
        for i in data:
            if 'temperature' in i.keys():
                t = i["temperature"]
        return torch.tensor(t).cuda()

    def forward(self,Simple_vector,label_list):
        if self.h == 0:
            file_list = self.cfg.CALIBRATION_MODEL.OUTPUT_DIR.split("/")
            file_list[-2] = "temperature_scale"
            file_path = "/".join(file_list) + "logging.json"
            #file_path = r"output/Calibration/Cifar10_SEED20/Wide_resnet50_2/CalibrationTrain/temperature_scale/logging.json"
            self.temperature = self.get_T(file_path)
            self.h = 1
        
        t = self.temperature
        self.w.data = self.w.clamp(0.1,0.5)

        p1 = softmax(Simple_vector,dim=1,dtype=torch.float64)
        Simple_vector = Simple_vector / t
        p0 = softmax(Simple_vector,dim=1,dtype=torch.float64)
        p2 = torch.ones_like(p0) / self.cfg.CALIBRATION_MODEL.NUM_CLASS
        p = self.w[0] * p0 + self.w[1] * p1 + (1-self.w[0]-self.w[1]) * p2

        log_p = torch.log(p)
        loss = self.lossfun(log_p,label_list)

        softmaxd = True
        return p,loss,softmaxd
    
@META_ARCH_REGISTRY.register()
def mix_n_match(cfg):
    return Mix_n_Match(cfg)