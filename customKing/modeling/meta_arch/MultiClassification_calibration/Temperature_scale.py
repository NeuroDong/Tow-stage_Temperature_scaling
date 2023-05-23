'''
Reference paper:《On calibration of modern neural networks》
'''

import torch
from torch import nn
from ..build import META_ARCH_REGISTRY
from customKing.solver.build import build_Calibration_lossFun


class Temperature_scale(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1,dtype=torch.float64)*1.5)
        self.lossfun = build_Calibration_lossFun(cfg)
        self.stage_num = 1
        self.stage = None

    def forward(self,Simple_vector,label_list):
        #温度放缩
        Simple_vector = Simple_vector/self.temperature

        #计算损失
        loss = self.lossfun(Simple_vector,label_list)
        
        softmaxed = None
        return Simple_vector,loss,None
    
@META_ARCH_REGISTRY.register()
def temperature_scale(cfg):
    return Temperature_scale(cfg)

if __name__=="__main__":
    model = Temperature_scale()
    input = torch.tensor([0.91,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01])
    after_calibration = model(input)
    print(after_calibration)
