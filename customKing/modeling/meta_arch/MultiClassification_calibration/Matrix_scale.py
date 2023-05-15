import torch
from torch import nn
from ..build import META_ARCH_REGISTRY
from customKing.solver.build import build_Calibration_lossFun

class Matrix_scale(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.lin = nn.Linear(cfg.CALIBRATION_MODEL.NUM_CLASS,cfg.CALIBRATION_MODEL.NUM_CLASS)
        self.lin.to(torch.float64)
        self.lossfun = build_Calibration_lossFun(cfg)
        self.stage_num = 1
        self.stage = None

    def forward(self,Simple_vector,label_list):
        Simple_vector = self.lin(Simple_vector)
        #计算损失
        loss = self.lossfun(Simple_vector,label_list)

        softmaxed = None
        return Simple_vector,loss,None
    
@META_ARCH_REGISTRY.register()
def matrix_scale(cfg):
    return Matrix_scale(cfg)

if __name__=="__main__":
    model = Matrix_scale()
    input = torch.tensor([0.91,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01])
    after_calibration = model(input)
    print(after_calibration)
