'''
Reference paper:《Beyond temperature scaling: Obtaining well-calibrated multi-class probabilities with dirichlet calibration》
'''


import torch
from torch import nn
from ..build import META_ARCH_REGISTRY
from torch.nn.functional import softmax
from customKing.solver.build import build_Calibration_lossFun

class Dirichlet_calibration(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.lin = nn.Linear(cfg.CALIBRATION_MODEL.NUM_CLASS,cfg.CALIBRATION_MODEL.NUM_CLASS)
        self.lin.to(torch.float64)
        self.cfg = cfg
        self.lossfun = build_Calibration_lossFun(cfg)
        self._init_weight()
        self.stage_num = 1
        self.stage = None

    def _init_weight(self):
        nn.init.eye_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self,Simple_vector,label_list):
        Simple_vector = softmax(Simple_vector,dim=1,dtype=torch.float64)
        ln = torch.log(Simple_vector)
        Simple_vector = self.lin(ln)

        #计算损失
        loss = self.lossfun(Simple_vector,label_list)

        softmaxed = None
        return Simple_vector,loss,softmaxed
    
@META_ARCH_REGISTRY.register()
def dirichlet_calibration(cfg):
    return Dirichlet_calibration(cfg)

if __name__=="__main__":
    model = Dirichlet_calibration()
    input = torch.tensor([0.91,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01])
    after_calibration = model(input)
    print(after_calibration)