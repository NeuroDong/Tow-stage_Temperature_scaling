import torch
from torch import nn
from ..build import META_ARCH_REGISTRY
from customKing.solver.build import build_Calibration_lossFun


class Top_label_emperature_scale(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.coarse_scaling_vector = nn.Parameter(torch.ones(cfg.CALIBRATION_MODEL.NUM_CLASS,dtype=torch.float64)*1.5,requires_grad=True)
        self.fine_scaling_matrix = nn.Parameter(torch.ones(cfg.CALIBRATION_MODEL.NUM_CLASS,cfg.CALIBRATION_MODEL.NUM_CLASS,dtype=torch.float64)*1.5)
        self.cfg = cfg
        self.lossfun = build_Calibration_lossFun(cfg)
        self.stage_num = 2
        self.stage = None
        self.stage_name = ["Coarse_scaling","Fine_scaling"]

    def forward(self,Simple_vector,label_list):
        assert self.stage != None,"Please set self.stage to Coarse scaling or Fine scaling before use!"
        if self.stage == 0:    #Coarse scaling
            _,index = Simple_vector.max(dim=1)
            self.coarse_scaling_vector.data = self.coarse_scaling_vector.clamp(torch.finfo(torch.float64).eps,torch.finfo(torch.float64).max)
            divisor = self.coarse_scaling_vector[index]
            divisor = divisor.unsqueeze(dim=1).repeat(1,len(self.coarse_scaling_vector))
            Simple_vector = Simple_vector/divisor
            #compute loss
            loss = self.lossfun(Simple_vector,label_list)
            softmaxed = None
            return Simple_vector,loss,softmaxed
        elif self.stage == 1:    #Fine scaling
            self.coarse_scaling_vector.requires_grad = False
            _,index = Simple_vector.max(dim=1)
            #Coarse scaling
            divisor = self.coarse_scaling_vector[index]
            divisor = divisor.unsqueeze(dim=1).repeat(1,len(self.coarse_scaling_vector))
            Simple_vector = Simple_vector/divisor
            #Fine scaling
            R = self.fine_scaling_matrix[index]
            Simple_vector = Simple_vector/R
            #compute loss: CrossEntropy + Toward-one regularization
            loss = self.lossfun(Simple_vector,label_list) + 1 * (torch.abs(self.fine_scaling_matrix-1).sum()/(self.cfg.CALIBRATION_MODEL.NUM_CLASS**2))
            softmaxed = None
            return Simple_vector,loss,softmaxed

    
@META_ARCH_REGISTRY.register()
def top_label_temperature_scale(cfg):
    return Top_label_emperature_scale(cfg)