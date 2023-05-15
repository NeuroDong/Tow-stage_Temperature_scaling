import torch
from customKing.config.config import CfgNode
import math
from torch.nn.functional import softmax,relu
import numpy as np
import torch.nn as nn

__all__=["build_optimizer","build_lr_scheduler"]

# ---------------------------------------------------------------------------- #
# build optimizer
# ---------------------------------------------------------------------------- #
def build_optimizer(cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """
    
    if cfg.SOLVER.OPTIMIZER == "SGD":
        optimizer = _get_SGD_optimizer(cfg.SOLVER,model)
    if cfg.SOLVER.OPTIMIZER == "Adam":
        optimizer = _get_adam_optimizer(cfg.SOLVER,model)
    
    return optimizer

def build_Calibration_optimizer(cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """
    if cfg.CALIBRATION_SOLVER.OPTIMIZER == "SGD":
        optimizer = _get_SGD_optimizer(cfg.CALIBRATION_SOLVER,model)
    if cfg.CALIBRATION_SOLVER.OPTIMIZER == "Adam":
        optimizer = _get_adam_optimizer(cfg.CALIBRATION_SOLVER,model)
    return optimizer

def _get_SGD_optimizer(Solver,model:torch.nn.Module):
    return torch.optim.SGD(model.parameters(),lr=Solver.BASE_LR,momentum=Solver.MOMENTUM,weight_decay=Solver.WEIGHT_DECAY,nesterov=Solver.NESTEROV)

def _get_adam_optimizer(Solver,model:torch.nn.Module):
    return torch.optim.Adam(model.parameters(),lr=Solver.BASE_LR,weight_decay=Solver.WEIGHT_DECAY)


def build_Calibration_optimizer_for_all(cfg: CfgNode, model: torch.nn.Module,i) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """
    if cfg.CALIBRATION_SOLVER.OPTIMIZER == "SGD":
        optimizer = _get_SGD_optimizer_for_all(cfg.CALIBRATION_SOLVER,model,i)
    if cfg.CALIBRATION_SOLVER.OPTIMIZER == "Adam":
        optimizer = _get_adam_optimizer_for_all(cfg.CALIBRATION_SOLVER,model,i)
    return optimizer

def _get_SGD_optimizer_for_all(Solver,model:torch.nn.Module,i):
    return torch.optim.SGD(model.parameters(),lr=Solver.BASE_LRS[i],momentum=Solver.MOMENTUM,weight_decay=Solver.WEIGHT_DECAY,nesterov=Solver.NESTEROV)

def _get_adam_optimizer_for_all(Solver,model:torch.nn.Module,i):
    return torch.optim.Adam(model.parameters(),lr=Solver.BASE_LRS[i],weight_decay=Solver.WEIGHT_DECAY)


# ---------------------------------------------------------------------------- #
# build lr_scheduler
# ---------------------------------------------------------------------------- #
def build_lr_scheduler(cfg: CfgNode, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build an scheduler from config.
    """
    Scheduler = Schedulers(cfg.SOLVER,optimizer)
    return Scheduler.Scheduler

def build_Calibration_lr_scheduler(cfg: CfgNode, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build an scheduler from config.
    """
    Scheduler = Schedulers(cfg.CALIBRATION_SOLVER,optimizer)
    return Scheduler.Scheduler

class Schedulers():
    def __init__(self,Solver, optimizer: torch.optim.Optimizer) -> None:
        self.Solver = Solver
        self.optimizer = optimizer
        
        if Solver.LR_SCHEDULER_NAME == "Step_Decay":
            self.Scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,self.Step_Decay)

        if Solver.LR_SCHEDULER_NAME == "CLR":
            self.Scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,self.CLR)

        if Solver.LR_SCHEDULER_NAME == "Linear_Warmup_With_Linear_Decay":
            self.Scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,self.Linear_Warmup_With_Linear_Decay) 

        if Solver.LR_SCHEDULER_NAME == "SGDR":
            self.Scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,self.SGDR) 

    def Step_Decay(self,iteration):
        '''
        Reference Paper “Deep residual learning for image recognition”
        '''
        if iteration < self.Solver.STEPS[0]:
            lr = self.optimizer.defaults["lr"]
        elif iteration > self.Solver.STEPS[len(self.Solver.STEPS)-1]:
            lr = self.optimizer.defaults["lr"] * (self.Solver.GAMMA ** len(self.Solver.STEPS))
        else:
            Left_index = 0    
            Right_index = len(self.Solver.STEPS)-1    
            while Left_index + 1 < Right_index:                
                mid_index = (Left_index + Right_index)//2
                if iteration > self.Solver.STEPS[mid_index]:
                    Left_index = mid_index
                else:
                    Right_index = mid_index
            lr = self.optimizer.defaults["lr"] * (self.Solver.GAMMA ** Right_index)
        return lr/self.optimizer.defaults["lr"]

    def Linear_Warmup_With_Linear_Decay(self,iteration):
        if iteration <= self.Solver.MAX_ITER*0.5:
            lr = self.optimizer.defaults["lr"]*0.01 + (self.optimizer.defaults["lr"]-self.optimizer.defaults["lr"]*0.01)*iteration/(self.Solver.MAX_ITER*0.5)
            return lr/self.optimizer.defaults["lr"]
        else:
            lr = self.optimizer.defaults["lr"] - (self.optimizer.defaults["lr"]-self.optimizer.defaults["lr"]*0.01)*(iteration-self.Solver.MAX_ITER*0.5)/(self.Solver.MAX_ITER*0.5)
            return lr/self.optimizer.defaults["lr"]

    def CLR(self,iteration):
        '''
        Reference Paper "Cyclical Learning Rates for Training Neural Networks"
        '''
        if (iteration // self.Solver.CLR_STEPS) % 2 == 0:
            lr = self.optimizer.defaults["lr"]*0.01 + ((iteration%self.Solver.CLR_STEPS)/self.Solver.CLR_STEPS)*(self.optimizer.defaults["lr"]-self.optimizer.defaults["lr"]*0.01)
            return lr/self.optimizer.defaults["lr"]
        else:
            lr = self.optimizer.defaults["lr"] - ((iteration%self.Solver.CLR_STEPS)/self.Solver.CLR_STEPS)*(self.optimizer.defaults["lr"]-self.optimizer.defaults["lr"]*0.01)
            return lr/self.optimizer.defaults["lr"]

    def SGDR(self,iteration):
        '''
        Reference Paper "SGDR: Stochastic Gradient Descent with Warm Restarts"
        '''
        T_0 = self.Solver.MAX_ITER*0.5
        T_cur = (iteration % int(self.Solver.MAX_ITER*0.5))+1
        lr = 0.001 * self.optimizer.defaults["lr"] + 0.5*(self.optimizer.defaults["lr"]-0.001*self.optimizer.defaults["lr"])*(1+math.cos(T_cur*math.pi/T_0))
        return lr/self.optimizer.defaults["lr"]

# ---------------------------------------------------------------------------- #
# build loss function
# ---------------------------------------------------------------------------- #
def build_Calibration_lossFun(cfg,model = None):
    if model == None:
        if cfg.CALIBRATION_SOLVER.LOSS_FUN == "CrossEntropy":
            return torch.nn.CrossEntropyLoss()
        if cfg.CALIBRATION_SOLVER.LOSS_FUN == "LeastSquare":
            return LeastSquare_Fun
    else:
        if cfg.CALIBRATION_SOLVER.LOSS_FUN == "Dirichlet_Fun":
            lossmodel = Dirichlet_Fun(cfg,model)
            return lossmodel.lossComput

def LeastSquare_Fun(X,Y):
    Sum = 0.
    for i in range(len(X)):
        Sum = Sum + sum([(X[i][j]-Y[i][j])**2 for j in range(len(X[i]))])
    return Sum/len(X)

class Dirichlet_Fun():
    def __init__(self,cfg,model) -> None:
        self.W = model.lin.weight
        self.b = model.lin.bias
        self.CrossEntropyLoss = torch.nn.CrossEntropyLoss()
        self.cfg = cfg

    def lossComput(self,X,Y):
        loss = self.CrossEntropyLoss(X,Y)
        W_sum = 0.
        k = self.cfg.MODEL.OUTPUT_NUM_ClASSES
        for i in range(k):
            for j in range(k):
                if i != j:
                    W_sum = W_sum + self.W[i][j]**2
        bias_sum = sum([bj**2 for bj in self.b])
        lamda = 0.001
        miu = 0.001
        #loss = loss + lamda/(k*(k-1))*W_sum + miu/k*bias_sum
        loss = loss + k*W_sum + miu*bias_sum
        return loss

