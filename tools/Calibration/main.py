import torch
import os
import sys

parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
sys.path.insert(0,parentdir) 
from customKing.utils.env import Set_seed
from customKing.config.config import get_cfg
from customKing.modeling.meta_arch.build import build_Calibration_models
from utils import doTrain

def main():
    Set_seed(seed=20)
    cfg = get_cfg()
    models = build_Calibration_models(cfg)

    for i in range(len(models)):
        print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in models[i].parameters())))
        if cfg.CALIBRATION_MODEL.JUST_EVAL:
            pass
        else:
            if cfg.CALIBRATION_MODEL.PRE_WEIGHT:
                model = torch.load(cfg.CALIBRATION_MODEL.PREWEIGHT)
            DoTrain = doTrain(cfg)
            DoTrain.do_train(models[i],i)

if __name__ == "__main__":
    main()