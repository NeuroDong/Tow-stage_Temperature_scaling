from .config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# ----------------------------------------------------------------------------- #
# Config definition
# ----------------------------------------------------------------------------- #

_C = CN()
# ---------------------------------------------------------------------------- #
# Classification model config
# ---------------------------------------------------------------------------- #
_C.DATASETS = CN()
_C.DATASETS.TRAIN = "Cifar10_train"    #train dataset
_C.DATASETS.VALID = "Cifar10_valid"    #valid dataset
_C.DATASETS.TEST = "Cifar10_test"      #test dataset

_C.MODEL = CN()
_C.MODEL.META_ARCHITECTURE = "densenet_k12_D40"  #select classification model
_C.MODEL.OUTPUT_NUM_ClASSES = 10    #set class num
_C.MODEL.INPUT_IMAGESIZE = (32,32)    #set image size
_C.MODEL.DEVICE = "cuda:0"     #select device
_C.MODEL.JUST_EVAL = False    
_C.MODEL.PRE_WEIGHT = False    
_C.MODEL.OUTPUT_DIR = "output/Calibration/"+_C.DATASETS.TRAIN[:-6]+"/"+_C.MODEL.META_ARCHITECTURE+"/metric.json"    #Path to save training log files and network weights files
_C.MODEL.PREWEIGHT = r" "    #The path of saving the pretrain weight 

_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER = "SGD"    #select optimizer，see：customKing\solver\build.py
_C.SOLVER.BATCH_SIZE = 128    #Set batch_size
_C.SOLVER.SHUFFLE = True    
_C.SOLVER.NUM_WORKERS = 8    #the num workers of the Dataloader
_C.SOLVER.IS_PARALLEL = False   #Whether to use multiple GPUs for training

_C.SOLVER.LR_SCHEDULER_NAME = "Step_Decay"     #select lr_scheduler，see：customKing\solver\build.py
_C.SOLVER.START_ITER = 0    
_C.SOLVER.MAX_EPOCH = 200    
_C.SOLVER.MAX_ITER = 64000    
_C.SOLVER.BASE_LR = 0.1    
_C.SOLVER.MOMENTUM = 0.9  
_C.SOLVER.NESTEROV = False  
_C.SOLVER.WEIGHT_DECAY = 0.0001     
_C.SOLVER.GAMMA = 0.1    #if using Step_Decay，the lr after decay is BASE_LR * GAMMA
# The iteration number to decrease learning rate by GAMMA.
_C.SOLVER.STEPS = (32000,48000)    #Set the decay step size, which must be smaller than the training MAX_ITER
_C.SOLVER.CLR_STEPS = 2000     #if using CLR lr_scheduler, the config need to set.
_C.SOLVER.WARMUP_FACTOR = 1.0 / 1000
_C.SOLVER.WARMUP_ITERS = 1000
_C.SOLVER.WARMUP_METHOD = "linear"

# ---------------------------------------------------------------------------- #
# Calibration model config
# ---------------------------------------------------------------------------- #
_C.CALIBRATION_MODEL = CN()
_C.CALIBRATION_MODEL.META_ARCHITECTURES = ["top_label_temperature_scale"]
_C.CALIBRATION_MODEL.NUM_CLASS = 10
_C.CALIBRATION_MODEL.DEVICE = "cuda"
_C.CALIBRATION_MODEL.JUST_EVAL = False
_C.CALIBRATION_MODEL.PRE_WEIGHT = False
_C.CALIBRATION_MODEL.PREWEIGHT = "output/Calibration/Cifar10_SEED20/Resnet20/36608.pth"

_C.CALIBRATION_SOLVER = CN()
_C.CALIBRATION_SOLVER.START_ITER = 0
_C.CALIBRATION_SOLVER.OPTIMIZER = "SGD"
_C.CALIBRATION_SOLVER.LR_SCHEDULER_NAME = "Step_Decay"
_C.CALIBRATION_SOLVER.BASE_LRS = [1.0]    #Corresponds to the method in _C.CALIBRATION_MODEL.META_ARCHITECTURES
_C.CALIBRATION_SOLVER.MOMENTUM = 0.9      
_C.CALIBRATION_SOLVER.WEIGHT_DECAY = 0.0001 
_C.CALIBRATION_SOLVER.NESTEROV = False  
_C.CALIBRATION_SOLVER.GAMMA = 0.1 
_C.CALIBRATION_SOLVER.STEPS = (500,) 
_C.CALIBRATION_SOLVER.CLR_STEPS = 2000     #If the CLR learning rate schedule is used, this parameter needs to be set
_C.CALIBRATION_SOLVER.WARMUP_FACTOR = 1.0 / 1000
_C.CALIBRATION_SOLVER.WARMUP_ITERS = 1000
_C.CALIBRATION_SOLVER.WARMUP_METHOD = "linear"
_C.CALIBRATION_SOLVER.MAX_ITER = 64000
_C.CALIBRATION_SOLVER.MAX_EPOCH = 1000
_C.CALIBRATION_SOLVER.LOSS_FUN = "CrossEntropy"    #see customKing\solver\build.py
_C.CALIBRATION_SOLVER.BATCHSIZE = 1000
_C.CALIBRATION_SOLVER.NUM_WORKS = 20

_C.CALIBRATION_DATASET = CN()
_C.CALIBRATION_DATASET.VALID_PATH = r"output/Calibration/Cifar10/densenet_k12_D40/Validdata_before_calibration.json"
_C.CALIBRATION_DATASET.TEST_PATH = r"output/Calibration/Cifar10/densenet_k12_D40/Testdata_before_calibration.json"
_C.CALIBRATION_DATASET.LOAD_METHOD = False      #Whether to use pytorch's Dataset class to load data

_C.CALIBRATION_EVALUATE = CN()
_C.CALIBRATION_EVALUATE.METHOD_list = ["confidence_ece_with_equal_interval",
                                    "confidence_ece_with_equal_sample",
                                    "classwise_ece_with_equal_interval",
                                    "classwise_ece_with_equal_sample",
                                    "tace_ece",
                                    "top_label_ece_with_equal_interval",
                                    "top_label_ece_with_equal_sample",
                                    ]
_C.CALIBRATION_EVALUATE.INTERVAL_NUM = [15]    #The number of bins divided into bins at equal intervals.
_C.CALIBRATION_EVALUATE.SAMPLE_NUM = [150]    #The number of bins divided into bins at equal samples.

file_name = _C.CALIBRATION_DATASET.TEST_PATH.split("/")
_C.CALIBRATION_MODEL.OUTPUT_DIRS = [_C.CALIBRATION_DATASET.TEST_PATH[:-len(file_name[-1])]+"CalibrationTrain/" + method + "/" for method in _C.CALIBRATION_MODEL.META_ARCHITECTURES]
