import json
import logging
import torch
import customKing.utils.comm as comm
from customKing.engine import default_writers
from customKing.solver.build import build_Calibration_optimizer_for_all,build_Calibration_lr_scheduler
from customKing.modeling.meta_arch.build import build_Calibration_Evaluate_model
from customKing.utils.events import EventStorage
import time
import torch.nn.functional as F
import traceback
torch.autograd.set_detect_anomaly(True)
from torch.utils.data import Dataset,DataLoader

class Load_z(Dataset):
    def __init__(self,cfg,data = "valid") -> None:
        super().__init__()
        self.z_list = []
        self.label_list = []
        if data == "valid":
            Path = cfg.CALIBRATION_DATASET.VALID_PATH
        elif data == "test":
            Path = cfg.CALIBRATION_DATASET.TEST_PATH

        with open(Path,"r",encoding="utf-8") as f:
            for line in f:
                output = json.loads(line)
                self.z_list.append(output[0])
                self.label_list.append(output[1])
        self.z_list = torch.tensor(self.z_list,dtype=torch.float64)
        self.label_list = torch.tensor(self.label_list,dtype=torch.float64)

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        return self.z_list[index],self.label_list[index]

class Load_p_hat(Dataset):
    def __init__(self,p_hat,label) -> None:
        super().__init__()
        self.p_hat = p_hat
        self.label = label

    def __len__(self):
        len(self.label)        

    def __getitem__(self, index):
        return self.p_hat[index],self.label[index]

class doTrain():
    def __init__(self,cfg):
        self.iteration = cfg.CALIBRATION_SOLVER.START_ITER
        self.cfg = cfg
        if self.cfg.CALIBRATION_DATASET.LOAD_METHOD == True:
            self.trainDataset = Load_z(cfg,"valid")
            self.testDataset = Load_z(cfg,"test")
            self.trainDataLoader = DataLoader(self.trainDataset,batch_size=cfg.CALIBRATION_SOLVER.BATCHSIZE,shuffle=True,num_workers=cfg.CALIBRATION_SOLVER.NUM_WORKS)
            self.testDataLoader = DataLoader(self.testDataset,batch_size=cfg.CALIBRATION_SOLVER.BATCHSIZE,shuffle=True,num_workers=cfg.CALIBRATION_SOLVER.NUM_WORKS)
        else:
            self.trainDataset = Load_z(cfg,"valid")
            self.testDataset = Load_z(cfg,"test")
            self.trainDataLoader = (self.trainDataset.z_list,self.trainDataset.label_list)
            self.testDataLoader = (self.testDataset.z_list,self.testDataset.label_list)

        self.best_ece_dict = {}
        self.best_epoch_dict = {}
        self.epoch = -1
        self.model = None
        self.stage = None

    def AfterCalibrationOutput(self,p_hat_list_tensor,label_list,index,data_mode = "Testdata"):
        p_hat_list_tensor = F.softmax(p_hat_list_tensor,dim=1)
        for i in range(len(p_hat_list_tensor)):
            with open(self.cfg.CALIBRATION_MODEL.OUTPUT_DIRS[index]+data_mode+"_After_calibration.json","a+",encoding="UTF-8") as fp:
                    fp.write(json.dumps([p_hat_list_tensor[i].tolist(),label_list[i].item()])+"\n")

    def compute_ece(self,Dataset,index,dataset = "Test",softmaxed = None):
        ece_dict = {}
        ece_dict["epoch"] = self.epoch

        for ece_method in self.cfg.CALIBRATION_EVALUATE.METHOD_list:
            ece =  build_Calibration_Evaluate_model(ece_method,self.cfg,Dataset,softmaxed).compute_ECE()
            ece_dict[ece_method] = ece

        if dataset == "Test":
            if self.model.stage_num > 1:
                with open(self.cfg.CALIBRATION_MODEL.OUTPUT_DIRS[index] + self.model.stage_name[self.stage] + "_Test_metirc.json","a+",encoding="UTF-8") as fp:
                    fp.write(json.dumps(ece_dict)+"\n")
            else:
                with open(self.cfg.CALIBRATION_MODEL.OUTPUT_DIRS[index] + "Test_metirc.json","a+",encoding="UTF-8") as fp:
                    fp.write(json.dumps(ece_dict)+"\n")
        elif dataset == "Valid":
            if self.model.stage_num > 1:
                with open(self.cfg.CALIBRATION_MODEL.OUTPUT_DIRS[index] + self.model.stage_name[self.stage] + "_Valid_metirc.json","a+",encoding="UTF-8") as fp:
                    fp.write(json.dumps(ece_dict)+"\n")
            else:
                with open(self.cfg.CALIBRATION_MODEL.OUTPUT_DIRS[index] + "Valid_metirc.json","a+",encoding="UTF-8") as fp:
                    fp.write(json.dumps(ece_dict)+"\n")

        #record best ece and best epoch
        if dataset == "Test" and self.epoch >= 0:
            if self.epoch == 0:
                for key in ece_dict.keys():
                    if key != "epoch":
                        self.best_ece_dict[key] = ece_dict[key]
                        self.best_epoch_dict[key] = 0
            elif self.epoch > 0:
                for key in ece_dict.keys():
                    if key != "epoch":
                        if ece_dict[key] < self.best_ece_dict[key]:
                            self.best_ece_dict[key] = ece_dict[key]
                            self.best_epoch_dict[key] = self.epoch
            if self.epoch == self.cfg.CALIBRATION_SOLVER.MAX_EPOCH-1:
                if self.model.stage_num > 1:
                    with open(self.cfg.CALIBRATION_MODEL.OUTPUT_DIRS[index] + self.model.stage_name[self.stage] + "_Test_metirc.json","a+",encoding="UTF-8") as fp:
                        fp.write(json.dumps(self.best_epoch_dict)+"\n")
                        fp.write(json.dumps(self.best_ece_dict)+"\n")
                else:
                    with open(self.cfg.CALIBRATION_MODEL.OUTPUT_DIRS[index] + "Test_metirc.json","a+",encoding="UTF-8") as fp:
                        fp.write(json.dumps(self.best_epoch_dict)+"\n")
                        fp.write(json.dumps(self.best_ece_dict)+"\n")

    def do_train(self,model,index):
        logging.basicConfig(level=logging.INFO)
        self.model = model
        model.train() 

        for stage in range(model.stage_num):
            model.stage = stage
            self.stage = stage

            if model.stage == 1:
                if hasattr(model, "coarse_scaling_vector"):
                    model.coarse_scaling_vector.requires_grad_(False)

            optimizer = build_Calibration_optimizer_for_all(self.cfg, model,index)
            scheduler = build_Calibration_lr_scheduler(self.cfg, optimizer) 
            if model.stage_num > 1:
                print(model.stage_name[stage]+" training")
                writers = default_writers(self.cfg.CALIBRATION_MODEL.OUTPUT_DIRS[index]+model.stage_name[stage]+"_logging.json", self.cfg.CALIBRATION_SOLVER.MAX_ITER) if comm.is_main_process() else []
            else:
                writers = default_writers(self.cfg.CALIBRATION_MODEL.OUTPUT_DIRS[index]+"logging.json", self.cfg.CALIBRATION_SOLVER.MAX_ITER) if comm.is_main_process() else []

            with EventStorage(self.cfg.CALIBRATION_SOLVER.START_ITER) as storage:
                #-----------Compute and log uncalibrated ECE-----------#
                self.epoch = -1
                self.compute_ece(self.testDataset,index,dataset="Test")

                for epoch in range(self.cfg.CALIBRATION_SOLVER.MAX_EPOCH):
                    time1 = time.time()
                    self.epoch = epoch
                    storage.iter = epoch
                    all_p_hat_list_tensor = []
                    all_label = []
                    if self.cfg.CALIBRATION_DATASET.LOAD_METHOD == True:
                        for z,label in self.trainDataLoader:
                            z = z.cuda()
                            label = label.cuda().long()
                            p_hat_list_tensor,loss_value,softmaxed = model(z,label)
                            optimizer.zero_grad()    
                            loss_value.backward()    
                            optimizer.step()   
                            all_p_hat_list_tensor.append(p_hat_list_tensor)
                            all_label.append(label)
                    else:
                        z,all_label = self.trainDataLoader
                        z = z.cuda()
                        all_label = all_label.cuda().long()
                        all_p_hat_list_tensor,loss_value,softmaxed = model(z,all_label)
                        optimizer.zero_grad()    
                        loss_value.backward()    
                        optimizer.step()

                    #----------Record the calibrated ECE--------#
                    all_test_p_hat_list_tensor = []
                    all_test_label = []
                    if self.cfg.CALIBRATION_DATASET.LOAD_METHOD == True:
                        for z,label in self.testDataLoader:
                            z = z.cuda()
                            label = label.cuda().long()
                            test_p_hat_list_tensor,_,softmaxed = model(z,label)
                            all_test_p_hat_list_tensor.append(test_p_hat_list_tensor)
                            all_test_label.append(label)
                        all_test_p_hat_list_tensor = torch.cat(all_test_p_hat_list_tensor,dim=0)
                        all_test_label = torch.cat(all_test_label,dim=0)
                    else:
                        z,all_test_label = self.testDataLoader
                        z = z.cuda()
                        all_test_label = all_test_label.cuda().long()
                        all_test_p_hat_list_tensor,_,softmaxed = model(z,all_test_label)
                    
                    if self.cfg.CALIBRATION_DATASET.LOAD_METHOD == True:
                        test_p_hat_Dataset = Load_p_hat(all_test_p_hat_list_tensor,all_test_label)
                    else:
                        test_p_hat_Dataset = (all_test_p_hat_list_tensor,all_test_label)
                    self.compute_ece(test_p_hat_Dataset,index,dataset="Test",softmaxed=softmaxed)

                    #-----Record the accuracy in test dataset-----#
                    _,test_predicted = torch.max(all_test_p_hat_list_tensor.data,1)
                    correct = test_predicted.eq(all_test_label).cpu().sum()
                    test_acc = correct / all_test_p_hat_list_tensor.shape[0]

                    storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
                    scheduler.step()
                    storage.put_scalar("loss", loss_value, smoothing_hint=False)
                    storage.put_scalar("train_time",time.time()-time1,smoothing_hint=False)
                    storage.put_scalar("test_acc",test_acc,smoothing_hint=False)
                    for writer in writers:
                        writer.write()

        #save calibration model
        torch.save(model.state_dict(), self.cfg.CALIBRATION_MODEL.OUTPUT_DIRS[index]+self.cfg.CALIBRATION_MODEL.META_ARCHITECTURES[index]+".pth")

        #Record output after calibration
        self.AfterCalibrationOutput(all_test_p_hat_list_tensor,all_test_label,index)
