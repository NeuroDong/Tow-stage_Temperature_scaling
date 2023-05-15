import matplotlib.pyplot as plt
import numpy as np
import json
import torch
import os,sys

parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
sys.path.insert(0,parentdir) 

class plot_confidence():
    def __init__(self) -> None:
        self.z_list,self.label_list = self.load_data()
        self.test_z_list,self.test_label_list = self.load_data(data="test")
        self.train_z_list,self.train_label_list = self.load_data(data="train")

    def load_data(self,data = "valid"):
        z_list = []
        label_list = []
        if data == "valid":
            Path = "output\Calibration\Cifar10\Resnet20\Validdata_before_calibration.json"
        elif data == "test":
            Path = "output\Calibration\Cifar10\Resnet20\Testdata_before_calibration.json"
        elif data == "train":
            Path = "output\Calibration\Cifar10\Resnet20\Traindata_before_calibration.json"

        if isinstance(Path,list):
            for path in Path:
                with open(path,"r",encoding="utf-8") as f:
                    for line in f:
                        output = json.loads(line)
                        z_list.append(output[0])
                        label_list.append(output[1])
        else:
            with open(Path,"r",encoding="utf-8") as f:
                for line in f:
                    output = json.loads(line)
                    z_list.append(output[0])
                    label_list.append(output[1])
        return z_list,label_list

    def load_temperature(self):
        model_path = r"output/Calibration/Cifar10/Resnet20/CalibrationTrain/top_label_temperature_scale/top_label_temperature_scale.pth"
        model = torch.load(model_path)
        return model["coarse_scaling_vector"].tolist()

    def plot_temperature_and_overfiting(self):
        '''
        Visualize the relationship between the reciprocal of temperature and the overfit curve
        '''
        self.coarse_scaling_temperature = self.load_temperature()
        reciprocal_temperature = [1/a for a in self.coarse_scaling_temperature] 
        plt.subplot(2,1,1)
        plt.subplots_adjust(hspace=0.6)
        plt.bar([i+1 for i in range(len(reciprocal_temperature))],reciprocal_temperature)
        plt.plot([i+1 for i in range(len(reciprocal_temperature))],reciprocal_temperature,linewidth = 3, linestyle='-',marker='o', color = "r")
        #plt.ylim([0.,1])
        plt.xlim([0.,len(reciprocal_temperature)])
        plt.xlabel("Classes",fontsize=40,fontname="Times New Roman")
        plt.ylabel("Value",fontsize=40,fontname="Times New Roman")
        plt.title("1/temperature",fontsize=40,fontname="Times New Roman")
        plt.tick_params(axis='both', labelsize=30)

        test_acc_list = [[] for i in range(len(reciprocal_temperature))]
        for i in range(len(self.test_label_list)):
            if self.test_label_list[i] == self.test_z_list[i].index(max(self.test_z_list[i])):
                test_acc_list[self.test_label_list[i]].append(1)
            else:
                test_acc_list[self.test_label_list[i]].append(0)

        test_acc = [sum(x)/len(x) for x in test_acc_list]

        train_acc_list = [[] for i in range(len(reciprocal_temperature))]
        for i in range(len(self.train_label_list)):
            if self.train_label_list[i] == self.train_z_list[i].index(max(self.train_z_list[i])):
                train_acc_list[self.train_label_list[i]].append(1)
            else:
                train_acc_list[self.train_label_list[i]].append(0)

        train_acc = [sum(x)/len(x) for x in train_acc_list]
        acc_rate = [test_acc[i]/train_acc[i] for i in range(len(train_acc))]

        plt.subplot(2,1,2)
        plt.bar([i+1 for i in range(len(acc_rate))],acc_rate)
        plt.plot([i+1 for i in range(len(acc_rate))],acc_rate,linewidth = 3, linestyle='-',marker='o', color = "r")
        plt.xlim([0.,len(acc_rate)])
        plt.xlabel("Classes",fontsize=40,fontname="Times New Roman")
        plt.ylabel("Value",fontsize=40,fontname="Times New Roman")
        plt.title("Degree of overfitting (test set acc / train set acc)",fontsize=40,fontname="Times New Roman")
        plt.tick_params(axis='both', labelsize=30)
        plt.show()
        
        gailv = []
        for i in range(len(reciprocal_temperature)-1):
            for j in range(i+1,len(reciprocal_temperature),1):
                if ((reciprocal_temperature[i]) - (reciprocal_temperature[j]))*((acc_rate[i]) - (acc_rate[j])) > 0:
                    gailv.append(1)
                else:
                    gailv.append(0)
        print(sum(gailv)/len(gailv))

if __name__ == "__main__":
    model = plot_confidence()
    #Visualize the relationship between the reciprocal of temperature and the overfit curve
    model.plot_temperature_and_overfiting()
