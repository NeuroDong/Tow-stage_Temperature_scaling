'''
Visualize gaps of different top-labels and components
'''

import matplotlib.pyplot as plt
import numpy as np
import json
import torch
from torch.nn.functional import softmax

class Confidence_ECE():
    def __init__(self,z_list,label_list,dimension = "top",mode = "equal_sample"):
        self.mode = mode
        self.dimension = dimension
        z_list = torch.tensor(z_list)
        label_list = torch.tensor(label_list)
        if isinstance(z_list,torch.Tensor):
            z_list = softmax(z_list,dim=1,dtype=torch.float64)
            self.z_list = z_list.tolist()
        if isinstance(label_list,torch.Tensor):
            self.label_list = label_list.tolist()

        self.bin_boundaries_list = []
        self.bin_lowers_list = []
        self.bin_uppers_list = []
        self.confidence_list,self.predict_label_list= self.get_confidence_and_labels()
        if self.mode == "equal_sample":
            for n_bin in [100]:     #100 samples per bin
                sorted_id = sorted(range(len(self.confidence_list)), key=lambda k: self.confidence_list[k], reverse=True)
                self.confidence_list = [self.confidence_list[id] for id in sorted_id]
                self.predict_label_list = [self.predict_label_list[id] for id in sorted_id]
                label_list = [label_list[id] for id in sorted_id]
                
                bin_boundaries = []
                for i in range(len(self.confidence_list)):
                    if i%n_bin == 0:
                        bin_boundaries.append(self.confidence_list[i])
                bin_boundaries.reverse()
                bin_lowers = bin_boundaries[:-1]
                bin_uppers = bin_boundaries[1:]
                self.bin_boundaries_list.append(bin_boundaries)
                self.bin_lowers_list.append(bin_lowers)
                self.bin_uppers_list.append(bin_uppers)

    def get_confidence_and_labels(self):
        confidence_list = []
        predict_label_list = []
        for i in range(len(self.z_list)):
            if self.dimension == "top":
                confidence_list.append(max(self.z_list[i]))
                predict_label_list.append(self.z_list[i].index(confidence_list[-1]))
            else:
                confidence_list.append(self.z_list[i][self.dimension])
                predict_label_list.append(self.dimension)
        return confidence_list,predict_label_list

    def compute_prop_confidence_acc_in_bin(self):
        acc_list = []
        for i in range(len(self.label_list)):
            if self.label_list[i]==self.predict_label_list[i]:
                acc_list.append(1)
            else:
                acc_list.append(0)

        sample_ece_list = [None] * len(self.label_list)
        for j in range(len(self.bin_lowers_list)):
            for bin_lower, bin_upper in zip(self.bin_lowers_list[j], self.bin_uppers_list[j]):
                in_bin = [(self.confidence_list[i] > bin_lower)*(self.confidence_list[i] <= bin_upper) for i in range(len(self.confidence_list))]
                prop_in_bin = sum(in_bin)/len(in_bin) 
                if prop_in_bin > 0:
                    acc_in_bin = 0.
                    confidence_in_bin = 0.
                    for i in range(len(in_bin)):
                        if in_bin[i] == 1:
                            acc_in_bin = acc_in_bin + acc_list[i]
                            confidence_in_bin = confidence_in_bin + self.confidence_list[i]
                    acc_in_bin = acc_in_bin/sum(in_bin)

                    for i in range(len(in_bin)):
                        if in_bin[i] == 1:
                            sample_ece_list[i] = abs(self.confidence_list[i]-acc_in_bin)
        ece_list = []
        for sample in sample_ece_list:
            if sample != None:
                ece_list.append(sample)
        return ece_list


class plot_confidence():
    def __init__(self) -> None:
        self.z_list,self.label_list = self.load_data("test")

    def load_data(self,data = "test"):
        z_list = []
        label_list = []
        if data == "valid":
            Path = "output/Calibration/Cifar10_SEED20/Resnet20/Validdata_before_calibration.json"
        elif data == "test":
            Path = "output/Calibration/Cifar10/Resnet20/CalibrationTrain/temperature_scale/Testdata_After_calibration.json"

        with open(Path,"r",encoding="utf-8") as f:
            for line in f:
                output = json.loads(line)
                z_list.append(output[0])
                label_list.append(output[1])
        return z_list,label_list
    
    def top_label_confidence(self):
        top_label_list = [[],[],[],[],[],[],[],[],[],[]]
        top_label_label = [[],[],[],[],[],[],[],[],[],[]]
        ece_list = [[],[],[],[],[],[],[],[],[],[]]

        for i in range(len(self.z_list)):
            top_label_list[self.z_list[i].index(max(self.z_list[i]))].append(self.z_list[i])
            top_label_label[self.z_list[i].index(max(self.z_list[i]))].append(self.label_list[i])
            
        for j in range(len(top_label_list)):
            ece_model = Confidence_ECE(top_label_list[j],top_label_label[j],dimension="top")
            ece_list[j] = ece_model.compute_prop_confidence_acc_in_bin()
            for k in range(len(ece_list[j])):
                if ece_list[j][k] == None:
                    print(k)
            assert None not in ece_list[j],"Have None"

        boxprops = dict(linestyle='-', linewidth=3, color='red')
        whiskerprops = dict(linestyle='--', linewidth=3, color='green')
        capprops = dict(linestyle='-', linewidth=3, color='blue')
        medianprops = dict(linestyle='-', linewidth=3, color='black')
        flierprops=dict(marker='o', markersize=10)

        plt.tick_params(axis='both', labelsize=30)
        plt.boxplot(ece_list,boxprops=boxprops,whiskerprops=whiskerprops, capprops=capprops,flierprops=flierprops, medianprops=medianprops,patch_artist=True)
        plt.title("Calibration bias after TS",fontsize=40,fontname="Times New Roman")
        plt.xlabel("Top-Lables",fontsize=40,fontname="Times New Roman")
        plt.ylabel("Error",fontsize=40,fontname="Times New Roman")
        plt.show()

    def dimension_confidence(self):
        top_label_list = [[],[],[],[],[],[],[],[],[],[]]
        top_label_label = [[],[],[],[],[],[],[],[],[],[]]
        ece_list = [[],[],[],[],[],[],[],[],[],[]]

        for i in range(len(self.z_list)):
            if self.z_list[i].index(max(self.z_list[i]))==0:
                top_label_list[self.z_list[i].index(max(self.z_list[i]))].append(self.z_list[i])
                top_label_label[self.z_list[i].index(max(self.z_list[i]))].append(self.label_list[i])

        for j in range(len(top_label_list)):
            ece_model = Confidence_ECE(top_label_list[0],top_label_label[0],dimension=j)
            ece_list[j] = ece_model.compute_prop_confidence_acc_in_bin()
            for k in range(len(ece_list[j])):
                if ece_list[j][k] == None:
                    print(k)
            assert None not in ece_list[j],"Have None"
        ece_list = ece_list[1:]

        boxprops = dict(linestyle='-', linewidth=3, color='red')
        whiskerprops = dict(linestyle='--', linewidth=3, color='green')
        capprops = dict(linestyle='-', linewidth=3, color='blue')
        medianprops = dict(linestyle='-', linewidth=3, color='black')
        flierprops=dict(marker='o', markersize=10,linewidth = 5)

        plt.tick_params(axis='both', labelsize=30)
        plt.boxplot(ece_list,boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops,flierprops=flierprops, medianprops=medianprops,patch_artist=True)
        plt.title("Calibration bias after TS under the same Top-Label",fontsize=40,fontname="Times New Roman")
        plt.xlabel("Non-Top-Label components",fontsize=40,fontname="Times New Roman")
        plt.ylabel("Error",fontsize=40,fontname="Times New Roman")
        plt.show()
            

if __name__ == "__main__":
    model = plot_confidence()

    #plot gap for different top-labels
    model.top_label_confidence()

    #plot gap for different components under the same top-lable
    model.dimension_confidence()
