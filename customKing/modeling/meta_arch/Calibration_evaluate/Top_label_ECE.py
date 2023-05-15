'''
This file is used to calculate Top-label ECE, refer to the paper "Top-label calibration and multiclass-to-binary reductions"
Include equal interval binning and equal sample binning
'''
import torch
from ..build import META_ARCH_REGISTRY
from torch.nn.functional import softmax
from torch.utils.data import Dataset

class Top_label_ECE():
    def __init__(self,cfg,Dataset,mode = "equal_interval",softmaxed = None):
        self.mode = mode
        self.Dataset = Dataset
        self.label_list = []
        self.softmaxed = softmaxed
        self.class_num = cfg.CALIBRATION_MODEL.NUM_CLASS

        self.bin_boundaries_list = []
        self.bin_lowers_list = []
        self.bin_uppers_list = []
        self.classwise_confidence_list,self.predict_label_list,self.real_label_list= self.get_classwise_confidence_and_predict()
        if self.mode == "equal_interval":
            for n_bin in cfg.CALIBRATION_EVALUATE.INTERVAL_NUM:
                class_bin_lowers = []
                class_bin_uppers = []
                for j in range(len(self.classwise_confidence_list)):
                    bin_boundaries = torch.linspace(0, 1, n_bin + 1)
                    bin_lowers = bin_boundaries[:-1]
                    bin_uppers = bin_boundaries[1:]
                    class_bin_lowers.append(bin_lowers.tolist())
                    class_bin_uppers.append(bin_uppers.tolist())
                self.bin_lowers_list.append(class_bin_lowers)
                self.bin_uppers_list.append(class_bin_uppers)
        
        if self.mode == "equal_sample":
            for n_bin in cfg.CALIBRATION_EVALUATE.SAMPLE_NUM:
                if len(self.classwise_confidence_list[0]) < n_bin:
                    n_bin = len(self.classwise_confidence_list[0])/2

                class_bin_lowers = []
                class_bin_uppers = []
                for j in range(len(self.classwise_confidence_list)):   
                    sorted_id = sorted(range(len(self.classwise_confidence_list[j])), key=lambda k: self.classwise_confidence_list[j][k], reverse=True)
                    confidence_list = [self.classwise_confidence_list[j][id] for id in sorted_id]
                    bin_boundaries = []
                    for i in range(len(confidence_list)):
                        if i%n_bin == 0:
                            bin_boundaries.append(confidence_list[i]) 
                    bin_boundaries.reverse()
                    bin_lowers = bin_boundaries[:-1]
                    bin_uppers = bin_boundaries[1:]
                    class_bin_lowers.append(bin_lowers)
                    class_bin_uppers.append(bin_uppers)
                self.bin_lowers_list.append(class_bin_lowers)
                self.bin_uppers_list.append(class_bin_uppers)

    def get_classwise_confidence_and_predict(self):
        confidence_list = []
        predict_label_list = []
        if isinstance(self.Dataset,Dataset):
            for z,label in self.Dataset:
                if self.softmaxed == None:
                    z = softmax(z,dim=0,dtype=torch.float64)
                z = z.tolist()
                label = label.item()
                self.label_list.append(label)
                confidence_list.append(max(z))
                predict_label_list.append(z.index(confidence_list[-1]))
        else:
            z,label = self.Dataset
            if self.softmaxed == None:
                z = softmax(z,dim=1,dtype=torch.float64)
            z = z.tolist()
            self.label_list = label.tolist()
            for sample in z:
                confidence_list.append(max(sample))
                predict_label_list.append(sample.index(confidence_list[-1]))

        max_confidence_list = []     
        clabel = []   
        ylabel = []    
        for j in range(self.class_num):
            max_confidence_list_ = []
            clabel_ =[]
            ylabel_ =[]
            for n,c in enumerate(predict_label_list):  
                if c == j:
                    max_confidence_list_.append(confidence_list[n])
                    clabel_.append(c)
                    ylabel_.append(self.label_list[n])
            max_confidence_list.append(max_confidence_list_)
            clabel.append(clabel_)
            ylabel.append(ylabel_)

        max_confidence_list = list(filter(lambda x: x != [], max_confidence_list))
        clabel = list(filter(lambda x: x != [], clabel))
        ylabel = list(filter(lambda x: x != [], ylabel))

        return max_confidence_list,clabel,ylabel

    def compute_prop_confidence_acc_in_bin(self):
        acc_lists = []
        for j in range(len(self.classwise_confidence_list)):
            acc_list = []
            for i in range(len(self.real_label_list[j])):
                if self.real_label_list[j][i]==self.predict_label_list[j][i]:
                    acc_list.append(1)
                else:
                    acc_list.append(0)
            acc_lists.append(acc_list)

        prop_in_bin_lists = []
        confidence_in_bin_lists = []
        acc_in_bin_lists = []
        for j in range(len(self.bin_lowers_list)):
            class_prop_in_bin_list = []
            class_confidence_in_bin_list = []
            class_acc_in_bin_list = []
            for k in range(len(self.bin_lowers_list[j])):    
                prop_in_bin_list = []
                confidence_in_bin_list = []
                acc_in_bin_list = []
                for bin_lower, bin_upper in zip(self.bin_lowers_list[j][k], self.bin_uppers_list[j][k]):
                    in_bin = [(self.classwise_confidence_list[k][i] > bin_lower)*(self.classwise_confidence_list[k][i] <= bin_upper) for i in range(len(self.classwise_confidence_list[k]))]
                    prop_in_bin = sum(in_bin)/(len(in_bin)+torch.finfo(torch.float64).eps)    
                    if prop_in_bin > 0:
                        acc_in_bin = 0.
                        confidence_in_bin = 0.
                        for i in range(len(in_bin)):
                            if in_bin[i] == 1:
                                acc_in_bin = acc_in_bin + acc_lists[k][i]
                                confidence_in_bin = confidence_in_bin + self.classwise_confidence_list[k][i]
                        acc_in_bin = acc_in_bin/sum(in_bin)   
                        confidence_in_bin = confidence_in_bin/sum(in_bin)
                        prop_in_bin_list.append(prop_in_bin)
                        confidence_in_bin_list.append(confidence_in_bin)
                        acc_in_bin_list.append(acc_in_bin)
                class_prop_in_bin_list.append(prop_in_bin_list)
                class_confidence_in_bin_list.append(confidence_in_bin_list)
                class_acc_in_bin_list.append(acc_in_bin_list)
            prop_in_bin_lists.append(class_prop_in_bin_list)
            confidence_in_bin_lists.append(class_confidence_in_bin_list)
            acc_in_bin_lists.append(class_acc_in_bin_list)
        return prop_in_bin_lists,confidence_in_bin_lists,acc_in_bin_lists

    def compute_ECE(self):
        
        prop_in_bin_lists,confidence_in_bin_lists,acc_in_bin_lists = self.compute_prop_confidence_acc_in_bin()
        ECE_list = []
        for n in range(len(prop_in_bin_lists)):
            self.ECE = 0.
            for j in range(len(prop_in_bin_lists[n])):
                class_ECE = 0.
                for t in range(len(prop_in_bin_lists[n][j])):
                    class_ECE = class_ECE + abs(acc_in_bin_lists[n][j][t]-confidence_in_bin_lists[n][j][t])*prop_in_bin_lists[n][j][t]
                self.ECE = self.ECE + class_ECE
            self.ECE = self.ECE/len(prop_in_bin_lists[n])
            ECE_list.append(self.ECE)
        return ECE_list

@META_ARCH_REGISTRY.register()
def top_label_ece_with_equal_interval(cfg,Dataset,softmaxed):
    return Top_label_ECE(cfg,Dataset,softmaxed=softmaxed)

@META_ARCH_REGISTRY.register()
def top_label_ece_with_equal_sample(cfg,Dataset,softmaxed):
    return Top_label_ECE(cfg,Dataset,mode="equal_sample",softmaxed=softmaxed)