'''
The file is used for computing Confidence ECE
Include equal interval binning and equal sample binning
'''
import torch
from ..build import META_ARCH_REGISTRY
from torch.nn.functional import softmax
from torch.utils.data import Dataset

class Confidence_ECE():
    def __init__(self,cfg,Dataset,mode = "equal_interval",softmaxed = None):
        self.mode = mode
        self.Dataset = Dataset
        self.label_list = []
        self.softmaxed = softmaxed

        self.bin_boundaries_list = []
        self.bin_lowers_list = []
        self.bin_uppers_list = []
        if self.mode == "equal_interval":
            for n_bin in cfg.CALIBRATION_EVALUATE.INTERVAL_NUM:
                bin_boundaries = torch.linspace(0, 1, n_bin + 1)
                bin_lowers = bin_boundaries[:-1]
                bin_uppers = bin_boundaries[1:]
                self.bin_boundaries_list.append(bin_boundaries)
                self.bin_lowers_list.append(bin_lowers.tolist())
                self.bin_uppers_list.append(bin_uppers.tolist())
        self.confidence_list,self.predict_label_list= self.get_confidence_and_labels()
        if self.mode == "equal_sample":
            for n_bin in cfg.CALIBRATION_EVALUATE.SAMPLE_NUM:
                sorted_id = sorted(range(len(self.confidence_list)), key=lambda k: self.confidence_list[k], reverse=True)
                self.confidence_list = [self.confidence_list[id] for id in sorted_id]
                self.predict_label_list = [self.predict_label_list[id] for id in sorted_id]
                
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
        return confidence_list,predict_label_list

    def compute_prop_confidence_acc_in_bin(self):
        acc_list = []
        for i in range(len(self.label_list)):
            if self.label_list[i]==self.predict_label_list[i]:
                acc_list.append(1)
            else:
                acc_list.append(0)

        prop_in_bin_lists = []
        confidence_in_bin_lists = []
        acc_in_bin_lists = []
        for j in range(len(self.bin_lowers_list)):
            prop_in_bin_list = []
            confidence_in_bin_list = []
            acc_in_bin_list = []
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
                    confidence_in_bin = confidence_in_bin/sum(in_bin)
                    prop_in_bin_list.append(prop_in_bin)
                    confidence_in_bin_list.append(confidence_in_bin)
                    acc_in_bin_list.append(acc_in_bin)
            prop_in_bin_lists.append(prop_in_bin_list)
            confidence_in_bin_lists.append(confidence_in_bin_list)
            acc_in_bin_lists.append(acc_in_bin_list)
        return prop_in_bin_lists,confidence_in_bin_lists,acc_in_bin_lists

    def compute_ECE(self):
        
        prop_in_bin_lists,confidence_in_bin_lists,acc_in_bin_lists = self.compute_prop_confidence_acc_in_bin()
        ECE_list = []
        for j in range(len(prop_in_bin_lists)):
            self.ECE = 0.
            for i in range(len(prop_in_bin_lists[j])):
                self.ECE = self.ECE + abs(acc_in_bin_lists[j][i]-confidence_in_bin_lists[j][i])*prop_in_bin_lists[j][i]
            ECE_list.append(self.ECE)
        return ECE_list

@META_ARCH_REGISTRY.register()
def confidence_ece_with_equal_interval(cfg,Dataset,softmaxed):
    return Confidence_ECE(cfg,Dataset,softmaxed = softmaxed)

@META_ARCH_REGISTRY.register()
def confidence_ece_with_equal_sample(cfg,Dataset,softmaxed):
    return Confidence_ECE(cfg,Dataset,mode="equal_sample",softmaxed=softmaxed)