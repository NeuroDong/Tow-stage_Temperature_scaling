'''
refer to the paper "Measuring Calibration in Deep learning"
The file is used for computing TACE (Thresholded Adaptive Calibration Error)
'''
import torch
from torch.nn.functional import softmax
from ..build import META_ARCH_REGISTRY
import copy
from torch.utils.data import Dataset

class TACE_ECE():
    def __init__(self,cfg,Dataset,softmaxed = None):
        self.threshold = 1/cfg.CALIBRATION_MODEL.NUM_CLASS
        self.Dataset = Dataset
        self.label_list = []
        self.softmaxed = softmaxed
        self.class_num = cfg.CALIBRATION_MODEL.NUM_CLASS

        self.bin_boundaries_list = []
        self.bin_lowers_list = []
        self.bin_uppers_list = []

        self.Thresholded_classwise_confidence_lists, self.Thresholded_label_list= self.get_classwise_confidence_and_predict()
        for n_bin in cfg.CALIBRATION_EVALUATE.SAMPLE_NUM:
            if len(self.Thresholded_classwise_confidence_lists[0]) < n_bin:
                n_bin = len(self.Thresholded_classwise_confidence_lists[0])/2

            class_bin_lowers = []
            class_bin_uppers = []
            for j in range(len(self.Thresholded_classwise_confidence_lists)):   
                sorted_id = sorted(range(len(self.Thresholded_classwise_confidence_lists[j])), key=lambda k: self.Thresholded_classwise_confidence_lists[j][k], reverse=True)
                confidence_list = [self.Thresholded_classwise_confidence_lists[j][id] for id in sorted_id]
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
        classwise_confidence_list = [[] for i in range(self.class_num)]
        if isinstance(self.Dataset,Dataset):
            for z,label in self.Dataset:
                if self.softmaxed == None:
                    z = softmax(z,dim=0,dtype=torch.float64)
                z = z.tolist()
                label = label.item()
                self.label_list.append(label)
                for j in range(self.class_num):
                    classwise_confidence_list[j].append(z[j])
        else:
            z,label = self.Dataset
            if self.softmaxed == None:
                z = softmax(z,dim=1,dtype=torch.float64)
            z = z.tolist()
            self.label_list = label.tolist()
            for sample in z:
                for j in range(self.class_num):
                    classwise_confidence_list[j].append(sample[j])

        Thresholded_classwise_confidence_lists = [[]for i in range(self.class_num)]
        Thresholded_label_list = [[]for i in range(self.class_num)]
        for j in range(self.class_num):
            for i in range (len(classwise_confidence_list[j])):
                if classwise_confidence_list[j][i] >= self.threshold:
                    Thresholded_classwise_confidence_lists[j].append(classwise_confidence_list[j][i])
                    Thresholded_label_list[j].append(self.label_list[i])

        Thresholded_classwise_confidence_lists = list(filter(lambda x: x != [], Thresholded_classwise_confidence_lists))
        Thresholded_label_list = list(filter(lambda x: x != [], Thresholded_label_list))
        return Thresholded_classwise_confidence_lists,Thresholded_label_list

    def compute_prop_confidence_acc_in_bin(self):
        acc_lists = []
        for j in range(len(self.Thresholded_classwise_confidence_lists)):
            acc_list = []
            for i in range(len(self.Thresholded_label_list[j])):
                if self.Thresholded_label_list[j][i]==j:
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
                    in_bin = [(self.Thresholded_classwise_confidence_lists[k][i] > bin_lower)*(self.Thresholded_classwise_confidence_lists[k][i] <= bin_upper) for i in range(len(self.Thresholded_classwise_confidence_lists[k]))]
                    prop_in_bin = sum(in_bin)/len(in_bin)   
                    if prop_in_bin > 0:
                        acc_in_bin = 0.
                        confidence_in_bin = 0.
                        for i in range(len(in_bin)):
                            if in_bin[i] == 1:
                                acc_in_bin = acc_in_bin + acc_lists[k][i]
                                confidence_in_bin = confidence_in_bin + self.Thresholded_classwise_confidence_lists[k][i]
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
def tace_ece(cfg,Dataset,softmaxed):
    return TACE_ECE(cfg,Dataset,softmaxed=softmaxed)


