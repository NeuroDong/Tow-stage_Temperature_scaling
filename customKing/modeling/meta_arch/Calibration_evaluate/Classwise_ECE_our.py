'''
The file is used for computing Classwise-ECE
Include equal interval binning and equal sample binning
'''
import torch
from ..build import META_ARCH_REGISTRY
from torch.nn.functional import softmax
import time
from torch.utils.data import Dataset

class Classwise_ECE():
    def __init__(self,cfg,Dataset,mode = "equal_interval", softmaxed = None):
        self.mode = mode
        self.Dataset = Dataset
        self.label_list = []
        self.softmaxed = softmaxed
        self.class_num = cfg.CALIBRATION_MODEL.NUM_CLASS
        self.bin_boundaries_list = []
        self.bin_lowers_list = []
        self.bin_uppers_list = []
        if self.mode == "equal_interval":
            for n_bin in cfg.CALIBRATION_EVALUATE.INTERVAL_NUM:
                class_bin_lowers = []
                class_bin_uppers = []
                for j in range(self.class_num):
                    bin_boundaries = torch.linspace(0, 1, n_bin + 1)
                    bin_lowers = bin_boundaries[:-1]
                    bin_uppers = bin_boundaries[1:]
                    class_bin_lowers.append(bin_lowers.tolist())
                    class_bin_uppers.append(bin_uppers.tolist())
                self.bin_lowers_list.append(class_bin_lowers)
                self.bin_uppers_list.append(class_bin_uppers)
        
        self.classwise_confidence_list= self.get_classwise_confidence_and_predict()  #4s
        
        if self.mode == "equal_sample":   #21s
            for n_bin in cfg.CALIBRATION_EVALUATE.SAMPLE_NUM:
                if len(self.classwise_confidence_list[0]) < n_bin:
                    n_bin = len(self.classwise_confidence_list[0])/2

                # 按列表confidence_list中元素的值进行排序，并返回元素对应索引序列
                class_bin_lowers = []
                class_bin_uppers = []
                for j in range(len(self.classwise_confidence_list)):    #j代表类别索引
                    sorted_id = sorted(range(len(self.classwise_confidence_list[j])), key=lambda k: self.classwise_confidence_list[j][k], reverse=True)
                    confidence_list = [self.classwise_confidence_list[j][id] for id in sorted_id]
                    bin_boundaries = []
                    bin_boundaries = confidence_list[::n_bin] #如果出现bin_boundaries中有很多1.0的情况，是softmax计算时精度不够导致的
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

        return classwise_confidence_list

    def compute_prop_confidence_acc_in_bin(self):
        '''
        等间隔装箱或等样本装箱由self.mode决定的
        '''
        acc_lists = []
        for j in range(len(self.classwise_confidence_list)):   #1.2s
            acc_list = [1 if x == j else 0 for x in self.label_list]
            acc_lists.append(acc_list)

        prop_in_bin_lists = []
        confidence_in_bin_lists = []
        acc_in_bin_lists = []
        for j in range(len(self.bin_lowers_list)):
            class_prop_in_bin_list = []
            class_confidence_in_bin_list = []
            class_acc_in_bin_list = []
            for k in range(len(self.bin_lowers_list[j])):    #k代表类别索引
                prop_in_bin_list = []
                confidence_in_bin_list = []
                acc_in_bin_list = []
                for bin_lower, bin_upper in zip(self.bin_lowers_list[j][k], self.bin_uppers_list[j][k]):
                    in_bin = [(self.classwise_confidence_list[k][i] > bin_lower)*(self.classwise_confidence_list[k][i] <= bin_upper) for i in range(len(self.classwise_confidence_list[k]))]
                    prop_in_bin = sum(in_bin)/len(in_bin)    #计算得到每个箱子的权重
                    if prop_in_bin > 0:
                        acc_in_bins = [acc_lists[k][i] if in_bin[i]==1 else 0 for i in range(len(in_bin))]
                        confidence_in_bins = [self.classwise_confidence_list[k][i] if in_bin[i]==1 else 0 for i in range(len(in_bin))]
                        acc_in_bin = sum(acc_in_bins)/sum(in_bin)    #计算得到每个箱子的准确度
                        confidence_in_bin = sum(confidence_in_bins)/sum(in_bin)
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
        '''
        mode代表计算模式:equal interval代表等间隔装箱,equal sample代表等样本装箱
        '''
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
def classwise_ece_with_equal_interval_our(cfg,Dataset,softmaxed):
    return Classwise_ECE(cfg,Dataset,softmaxed=softmaxed)

@META_ARCH_REGISTRY.register()
def classwise_ece_with_equal_sample_our(cfg,Dataset,softmaxed):
    return Classwise_ECE(cfg,Dataset,mode = "equal_sample",softmaxed=softmaxed)