# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pandas
import numpy
import json
import torch.utils.data as data
import os
import torch
import opts

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data

class VideoDataSet(data.Dataset):
    def __init__(self,opt,subset="train"):
        self.temporal_scale = opt["temporal_scale"]
        self.temporal_gap = 1. / self.temporal_scale
        self.subset = subset
        self.mode = opt["mode"]
        self.feature_path = opt["feature_path"]
        self.boundary_ratio = opt["boundary_ratio"]
        # 记录所有训练样本名字和起始帧的文件
        if subset=="train":
            train_file = open(opt["train_list"],'r')
            self.list = train_file.readlines()
        if subset=="validation":
            test_file = open(opt["test_list"],'r')
            self.list = test_file.readlines()
        self.out_list = open("list/true_test_list.txt","w")
        # 帧率
        self.fps = opt["fps"]
        # 所有gt的记录文件
        self.video_anno_path = opt["video_anno"]
        self._getDatasetDict()
        
    def _getDatasetDict(self):
        self.video_dict = {}
        for anno_filename in os.listdir(self.video_anno_path):
            anno_file = open(os.path.join(self.video_anno_path,anno_filename),"r")
            anno_content = anno_file.readlines()
            for anno_record in anno_content:
                anno_record = anno_record.rstrip('\n')
                anno_record = anno_record.split()
                video_name = anno_record[0]
                gt_start_frame = float(anno_record[1])
                gt_end_frame = float(anno_record[2])
                if not video_name in self.video_dict.keys():
                    self.video_dict[video_name] = []
                    self.video_dict[video_name].append([gt_start_frame,gt_end_frame])
                else:
                    self.video_dict[video_name].append([gt_start_frame,gt_end_frame])
        # 返回一个字典，其键为视频名字，值为视频相关信息
        self.video_list = self.video_dict.keys()
        print("%s subset video numbers: %d" %(self.subset,len(self.video_list)))

    def __getitem__(self, index):
        video_data,anchor_xmin,anchor_xmax,start_frame,video_name = self._get_base_data(index)
        if self.mode == "train":
            self._get_train_label(\
                index,anchor_xmin,anchor_xmax,start_frame,video_name)
            return
        else:
            return
        
    def _get_base_data(self,index):
        # 从训练列表中得到csv文件名和起始帧
        csv_name = self.list[index].rstrip('\n')
        record = csv_name.split('.')[0].split('_')
        # 典型文件名 video_validation_000051_1.csv
        video_name = ('_').join(record[0:3])
        start_frame = int(record[-1])
        # 0.00 0.01 ... 0.99
        anchor_xmin=[self.temporal_gap*i for i in range(self.temporal_scale)]
        # 0.01 0.02 ... 1.00
        anchor_xmax=[self.temporal_gap*i for i in range(1,self.temporal_scale+1)]
        video_data = []
        return video_data,anchor_xmin,anchor_xmax,start_frame,video_name
    
    def _get_train_label(self,index,anchor_xmin,anchor_xmax,start_frame,video_name): 
        # 将时间戳归一化为0-1之间的数
        # 获得一个训练样本的标注
        video_info=self.video_dict[video_name]
    
        gt_bbox = []
        for j in range(len(video_info)):
            tmp_info=video_info[j]
            tmp_start=max(min(1,(tmp_info[0]*self.fps-start_frame)/float(self.temporal_scale)),0)
            tmp_end=max(min(1,(tmp_info[1]*self.fps-start_frame)/float(self.temporal_scale)),0)
            # 排除gt完全不在该样本滑窗范围内的情况
            if (tmp_start==0 and tmp_end==0) or (tmp_start==1 and tmp_end==1):
                continue 
            gt_bbox.append([tmp_start,tmp_end])
            
        if (len(gt_bbox)==0):
            print("\033[31m [WARINING] {} {} do not have gt\033[0m".format(video_name,start_frame))
        else:
            self.out_list.write(video_name+'_'+str(start_frame)+'.csv\n')

    def _ioa_with_anchors(self,anchors_min,anchors_max,box_min,box_max):
        len_anchors=anchors_max-anchors_min
        int_xmin = np.maximum(anchors_min, box_min)
        int_xmax = np.minimum(anchors_max, box_max)
        inter_len = np.maximum(int_xmax - int_xmin, 0.)
        scores = np.divide(inter_len, len_anchors)
        return scores
    
    def __len__(self):
        return len(self.list)

if __name__=='__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    dataset = VideoDataSet(opt,subset='validation')
    for i in range(dataset.__len__()):
        dataset.__getitem__(i)


        