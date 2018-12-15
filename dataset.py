# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pandas
import numpy
import json
import torch.utils.data as data
import os
import torch

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
        print("%s subset video numbers: %d" %(self.subset,len(self.list)))

    def __getitem__(self, index):
        video_data,anchor_xmin,anchor_xmax,start_frame,video_name = self._get_base_data(index)
        if self.mode == "train":
            match_score_action,match_score_start,match_score_end =  self._get_train_label(\
                index,anchor_xmin,anchor_xmax,start_frame,video_name)
            return video_data,match_score_action,match_score_start,match_score_end
        else:
            return index,video_data,anchor_xmin,anchor_xmax
        
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
        video_df=pd.read_csv(os.path.join(self.feature_path,csv_name))
        video_data = video_df.values[:,:]
        video_data = torch.Tensor(video_data)
        video_data = torch.transpose(video_data,0,1)
        video_data.float()
        return video_data,anchor_xmin,anchor_xmax,start_frame,video_name
    
    def _get_train_label(self,index,anchor_xmin,anchor_xmax,start_frame,video_name): 
        # 将时间戳归一化为0-1之间的数
        # 获得一个训练样本的标注
        video_info=self.video_dict[video_name]
    
        gt_bbox = []
        for j in range(len(video_info)):
            tmp_info=video_info[j]
            # 一个滑窗中共有1600帧
            tmp_start=max(min(1,(tmp_info[0]*self.fps-start_frame)/float(self.temporal_scale*16)),0)
            tmp_end=max(min(1,(tmp_info[1]*self.fps-start_frame)/float(self.temporal_scale*16)),0)
            # 排除gt完全不在该样本滑窗范围内的情况
            if (tmp_start==0 and tmp_end==0) or (tmp_start==1 and tmp_end==1):
                continue 
            # if (tmp_start==0 and tmp_end==1):
                # print("\033[31m [INFO] start frame:{}\t gt_start:{}\t \
                        # gt_end{}\t\033[0m".format(start_frame, tmp_info[0]*self.fps, tmp_info[1]*self.fps)) 
            gt_bbox.append([tmp_start,tmp_end])
            
        if (len(gt_bbox)==0):
            print("\033[31m [WARINING] {} {} do not have gt\033[0m".format(video_name,start_frame))
        gt_bbox=np.array(gt_bbox)
        # print("\033[33m [DEBUG]\033[0m{}".format(gt_bbox))
        gt_xmins=gt_bbox[:,0]
        gt_xmaxs=gt_bbox[:,1]

        gt_lens=gt_xmaxs-gt_xmins
        gt_len_small=np.maximum(self.temporal_gap,self.boundary_ratio*gt_lens)
        gt_start_bboxs=np.stack((gt_xmins-gt_len_small/2,gt_xmins+gt_len_small/2),axis=1)
        gt_end_bboxs=np.stack((gt_xmaxs-gt_len_small/2,gt_xmaxs+gt_len_small/2),axis=1)
        
        match_score_action=[]
        for jdx in range(len(anchor_xmin)):
            # 对每一个anchor计算其与所有gt的ioa
            # 并取其中的最大值
            match_score_action.append(np.max(self._ioa_with_anchors(anchor_xmin[jdx],anchor_xmax[jdx],gt_xmins,gt_xmaxs)))
        match_score_start=[]
        for jdx in range(len(anchor_xmin)):
            match_score_start.append(np.max(self._ioa_with_anchors(anchor_xmin[jdx],anchor_xmax[jdx],gt_start_bboxs[:,0],gt_start_bboxs[:,1])))
        match_score_end=[]
        for jdx in range(len(anchor_xmin)):
            match_score_end.append(np.max(self._ioa_with_anchors(anchor_xmin[jdx],anchor_xmax[jdx],gt_end_bboxs[:,0],gt_end_bboxs[:,1])))
        match_score_action = torch.Tensor(match_score_action)
        match_score_start = torch.Tensor(match_score_start)
        match_score_end = torch.Tensor(match_score_end)
        return match_score_action,match_score_start,match_score_end

    def _ioa_with_anchors(self,anchors_min,anchors_max,box_min,box_max):
        len_anchors=anchors_max-anchors_min
        int_xmin = np.maximum(anchors_min, box_min)
        int_xmax = np.minimum(anchors_max, box_max)
        inter_len = np.maximum(int_xmax - int_xmin, 0.)
        scores = np.divide(inter_len, len_anchors)
        return scores
    
    def __len__(self):
        return len(self.list)


class ProposalDataSet(data.Dataset):
    def __init__(self,opt,subset="train"):
        
        self.subset=subset
        self.mode = opt["mode"]
        if self.mode == "train":
            self.top_K = opt["pem_top_K"]
        else:
            self.top_K = opt["pem_top_K_inference"]
        self.video_info_path = opt["video_info"]
        self.video_anno_path = opt["video_anno"]

        self._getDatasetDict()
        
    def _getDatasetDict(self):
        anno_df = pd.read_csv(self.video_info_path)
        anno_database= load_json(self.video_anno_path)
        self.video_dict = {}
        for i in range(len(anno_df)):
            video_name=anno_df.video.values[i]
            video_info=anno_database[video_name]
            video_subset=anno_df.subset.values[i]
            if self.subset == "full":
                self.video_dict[video_name] = video_info
            if self.subset in video_subset:
                self.video_dict[video_name] = video_info
        self.video_list = self.video_dict.keys()
        print("%s subset video numbers: %d" %(self.subset,len(self.video_list)))

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        video_name = self.video_list[index]
        pdf=pandas.read_csv("./output/PGM_proposals/"+video_name+".csv")
        pdf=pdf[:self.top_K]
        video_feature = numpy.load("./output/PGM_feature/" + video_name+".npy")
        video_feature = video_feature[:self.top_K,:]
        #print len(video_feature),len(pdf)
        video_feature = torch.Tensor(video_feature)

        if self.mode == "train":
            video_match_iou = torch.Tensor(pdf.match_iou.values[:])
            return video_feature,video_match_iou
        else:
            video_xmin =pdf.xmin.values[:]
            video_xmax =pdf.xmax.values[:]
            video_xmin_score = pdf.xmin_score.values[:]
            video_xmax_score = pdf.xmax_score.values[:]
            return video_feature,video_xmin,video_xmax,video_xmin_score,video_xmax_score
        