import os
import pickle as pk
import reader
import numpy as np
import sys

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from generate_frames_p2 import extract_frames

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class DATA(Dataset):
    def __init__(self, opt):

        # Call frame generator function
        self.frames_data = extract_frames(opt)
        
        # Transform the image
        self.transform = transforms.Compose([
            transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
            transforms.Normalize(MEAN, STD)
        ])
        

    def __getitem__(self, idx):
        
        frames_list = self.frames_data[idx] # contains sampled frames for one video, in a list

        N = len(frames_list)    # number of frames in this particular video
        frames_tensor = torch.zeros(N, 3, 240, 320) # tensor of dimension NxCx240x320, which will contain all N frames for one video

        # Transform each frame (currently numpy array) in the list into a tensor, and put it into the pre-allocated tensor
        for i in range(N):
            frames_tensor[i,:,:,:] = self.transform(frames_list[i]) # each frame is now a tensor, Cx240x320            

        return frames_tensor


    def __len__(self):
        return len(self.frames_data)    # number of videos that were sampled

