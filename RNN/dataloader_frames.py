import os
import pickle as pk
import reader
import numpy as np
import sys

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

class DATA(Dataset):
    def __init__(self, opt, mode):

        # Set directory where to load separated video frames, and how many videos to load
        if mode == "train":
            load_dir = opt.load_train_frames_dir
        elif mode == "val":
            load_dir = opt.load_val_frames_dir
        else:
            print("ERROR: invalid mode in data handler")
            sys.exit()
        
        filenames = sorted(os.listdir(load_dir))

        print("Loading frames for {} videos from {}".format(len(filenames), load_dir))

        # Iterate through directory and append the contents of each loaded file into 2 lists: one for frames, one for labels
        self.frames_data = []
        self.label_data = []
        
        for i in range(len(filenames)):
            
            # Load video dict
            print("Loading frames from video %d..." % (i+1))

            with open(os.path.join(load_dir, filenames[i]), "rb") as f:
                data_dict = pk.load(f)
            
            # Append to list
            self.frames_data.append(data_dict["frame_list"])
            self.label_data.append(data_dict["label"])
        
        # Transform the image
        self.transform = transforms.Compose([
            transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
            transforms.Normalize(MEAN, STD)
        ])
        

    def __getitem__(self, idx):
        
        frames_list = self.frames_data[idx] # contains sampled frames for one video, in a list
        label = self.label_data[idx]

        N = len(frames_list)    # number of frames in this particular video
        frames_tensor = torch.zeros(N, 3, 240, 320) # tensor of dimension NxCx240x320, which will contain all N frames for one video

        # Transform each frame (currently numpy array) in the list into a tensor, and put it into the pre-allocated tensor
        for i in range(N):
            frames_tensor[i,:,:,:] = self.transform(frames_list[i]) # each frame is now a tensor, Cx240x320            

        return frames_tensor, int(label)


    def __len__(self):
        return len(self.frames_data)    # number of videos that were sampled

