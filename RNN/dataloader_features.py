import os
import pickle as pk
import reader
import numpy as np
import sys

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class DATA(Dataset):
    def __init__(self, opt, mode):

        # Set directory where to load video features, and how many videos to load
        if mode == "train":
            load_dir = opt.load_train_features_dir
        elif mode == "val":
            load_dir = opt.load_val_features_dir
        else:
            print("ERROR: invalid mode in data handler")
            sys.exit()
        
        filenames = sorted(os.listdir(load_dir))

        print("Loading features for {} videos from {}".format(len(filenames), load_dir))

        # Iterate through directory and append the contents of each loaded file into 2 lists: one for frames, one for labels
        self.features_data = []
        self.label_data = []
        
        for i in range(len(filenames)):
            
            # Load video dict
            print("Loading features from video %d..." % (i+1))

            with open(os.path.join(load_dir, filenames[i]), "rb") as f:
                data_dict = pk.load(f)
            
            # Append to list
            self.features_data.append(data_dict["features"])
            self.label_data.append(data_dict["label"])
        

    def __getitem__(self, idx):
        
        feature = self.features_data[idx] # contains feature vector for one video, as a tensor
        label = self.label_data[idx]

        return feature, int(label)


    def __len__(self):
        return len(self.features_data)  # number of videos loaded

