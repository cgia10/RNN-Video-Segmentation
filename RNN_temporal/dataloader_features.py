import os
import pickle as pk
import sys

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
            print("ERROR: invalid mode in feature dataloader")
            sys.exit()
        
        # Names of .pk files to load
        filenames = sorted(os.listdir(load_dir))

        # Initialize list of filepaths
        self.filepaths = []
        print("Loading feature filepaths for {} videos from {}".format(len(filenames), load_dir))

        # Load path to each .pk file
        for fname in filenames:
            self.filepaths.append( os.path.join(load_dir, fname) )


    def __getitem__(self, idx):
        
        # Load path to .pk file for a single video
        filepath = self.filepaths[idx]

        # Load video dict into memory
        with open(filepath, "rb") as f:
            data_dict = pk.load(f)
        
        feature = data_dict["features"] # tensor, (N, 2048). Features for one video
        label = data_dict["label"]      # tensor, (1, N)

        return feature, label


    def __len__(self):
        return len(self.filepaths)  # number of videos loaded

