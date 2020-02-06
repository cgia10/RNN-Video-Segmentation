import os
import pickle as pk
import reader
import numpy as np
import sys

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from generate_features_p2 import extract_features


class DATA(Dataset):
    def __init__(self, opt):
    
        self.features_data = extract_features(opt)
        

    def __getitem__(self, idx):

        feature = self.features_data[idx] # contains feature vectors for all frames of one video, as a tensor (N, 2048)

        return feature


    def __len__(self):
        return len(self.features_data)  # number of videos loaded

