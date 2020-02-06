import os
import sys
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

class DATA(Dataset):
    def __init__(self, opt):
        
        # Set directory 
        video_dir = opt.val_video_dir           

        # Set video folders
        self.foldernames = sorted(os.listdir(video_dir))

        # Initialize lists of filepaths
        self.frame_paths = []

        print("")
        print("Loading frame filepaths for {} videos from {}".format(len(self.foldernames), video_dir))

        # Iterate through the video folders
        for i, folder in enumerate(self.foldernames):
            
            # Holds path names to all frames, for one video at a time
            framepaths = []
            
            print("")
            print("***** VIDEO %d *****" % (i+1))

            # Load paths to the frames
            frame_dir = os.path.join(video_dir, folder)     
            print("Category: {}".format(folder))

            for frame in sorted(os.listdir(frame_dir)):
                img_path = os.path.join(frame_dir, frame)   
                framepaths.append(img_path)

            print("Found %d frames" % len(framepaths))
            print("")

            self.frame_paths.append(framepaths)     # list of lists. Length equal to number of videos. Each sublist contains the filepaths to the frames of a video

        # Transform
        self.transform_frames = transforms.Compose([
            transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
            transforms.Normalize(MEAN, STD)
        ])


    def __getitem__(self, idx):
        
        # Load paths to frames for a single video
        frame_paths = self.frame_paths[idx]     # paths to the frames for a video

        # Load frames into memory, and convert labels to ints
        frames_list = []

        for path in frame_paths:
            
            # Load image, transform to tensor and reshape
            img = Image.open(path).convert("RGB")
            img = self.transform_frames(img)                            # tensor (3, 240, 320)
            img = img.view(1, img.shape[0], img.shape[1], img.shape[2]) # tensor (1, 3, 240, 320)

            # Append to temporary lists
            frames_list.append(img)                     # list of tensors, length = number of frames in current video

        # Convert to tensor
        frames_tensor = torch.cat(frames_list, dim=0)               # tensor (N, 3, 240, 320). All N frames for one video

        return frames_tensor, self.foldernames


    def __len__(self):
        return len(self.frame_paths)    # number of videos that were sampled

