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
    def __init__(self, opt, mode):
        
        # Set directories
        if mode == "train":
            label_dir = opt.train_label_dir         
            video_dir = opt.train_video_dir         
            end_video_index = opt.num_videos_train  # number of folders to look through (one folder is one video)
        elif mode == "val":
            label_dir = opt.val_label_dir
            video_dir = opt.val_video_dir
            end_video_index = opt.num_videos_val
        else:
            print("ERROR: invalid mode in frame dataloader")
            sys.exit()
        
        # Set video folders and .txt file names
        foldernames = sorted(os.listdir(video_dir))
        label_txtfilenames = sorted(os.listdir(label_dir))

        # Initialize lists of filepaths
        self.frame_paths = []
        self.label_paths = []

        print("")
        print("Loading frame filepaths for {} videos from {}".format(end_video_index, video_dir))

        # Iterate through the video folders
        for i, folder in enumerate(foldernames):
            
            # Holds path names to all frames, for one video at a time
            framepaths = []

            # Break if number of videos is reached
            if i == end_video_index:
                break
            
            print("")
            print("***** VIDEO %d *****" % (i+1))

            # Load path to the label .txt file
            labelpath = os.path.join(label_dir, label_txtfilenames[i]) 
            print("Label file: {}".format(label_txtfilenames[i]))

            # Load paths to the frames
            frame_dir = os.path.join(video_dir, folder)     
            print("Frames folder: {}".format(folder))

            for frame in sorted(os.listdir(frame_dir)):
                img_path = os.path.join(frame_dir, frame)   
                framepaths.append(img_path)

            print("Found %d frames" % len(framepaths))
            print("")

            self.frame_paths.append(framepaths)     # list of lists. Length equal to number of videos. Each sublist contains the filepaths to the frames of a video
            self.label_paths.append(labelpath)      # list of strings. Length equal to number of videos. Each string is the path to the .txt label file for a video

        # Transform
        self.transform_frames = transforms.Compose([
            transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
            transforms.Normalize(MEAN, STD)
        ])


    def __getitem__(self, idx):
        
        # Load paths to frames/label file for a single video
        frame_paths = self.frame_paths[idx]     # paths to the frames for a video
        label_path = self.label_paths[idx]      # path to the label file for a video

        # Load labels into memory
        with open(label_path) as f:
            labels = f.read().splitlines() # list of strings, length = number of frames in current video
        f.close()

        # Load frames into memory, and convert labels to ints
        frames_list = []
        labels_list = []

        for i, path in enumerate(frame_paths):
            
            # Load image, transform to tensor and reshape
            img = Image.open(path).convert("RGB")
            img = self.transform_frames(img)                            # tensor (3, 240, 320)
            img = img.view(1, img.shape[0], img.shape[1], img.shape[2]) # tensor (1, 3, 240, 320)

            # Append to temporary lists
            frames_list.append(img)                     # list of tensors, length = number of frames in current video
            labels_list.append(int(labels[i]))          # list of integers, length = number of frames in current video

        # Convert to tensor
        frames_tensor = torch.cat(frames_list, dim=0)               # tensor (N, 3, 240, 320). All N frames for one video
        labels_tensor = torch.from_numpy( np.array(labels_list) )   # tensor (N). All N labels for one video

        return frames_tensor, labels_tensor


    def __len__(self):
        return len(self.frame_paths)    # number of videos that were sampled

