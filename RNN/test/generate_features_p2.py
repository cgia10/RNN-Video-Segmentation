import os
import numpy as np
import random

import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
import torch.optim as optim

import extractor
import dataloader_frames_p2


def extract_features(opt):

    # ---------------------
    #  Miscellaneous Setup
    # ---------------------

    # Set up GPU
    if torch.cuda.is_available():
        cuda = True
        torch.cuda.set_device(opt.gpu)
    else:
        cuda = False

    # Set seed for batch shuffling
    random.seed(1)
    torch.manual_seed(1)


    # ------------
    #  Dataloader
    # ------------

    print("")
    print("---> preparing to extract frames...")

    # P1 dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset=dataloader_frames_p2.DATA(opt),
        batch_size=1,
        shuffle=False,
        num_workers=opt.n_cpu
    )


    # -------------------
    #  Model Declaration
    # -------------------

    # Initialize model
    print("")
    print("---> preparing to extract features...")
    
    my_net = extractor.Feat_extractor()

    if cuda:
        my_net = my_net.cuda()


    # --------------------
    #  Feature Extraction
    # --------------------
        
    my_net.eval()
    features_list = [] # length = num of videos. Contains tensors of size 2048

    with torch.no_grad():

        for idx, frames in enumerate(dataloader):    # dataloader has length equal to number of videos loaded
            
            frames = frames[0]  # python pads the tensor with an extra dimension for some reason. 1xNxCx240x320 -> NxCx240x320

            # Move to GPU
            if cuda:
                frames = frames.cuda()

            features = my_net(frames)   # tensor, Nx2048x2x4
            features = torch.mean(features, (2,3) ) # remove dimensions 2 and 3 by averaging them. Nx2048

            # Save features for current video
            print("Extracting features from video %d..." % (idx+1))

            features_list.append(features)

    return features_list