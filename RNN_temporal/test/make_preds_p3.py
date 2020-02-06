import numpy as np
import random
import argparse
import os
import sys

import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.models as models

import extractor
import RNN
import dataloader_frames_p3

# ---------
#  Parser
# ---------

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0, help="GPU number")
parser.add_argument("--save_txtfile_dir", type=str, default="./output/", help="Directory to save generated text file of predictions")
parser.add_argument("--saved_model_dir", type=str, default="./Q3/saved_models/", help="Directory to load model")
parser.add_argument("--saved_model_name", type=str, default="RNN49.pth.tar", help="Name of saved model to load")
parser.add_argument("--val_video_dir", type=str, default="./data/", help="Path to videos")
opt = parser.parse_args()


# ---------------------
#  Miscellaneous Setup
# ---------------------

# Create save directory for txt file of predictions
if not os.path.exists(opt.save_txtfile_dir):
    print("Created directory to save .txt file of predictions: {}".format(opt.save_txtfile_dir))
    os.makedirs(opt.save_txtfile_dir)

# Set up GPU
if torch.cuda.is_available():
    print("GPU found")
    cuda = True
    torch.cuda.set_device(opt.gpu)
else:
    print("GPU not found")
    cuda = False

# Function to load model
def load_model(model, name):
    checkpoint = torch.load(os.path.join(opt.saved_model_dir, name))
    model.load_state_dict(checkpoint)


# ------------
#  Dataloader
# ------------

# Frame dataloader
dataloader = torch.utils.data.DataLoader(
    dataset=dataloader_frames_p3.DATA(opt),
    batch_size=1,
    shuffle=False,
    num_workers=0
)


# --------------------
#  Model Declarations
# --------------------

# Initialize models
print("")
print("---> preparing models...")

my_extractor = extractor.Feat_extractor()
my_RNN = RNN.RNN()

model_pth = os.path.join(opt.saved_model_dir, opt.saved_model_name)

if os.path.exists(model_pth):
    print("---> found previously saved {}, loading checkpoint...".format(opt.saved_model_name))
    load_model(my_RNN, opt.saved_model_name)
else:
    print("ERROR: unable to find saved model")
    sys.exit()

if cuda:
    my_extractor = my_extractor.cuda()
    my_RNN = my_RNN.cuda()


# -------------------------------
#  Prediction/Feature Extraction
# -------------------------------

print("---> begin evaluation...")
my_extractor.eval()
my_RNN.eval()

with torch.no_grad():

    # Cycle through each video
    for idx, (frames, vid_categories) in enumerate(dataloader):    # dataloader has length equal to number of videos loaded

        features_list = []  # each element in the list is a feature vector for a frame in the current video
        frames = frames[0]  # all frames in the current video. (1, N, 3, 240, 320) -> (N, 3, 240, 320)


        # --------------------
        #  Feature Extraction
        # --------------------

        print("")
        print("***** VIDEO %d *****" % (idx+1))
        print("---> extracting features...")

        # Cycle through each frame in current video
        for img in frames:  # img = (3, 240, 320)

            img = img.view(1, img.shape[0], img.shape[1], img.shape[2]) # add an extra dimension so the model works. (3, 240, 320) -> (1, 3, 240, 320)

            # Move to GPU
            if cuda:
                img = img.cuda()

            # Extract features
            features = my_extractor(img)   # (1, 2048, 2, 4)
            features_list.append( torch.mean(features, (2,3)).cpu() ) # each element is (1, 2048)

        # Concatenate into a tensor. All N features for 1 video
        features = torch.cat(features_list, dim=0)  # (N, 2048)
        features = features.view(1, features.shape[0], features.shape[1]) # add an extra dimension so the model works. (1, N, 2048)
        features = features.cuda()

        # Clear memory
        del features_list
        del img
        del frames


        # ------------
        #  Prediction
        # ------------

        # Generate predictions for frames of the current video
        print("---> generating predictions...")
        preds = my_RNN(features).squeeze()  # tensor (N, 11)
        
        # Clear memory
        del features

        # Append to list of predictions
        _, preds = torch.max(preds, dim=1)  # tensor (N)
        preds = preds.cpu().numpy()         # numpy (N)

        
        # ------------
        #  .TXT files
        # ------------

        # Generate .txt file for current video
        print("---> creating .txt file...")

        fname = "{}.txt".format(vid_categories[idx][0])
        f = open(os.path.join(opt.save_txtfile_dir, fname), "w+")

        for i in range(preds.shape[0]):
            f.write("%d\n" % (preds[i]))

        f.close()

        print("---> {} created, saved to: {}".format(fname, opt.save_txtfile_dir))
        
        # Clear memory
        del preds


print("***** FINISHED *****")
