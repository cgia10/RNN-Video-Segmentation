import argparse
import os
import random
import pickle as pk

import torch
import torch.utils.data
import torch.backends.cudnn as cudnn

import extractor
import dataloader_frames


# ---------
#  Parser
# ---------

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--gpu", type=int, default=0, help="GPU number")

parser.add_argument("--train_label_dir", type=str, default="./train/", help="")
parser.add_argument("--train_video_dir", type=str, default="./train/", help="")
parser.add_argument("--num_videos_train", type=int, default=1, help="")

parser.add_argument("--val_label_dir", type=str, default="./valid/", help="")
parser.add_argument("--val_video_dir", type=str, default="./valid/", help="")
parser.add_argument("--num_videos_val", type=int, default=1, help="")

parser.add_argument("--save_train_features_dir", type=str, default="./features_train/", help="Directory to save generated features from train set")
parser.add_argument("--save_val_features_dir", type=str, default="./features_val/", help="Directory to save generated features from validation set")
opt = parser.parse_args()


# ---------------------
#  Miscellaneous Setup
# ---------------------

# Create save directory for features from train set
if not os.path.exists(opt.save_train_features_dir):
    print("Created directory to save features from train set: {}".format(opt.save_train_features_dir))
    os.makedirs(opt.save_train_features_dir)

# Create save directory for features from val set
if not os.path.exists(opt.save_val_features_dir):
    print("Created directory to save features from validation set: {}".format(opt.save_val_features_dir))
    os.makedirs(opt.save_val_features_dir)

# Set up GPU
if torch.cuda.is_available():
    print("GPU found")
    cuda = True
    torch.cuda.set_device(opt.gpu)
else:
    print("GPU not found")
    cuda = False

# Set seed for batch shuffling
random.seed(1)
torch.manual_seed(1)


# -------------
#  Dataloaders
# -------------

print("---> preparing dataloaders...")

# Training dataloader
dataloader_train = torch.utils.data.DataLoader(
    dataset=dataloader_frames.DATA(opt, mode="train"),
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=opt.n_cpu
)

# Validation dataloader
dataloader_val = torch.utils.data.DataLoader(
    dataset=dataloader_frames.DATA(opt, mode="val"),
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=opt.n_cpu
)


# -------
#  Model
# -------

# Initialize model
print("")
print("---> preparing model...")
my_net = extractor.Feat_extractor()

if cuda:
    my_net = my_net.cuda()


# --------------------
#  Feature Extraction
# --------------------

def extract_features(model, dataloader, save_dir):
    
    model.eval()
    data_dict = {}

    with torch.no_grad():

        # Outer for loop cycles through each video
        for idx, (frames, label) in enumerate(dataloader):    # dataloader has length equal to number of videos loaded
            
            frames = frames[0]  # all frames in the current video. (1, N, 3, 240, 320) -> (N, 3, 240, 320)
            features_list = []  # features for the frames of the current video

            print("")
            print("***** VIDEO %d *****" % (idx+1))
            print("--> found %d frames" % frames.shape[0])
            print("--> extracting features...")

            # Inner for loop cycles through the frames in current video
            for img in frames:  # img = (3, 240, 320)

                img = img.view(1, img.shape[0], img.shape[1], img.shape[2]) # add an extra dimension so the model works. (3, 240, 320) -> (1, 3, 240, 320)

                # Move to GPU
                if cuda:
                    img = img.cuda()

                features = my_net(img)   # (1, 2048, 2, 4)
                features_list.append( torch.mean(features, (2,3)).cpu() ) # remove dimensions 2 and 3 by averaging, and append into list. (1, 2048)

            features = torch.cat(features_list, dim=0) # concatenate list into a tensor (N, 2048)

            # Save features for current video
            print("--> saving features...")

            data_dict["features"] = features    # tensor, (N, 2048)
            data_dict["label"] = label          # tensor, (1, N)

            with open(os.path.join(save_dir, "{}.pk".format(idx+1)), "wb") as f:
                pk.dump(data_dict, f)


# Training set
print("---> start extracting features from train set...")
extract_features(my_net, dataloader_train, opt.save_train_features_dir)
print("***** Finished extracting from train set *****")
print("")

# Validation set
print("---> start extracting features from validation set...")
extract_features(my_net, dataloader_val, opt.save_val_features_dir)
print("***** Finished extracting from validation set *****")
print("")

print("***** FINISHED *****")