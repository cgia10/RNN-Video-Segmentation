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

import RNN
import dataloader_features_p2

# ---------
#  Parser
# ---------

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--gpu", type=int, default=0, help="GPU number")
parser.add_argument("--save_txtfile_dir", type=str, default="./output/", help="Directory to save generated text file of predictions")
parser.add_argument("--saved_model_dir", type=str, default="./Q2/saved_models/", help="Directory to load model")
parser.add_argument("--saved_model_name", type=str, default="RNN15.pth.tar", help="Name of saved model to load")
parser.add_argument("--val_label_dir", type=str, default="./data/", help="Path to .csv label file")
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

# Set seed for batch shuffling
random.seed(1)
torch.manual_seed(1)

# Function to load model
def load_model(model, name):
    checkpoint = torch.load(os.path.join(opt.saved_model_dir, name))
    model.load_state_dict(checkpoint)


# ------------
#  Dataloader
# ------------

# P2 dataloader
dataloader = torch.utils.data.DataLoader(
    dataset=dataloader_features_p2.DATA(opt),
    batch_size=1,
    shuffle=False,
    num_workers=opt.n_cpu
)


# --------------------
#  Model Declarations
# --------------------

# Initialize model
print("")
print("---> preparing model...")
my_net = RNN.RNN()
model_pth = os.path.join(opt.saved_model_dir, opt.saved_model_name)

if os.path.exists(model_pth):
    print("---> found previously saved {}, loading checkpoint...".format(opt.saved_model_name))
    load_model(my_net, opt.saved_model_name)
else:
    print("ERROR: unable to find saved model")
    sys.exit()

if cuda:
    my_net = my_net.cuda()


# -------------
#  Predictions
# -------------

print("---> generating predictions...")

my_net.eval()
preds = []

with torch.no_grad(): # do not need to calculate information for gradient during eval
    for idx, imgs in enumerate(dataloader):
        
        # Generate prediction
        imgs = imgs.cuda()
        pred = my_net(imgs)
        _, pred = torch.max(pred, dim=1)    # pred = tensor, dimension (batch_size)
        
        # Create list of predictions and ground truths
        pred = pred.cpu().numpy()     # pred = numpy array, dimension (batch_size)        
        preds.append(pred)

preds = np.concatenate(preds)


# -----------
#  .TXT file
# -----------

# Generate txt file
print("---> creating .txt file...")

f = open(os.path.join(opt.save_txtfile_dir, "p2_result.txt"), "w+")

for i in range(preds.shape[0]):
     f.write("%d\n" % (preds[i]))

f.close()

print("---> .txt file created, saved to: {}".format(opt.save_txtfile_dir))

print("***** FINISHED *****")
