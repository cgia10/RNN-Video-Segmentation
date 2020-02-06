import os
import numpy as np
import argparse
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import sys

import torch

import dataloader_features
import RNN


# ---------
#  Parser
# ---------

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--gpu", type=int, default=0, help="GPU number")
parser.add_argument("--load_val_features_dir", type=str, default="./features_val/", help="Directory from which to load features for val set")
parser.add_argument("--saved_model_dir", type=str, default="./saved_models/", help="Directory to save/load models")
parser.add_argument("--saved_model_name", type=str, default="RNN.pth.tar", help="Name of saved model to load")
opt = parser.parse_args()


# ---------------------
#  Miscellaneous Setup
# ---------------------

# Check if directory of val features exists
if not os.path.exists(opt.load_val_features_dir):
    print("ERROR: could not find directories of generated features")
    sys.exit()

# Set up GPU
if torch.cuda.is_available():
    print("GPU found")
    cuda = True
    torch.cuda.set_device(opt.gpu)
else:
    print("GPU not found")
    cuda = False

# Function to load model
def load_model(model, load_dir, name):
    checkpoint = torch.load(os.path.join(load_dir, name))
    model.load_state_dict(checkpoint)


# ------------
#  Dataloader
# ------------

print("---> preparing dataloader...")

dataloader = torch.utils.data.DataLoader(
    dataset=dataloader_features.DATA(opt, mode="val"),
    batch_size=1,
    shuffle=False,
    num_workers=opt.n_cpu
)


# -------
#  Model
# -------

# Initialize model
print("---> preparing model...")
my_net = RNN.RNN()
model_pth = os.path.join(opt.saved_model_dir, opt.saved_model_name)

if os.path.exists(model_pth):
    print("---> found previously saved {}, loading checkpoint...".format(opt.saved_model_name))
    load_model(my_net, opt.saved_model_dir, opt.saved_model_name)

my_net = my_net.cuda()

# ----------
#  Features
# ----------

print("---> loading features...")

# Set model to evaluation mode
my_net.eval()
features_RNN_list = []
labels_list = []

with torch.no_grad(): # do not need to calculate information for gradient during eval
    for idx, (features, label) in enumerate(dataloader):

        # Generate prediction
        features = features.cuda()  # (1, N, 2048)
        features_RNN = my_net.return_features(features).cpu().numpy()  # (1, 1024)
        label = label.numpy()

        features_RNN_list.append(features_RNN)
        labels_list.append(label)

features_RNN = np.concatenate(features_RNN_list)   # (N, 1024) where N is number of videos
labels = np.concatenate(labels_list)


# -------
#  t-SNE
# -------

# Perform t-SNE on features
print("---> performing t-SNE...")    
tsne = TSNE(n_components=2, random_state=1)
features_tsne = tsne.fit_transform(features_RNN)     # numpy array, batch size x 2

# Assign different colors for each action label
colors_label = []

for i in range(features_tsne.shape[0]):
    if labels[i] == 0:
        colors_label.append('k')
    elif labels[i] == 1:
        colors_label.append('r')
    elif labels[i] == 2:
        colors_label.append('g')
    elif labels[i] == 3:
        colors_label.append('b')
    elif labels[i] == 4:
        colors_label.append('c')
    elif labels[i] == 5:
        colors_label.append('m')
    elif labels[i] == 6:
        colors_label.append('pink')
    elif labels[i] == 7:
        colors_label.append('gold')
    elif labels[i] == 8:
        colors_label.append('cyan')
    elif labels[i] == 9:
        colors_label.append('orange')
    elif labels[i] == 10:
        colors_label.append('y')

# Plot t-SNE features, with color indicating action label
plt.figure()
plt.scatter(features_tsne[:,0], features_tsne[:,1], s=4, c=colors_label)
plt.savefig("tsne")
plt.close("all")

print("***** Plot Saved *****")
