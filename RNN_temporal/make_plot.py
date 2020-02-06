import argparse
import numpy as np
import os
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--preds", type=str, default="./preds.txt", help="Path to predictions")
parser.add_argument("--gt", type=str, default="./valid.txt", help="Path to ground truths")
parser.add_argument("--output_dir", type=str, default="./output/", help="Directory for output images")
opt = parser.parse_args()


# Create save directory for txt file of predictions
if not os.path.exists(opt.output_dir):
    print("Created directory to save outputs: {}".format(opt.output_dir))
    os.makedirs(opt.output_dir)

# Predictions
with open(opt.preds) as f:
    preds = f.read().splitlines() # list of strings, length = number of lines
f.close()

# Ground truths
with open(opt.gt) as f:
    gts = f.read().splitlines() # list of strings, length = number of lines
f.close()

# Convert to ints
for i in range(len(gts)):
    preds[i] = int(preds[i])
    gts[i] = int(gts[i])

# Create image arrays
W = 800 # width of image (number of frames to plot)
H = 30 # height of image
preds_array = np.zeros((H, W, 3), dtype="uint8")
gts_array = np.zeros((H, W, 3), dtype="uint8")

colour_dict = {0: np.array([255, 0, 0]),      # red (OTHER)
               1: np.array([0, 255, 0]),      # green (INSPECT/READ)
               2: np.array([0, 0, 255]),      # blue (OPEN)
               3: np.array([255, 255, 0]),    # yellow (TAKE)
               4: np.array([153, 0, 153]),    # purple (CUT)
               5: np.array([255, 128, 0]),    # orange (PUT)
               6: np.array([255, 0, 127]),    # pink (CLOSE)
               7: np.array([0, 153, 153]),    # cyan (MOVE AROUND)
               8: np.array([0, 0, 0]),        # black (DIVIDE/PULL APART)
               9: np.array([153, 153, 0]),    # gold (POUR)
               10: np.array([255, 153, 153])} # pale red (TRANSFER)

for i, (pred, gt) in enumerate(zip(preds, gts)):
    for j in range(H):
        preds_array[j,i,:] = colour_dict[pred]
        gts_array[j,i,:] = colour_dict[gt]

    # Break if desired number of frames is reached
    if i == (W-1):
        break

preds_img = Image.fromarray(preds_array, "RGB")
preds_img.save(os.path.join(opt.output_dir, "preds.jpg"))

gts_img = Image.fromarray(gts_array, "RGB")
gts_img.save(os.path.join(opt.output_dir, "gts.jpg"))

print("***** FINISHED *****")