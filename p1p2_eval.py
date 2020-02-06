import argparse
import reader

parser = argparse.ArgumentParser()
parser.add_argument("--preds", type=str, default="", help="Path to predictions")
parser.add_argument("--gt", type=str, default="", help="Path to ground truths")
opt = parser.parse_args()

# Predictions
with open(opt.preds) as f:
    preds = f.read().splitlines() # list of strings, length = number of lines
f.close()

# Ground truth labels
video_dict = reader.getVideoList(opt.gt)

# Accuracy
correct = 0
total = 0

for idx, prediction in enumerate(preds):
    if prediction == video_dict["Action_labels"][idx]:
        correct += 1
    total += 1

acc = (correct/total) * 100

print("ACC: %d/%d correct (%.2f%%)" % (correct, total, acc))
print("***** FINISHED *****")
