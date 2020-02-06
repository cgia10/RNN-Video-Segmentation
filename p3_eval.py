import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--pred_dir", type=str, default="./predictions/", help="Path to directory containing predicted .txt files")
parser.add_argument("--gt_dir", type=str, default="../hw4_data/FullLengthVideos/labels/valid/", help="Path to directory containing gt .txt files")
opt = parser.parse_args()

files_preds = sorted(os.listdir(opt.pred_dir))
files_gts = sorted(os.listdir(opt.gt_dir))
all_total = 0
all_correct = 0

for (predfile, gtfile) in zip(files_preds, files_gts):

    pred_path = os.path.join(opt.pred_dir, predfile)
    gt_path = os.path.join(opt.gt_dir, gtfile)

    # Predictions
    with open(pred_path) as f:
        preds = f.read().splitlines() # list of strings, length = number of lines
    f.close()

    # Ground truths
    with open(gt_path) as f:
        gts = f.read().splitlines() # list of strings, length = number of lines
    f.close()

    # Accuracy
    correct = 0
    total = 0

    for prediction, gt in zip(preds, gts):
        if prediction == gt:
            correct += 1
            all_correct += 1
        total += 1
        all_total += 1

    acc = (correct/total) * 100

    print("%s: %d/%d correct (%.2f%%)" % (predfile, correct, total, acc))

all_acc = (all_correct/all_total) * 100
print("Total: %d/%d correct (%.2f%%)" % (all_correct, all_total, all_acc))

print("***** FINISHED *****")