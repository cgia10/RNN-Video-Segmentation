import os
import reader
import pickle as pk
import argparse


# ---------
#  Parser
# ---------

parser = argparse.ArgumentParser()
parser.add_argument("--train_label_dir", type=str, default="./gt_train.csv", help="Path to ground truth CSV file for train set")
parser.add_argument("--val_label_dir", type=str, default="./gt_valid.csv", help="Path to ground truth CSV file for validation set")
parser.add_argument("--train_video_dir", type=str, default="./train/", help="Path to video directory for train set")
parser.add_argument("--val_video_dir", type=str, default="./valid/", help="Path to video directory for validation set")
parser.add_argument("--save_train_frames_dir", type=str, default="./frames_train/", help="Directory to save generated frames for train set")
parser.add_argument("--save_val_frames_dir", type=str, default="./frames_val/", help="Directory to save generated frames for validation set")
parser.add_argument("--num_videos_train", type=int, default="50", help="Number of videos to load for train set")
parser.add_argument("--num_videos_val", type=int, default="30", help="Number of videos to load for validation set")
opt = parser.parse_args()


# -------------
#  Directories
# -------------

# Create save directory for frames from train set
if not os.path.exists(opt.save_train_frames_dir):
    print("Created directory to save frames from train set: {}".format(opt.save_train_frames_dir))
    os.makedirs(opt.save_train_frames_dir)

# Create save directory for frames from val set
if not os.path.exists(opt.save_val_frames_dir):
    print("Created directory to save frames from validation set: {}".format(opt.save_val_frames_dir))
    os.makedirs(opt.save_val_frames_dir)


# ------------------
#  Frame extraction
# ------------------

# Function to generate frames
def extract_frames(opt, mode):
    
    if mode == "train":
        label_dir = opt.train_label_dir
        video_dir = opt.train_video_dir
        save_dir = opt.save_train_frames_dir
        end_video_index = opt.num_videos_train
    elif mode == "val":
        label_dir = opt.val_label_dir
        video_dir = opt.val_video_dir
        save_dir = opt.save_val_frames_dir
        end_video_index = opt.num_videos_val
    else:
        print("ERROR: invalid mode in frame generator")

    # Read CSV label file
    video_dict = reader.getVideoList(label_dir)

    # For length of the csv file:
    for i in range(end_video_index):
        
        # Clear dict and list for each new video
        data_dict = {}
        frame_list = []

        # Take video category and video name from current dict entry
        folder_name = video_dict["Video_category"][i]
        file_name = video_dict["Video_name"][i]

        # Present to helper function
        frames = reader.readShortVideo(video_dir, folder_name, file_name)

        # Separate each frame in returned array and put into a list
        for j in range(frames.shape[0]):
            frame_list.append(frames[j, :, :, :])
        
        # Populate a dictionary with the list of individual frames, and the corresponding label
        data_dict["frame_list"] = frame_list
        data_dict["label"] = video_dict["Action_labels"][i]

        # Save dict of frames/label for current video
        print("Saving frames from video %d..." % (i+1))

        with open(os.path.join(save_dir, "{}.pk".format(i+1)), "wb") as f:
            pk.dump(data_dict, f)


# Generate frames for train set
print("")
print("---> start extracting frames from train set...")
extract_frames(opt, "train")
print("***** Finished extracting from train set *****")
print("")

# Generate frames for validation set
print("---> start extracting frames from validation set...")
extract_frames(opt, "val")
print("***** Finished extracting from validation set *****")
print("")

print("***** FINISHED *****")   