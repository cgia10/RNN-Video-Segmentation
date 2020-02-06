import os
import reader

# Function to generate frames
def extract_frames(opt):
    
    label_dir = opt.val_label_dir
    video_dir = opt.val_video_dir

    # Read CSV label file
    video_dict = reader.getVideoList(label_dir)

    # Initialize return lists
    all_frames = [] # length equal to number of videos. Elements are sublists. Those lists contain numpy arrays of frames (240, 320, 3)
    
    # For length of the csv file:
    for i in range(len(video_dict["Video_index"])):

        print("Extracting frames from video %d..." % (i+1))
        frame_list = []

        # Take video category and video name from current dict entry
        folder_name = video_dict["Video_category"][i]
        file_name = video_dict["Video_name"][i]

        # Present to helper function
        current_frames = reader.readShortVideo(video_dir, folder_name, file_name)

        # Separate each frame in returned array and put into a list
        for j in range(current_frames.shape[0]):
            frame_list.append(current_frames[j, :, :, :])

        # Append the list of individual frames, and the corresponding label, onto the lists
        all_frames.append(frame_list)

    return all_frames
    

