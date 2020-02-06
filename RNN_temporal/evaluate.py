import numpy as np
import torch

# Function to calculate classification accuracy during training
def evaluate(model, data_loader):

    # Set model to evaluation mode
    model.eval()
    preds_list = []     # list of numpy arrays. Length = number of videos loaded
    labels_list = []    # list of numpy arrays. Length = number of videos loaded                

    with torch.no_grad():
        for _, (features, labels) in enumerate(data_loader):

            # features = (1, N, 2048)
            # labels = (1, 1, N)

            features = features.cuda()

            # Generate prediction
            preds = model(features).squeeze()  # tensor (N, 11)
            _, preds = torch.max(preds, dim=1) # tensor (N)

            # Append to list of predictions and ground truths
            preds = preds.cpu().numpy()             # numpy (N)
            labels = labels.squeeze().cpu().numpy() # numpy (N)

            preds_list.append(preds)
            labels_list.append(labels)

    preds_array = np.concatenate(preds_list)    # numpy, dimension = total number of frames in all videos loaded
    gts_array = np.concatenate(labels_list)     # numpy, dimension = total number of frames in all videos loaded

    # Calculate accuracy
    correct = 0
    total = 0
    
    for i, prediction in enumerate(preds_array):
        if prediction == gts_array[i]:
            correct += 1
        total += 1

    acc = (correct/total) * 100
    
    return correct, total, acc