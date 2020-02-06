import argparse
import os
import numpy as np
import random
import pickle as pk
import sys
import matplotlib.pyplot as plt

import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
import torch.optim as optim

import RNN
import dataloader_features
from evaluate import evaluate


# ---------
#  Parser
# ---------

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1, help="number of epochs of training")
parser.add_argument("--val_epoch", type=int, default=1, help="on which epoch to save model and image")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--gpu", type=int, default=0, help="GPU number")
parser.add_argument("--continue_epoch", type=int, default=1, help="Epoch number which was reached last time. Used to name saved models/images")
parser.add_argument("--load_train_features_dir", type=str, default="./features_train/", help="Directory from which to load features for train set")
parser.add_argument("--load_val_features_dir", type=str, default="./features_val/", help="Directory from which to load features for val set")
parser.add_argument("--saved_model_dir", type=str, default="./saved_models/", help="Directory to save/load models")
parser.add_argument("--saved_info_dir", type=str, default="./train_info/", help="Directory to save/load training loss")
parser.add_argument("--output_dir", type=str, default="./output/", help="Directory to save miscellaneous outputs")
parser.add_argument("--saved_model_name", type=str, default="RNN.pth.tar", help="Name of saved model to load")
parser.add_argument("--saved_loss_name", type=str, default="loss1.pk", help="Name of saved training loss to load")
parser.add_argument("--saved_iter_name", type=str, default="iters1.pk", help="Name of saved iterations to load")
parser.add_argument("--saved_acc_name", type=str, default="acc1.pk", help="Name of saved accuracies to load")
parser.add_argument("--saved_acc_iters_name", type=str, default="acc_iters1.pk", help="Name of saved accuracy iterations to load")
opt = parser.parse_args()


# ---------------------
#  Miscellaneous Setup
# ---------------------

# Check if directories of train/val features exists
if (not os.path.exists(opt.load_train_features_dir)) or (not os.path.exists(opt.load_val_features_dir)):
    print("ERROR: could not find directories of generated features")
    sys.exit()

# Set naming of model
if opt.continue_epoch == 1:
    opt.continue_epoch = opt.val_epoch
else:
    opt.continue_epoch += opt.val_epoch

# Directory for saved models
if not os.path.exists(opt.saved_model_dir):
    print("Created directory for saved models: {}".format(opt.saved_model_dir))
    os.makedirs(opt.saved_model_dir)

# Directory for saved training info
if not os.path.exists(opt.saved_info_dir):
    print("Created directory for saved training loss: {}".format(opt.saved_info_dir))
    os.makedirs(opt.saved_info_dir)

# Directory for miscellaneous outputs
if not os.path.exists(opt.output_dir):
    print("Created directory for miscellaneous outputs: {}".format(opt.output_dir))
    os.makedirs(opt.output_dir)

# Set up GPU
if torch.cuda.is_available():
    print("GPU found")
    cuda = True
    torch.cuda.set_device(opt.gpu)
else:
    print("GPU not found")
    cuda = False

# Set seeds for batch shuffling
random.seed(1)
torch.manual_seed(1)

# Function to save model
def save_model(model, save_dir, name):
    checkpoint = model.state_dict()
    torch.save(checkpoint, os.path.join(save_dir, name))

# Function to load model
def load_model(model, load_dir, name):
    checkpoint = torch.load(os.path.join(load_dir, name))
    model.load_state_dict(checkpoint)

# Function to plot training loss
def plot_loss(loss_record, iteration_record, number, save_dir):
    losses = np.array(loss_record)
    iterations = np.array(iteration_record)
    pth = os.path.join(save_dir, "Loss_vs_Iters_{}".format(number))

    plt.figure()
    plt.plot(iterations, losses, linewidth=0.5)
    plt.title("Loss vs Training Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig(pth)
    plt.close("all")

# Function to plot accuracy
def plot_acc(acc_record, iteration_record, number, save_dir):
    accs = np.array(acc_record)
    iterations = np.array(iteration_record)
    pth = os.path.join(save_dir, "Acc_vs_Iters_{}".format(number))

    plt.figure()
    plt.plot(iterations, accs, linewidth=0.5)
    plt.title("Accuracy vs Training Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.savefig(pth)
    plt.close("all")

# Function to save training loss information
def save_info(loss_record, iteration_record, acc_record, acc_iters_record, number, save_dir):

    with open(os.path.join(save_dir, "loss{}.pk".format(number)), "wb") as f:
        pk.dump(loss_record, f)
    
    with open(os.path.join(save_dir, "iters{}.pk".format(number)), "wb") as f:
        pk.dump(iteration_record, f)

    with open(os.path.join(save_dir, "acc{}.pk".format(number)), "wb") as f:
        pk.dump(acc_record, f)
    
    with open(os.path.join(save_dir, "acc_iters{}.pk".format(number)), "wb") as f:
        pk.dump(acc_iters_record, f)

# Function to load training loss information
def load_loss(load_dir, name):
    
    with open(os.path.join(load_dir, name), "rb") as f:
        item = pk.load(f)
    
    return item


# -------------
#  Dataloaders
# -------------

print("---> preparing dataloaders...")
print("")

# Training dataloader
dataloader_train = torch.utils.data.DataLoader(
    dataset=dataloader_features.DATA(opt, mode="train"),
    batch_size=1,
    shuffle=True,
    num_workers=opt.n_cpu
)

# Validation dataloader
dataloader_val = torch.utils.data.DataLoader(
    dataset=dataloader_features.DATA(opt, mode="val"),
    batch_size=1,
    shuffle=False,
    num_workers=opt.n_cpu
)


# -------
#  Model
# -------

# Initialize model
print("")
print("---> preparing model...")
my_net = RNN.RNN()
model_pth = os.path.join(opt.saved_model_dir, opt.saved_model_name)

if os.path.exists(model_pth):
    print("---> found previously saved {}, loading checkpoint...".format(opt.saved_model_name))
    load_model(my_net, opt.saved_model_dir, opt.saved_model_name)


# ----------------
#  Loss/Optimizer
# ----------------

# Loss function
print("---> preparing loss function...")
loss_criteria = torch.nn.CrossEntropyLoss()

# Optimizer
print("---> preparing optimizer...")
optimizer = optim.Adam(my_net.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Move everything to GPU
if cuda:
    my_net = my_net.cuda()
    loss_criteria = loss_criteria.cuda()

# Load training information
loss_pth = os.path.join(opt.saved_info_dir, opt.saved_loss_name)
iters_pth = os.path.join(opt.saved_info_dir, opt.saved_iter_name)
acc_pth = os.path.join(opt.saved_info_dir, opt.saved_acc_name)
acc_iters_pth = os.path.join(opt.saved_info_dir, opt.saved_acc_iters_name)

if os.path.exists(loss_pth):
    print("---> found previously saved {}, loading data...".format(opt.saved_loss_name))
    loss_record = load_loss(opt.saved_info_dir, opt.saved_loss_name)
else:
    loss_record = []

if os.path.exists(iters_pth):
    print("---> found previously saved {}, loading data...".format(opt.saved_iter_name))
    iteration_record = load_loss(opt.saved_info_dir, opt.saved_iter_name)
    iters = iteration_record[-1]
else:
    iteration_record = []
    iters = 1

if os.path.exists(acc_pth):
    print("---> found previously saved {}, loading data...".format(opt.saved_acc_name))
    acc_record = load_loss(opt.saved_info_dir, opt.saved_acc_name)
    best_acc = max(acc_record)
    best_epoch = -1
else:
    acc_record = []
    best_acc = 0
    best_epoch = 1

if os.path.exists(acc_iters_pth):
    print("---> found previously saved {}, loading data...".format(opt.saved_acc_iters_name))
    acc_iters_record = load_loss(opt.saved_info_dir, opt.saved_acc_iters_name)
else:
    acc_iters_record = []


# ----------
#  Training
# ----------

print("---> start training...")

for epoch in range(1, opt.n_epochs+1):

    my_net.train()

    for idx, (features, labels) in enumerate(dataloader_train):    # dataloader has length equal to number of videos loaded
        
        # features = (1, N, 2048)
        # labels = (1, 1, N)
            
        # Move to GPU
        if cuda:
            features = features.cuda()
            labels = labels.cuda()

        # Model output and classification loss
        preds = my_net(features).squeeze()  # (N, 11)
        labels = labels.squeeze()           # (N)
        loss = loss_criteria(preds, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Training info
        if iters % 1 == 0:
            loss_record.append(loss.item())
            iteration_record.append(iters)

        print("[Epoch %d/%d] [Batch %d/%d] [Loss: %f]" % (epoch, opt.n_epochs, (idx+1), len(dataloader_train), loss.item()))

        iters += 1

    
    # ------------------
    #  Acc/Model Saving
    # ------------------

    if epoch % opt.val_epoch == 0:     
        
        # Accuracy on test set
        correct, total, acc = evaluate(my_net, dataloader_val)
        acc_record.append(acc)
        acc_iters_record.append(iters)

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch

        # Print info
        print("")
        print("********** Current: epoch %d   Total: epoch %d **********" % (epoch, opt.continue_epoch))
        print("ACC: %d/%d correct (%.2f%%)" % (correct, total, acc))
        
        # Save model
        save_model(my_net, opt.saved_model_dir, "RNN{}.pth.tar".format(opt.continue_epoch))
        print("Saving model: RNN{}.pth.tar".format(opt.continue_epoch))

        # Plot training loss
        plot_loss(loss_record, iteration_record, opt.continue_epoch, opt.output_dir)
        print("Saving plot: Loss_vs_Iters_{}".format(opt.continue_epoch))

        # Plot accuracy
        plot_acc(acc_record, acc_iters_record, opt.continue_epoch, opt.output_dir)
        print("Saving plot: Acc_vs_Iters_{}".format(opt.continue_epoch))

        # Save training info
        save_info(loss_record, iteration_record, acc_record, acc_iters_record, opt.continue_epoch, opt.saved_info_dir)
        print("Saving train info: loss{}.pk, iters{}.pk, acc{}.pk, acc_iters{}".format(opt.continue_epoch, opt.continue_epoch, opt.continue_epoch, opt.continue_epoch))

        print("")
        opt.continue_epoch += opt.val_epoch


print("Best ACC: %.2f%% on epoch %d" % (best_acc, best_epoch))
print("")
print("***** FINISHED *****")
