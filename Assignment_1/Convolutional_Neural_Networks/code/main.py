import numpy as np
import time
import h5py
import argparse
import os.path
import torch
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import json
import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torchsummary import summary

from Convolutional_Neural_Networks.code.cnn import CNNModel


# from utils import str2bool  # Utility function for argument parsing


def load_data(DATA_PATH, batch_size):
    print(f"data_path: {DATA_PATH}")

    # Define transformations
    train_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Create train and test datasets
    train_dataset = ImageFolder(root=f"{DATA_PATH}train", transform=train_trans)
    test_dataset = ImageFolder(root=f"{DATA_PATH}test", transform=test_trans)

    # Create data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    return train_loader, test_loader


def compute_accuracy(y_pred, y_batch):
    accy = (y_pred == y_batch).sum().item() / len(y_batch)
    return accy


def validate_model(model, val_loader, device):
    model.eval()
    val_loss = 0
    val_accuracy = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x_batch, y_labels in val_loader:
            x_batch, y_labels = x_batch.to(device), y_labels.to(device)
            output_y = model(x_batch)
            loss = nn.CrossEntropyLoss()(output_y, y_labels)
            val_loss += loss.item()
            _, preds = torch.max(output_y, 1)
            val_accuracy += (preds == y_labels).float().mean()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_labels.cpu().numpy())

    confusion = confusion_matrix(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    return val_loss / len(val_loader), val_accuracy / len(val_loader), confusion, precision, recall, f1


def print_model_size(model):
    summary(model, (1, 28, 28))


def adjust_learning_rate(learning_rate, optimizer, epoch, decay):
    lr = learning_rate
    if epoch > 5:
        lr = 0.001
    if epoch >= 10:
        lr = 0.0001
    if epoch > 20:
        lr = 0.00001

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_one_epoch(model, optimizer, train_loader, device):
    model.train()
    for batch_id, (x_batch, y_labels) in tqdm(enumerate(train_loader), desc="Training", leave=False):
        x_batch, y_labels = Variable(x_batch).to(device), Variable(y_labels).to(device)

        output_y = model(x_batch)
        loss = nn.CrossEntropyLoss()(output_y, y_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, y_pred = torch.max(output_y.data, 1)
        accy = compute_accuracy(y_pred, y_labels)

        # Here, you can add code to log or print the loss and accuracy if you want


def test_model(model, test_loader, device):
    model.eval()
    total_accy = 0
    for batch_id, (x_batch, y_labels) in tqdm(enumerate(test_loader), desc="Testing", leave=False):
        x_batch, y_labels = Variable(x_batch).to(device), Variable(y_labels).to(device)
        output_y = model(x_batch)
        _, y_pred = torch.max(output_y.data, 1)
        accy = compute_accuracy(y_pred, y_labels)
        total_accy += accy
    return total_accy / len(test_loader)


def main():
    # TODO These args are for use in main and also for building your network. The current values are defaults and can be edited to suite your needs!
    """
    args:
    "-mode", dest="mode", type=str, default='train', help="train or test"
    "-num_epochs", dest="num_epoches", type=int, default=40, help="num of epoches"
    "-fc_hidden1", dest="fc_hidden1", type=int, default=100, help="dim of hidden neurons"
    "-fc_hidden2", dest="fc_hidden2", type=int, default=100, help="dim of hidden neurons"
    "-learning_rate", dest ="learning_rate", type=float, default=0.001, help = "learning rate"
    "-decay", dest ="decay", type=float, default=0.5, help = "learning rate"
    "-batch_size", dest="batch_size", type=int, default=100, help="batch size"
    "-dropout", dest ="dropout", type=float, default=0.4, help = "dropout prob"
    "-rotation", dest="rotation", type=int, default=10, help="image rotation"
    "-load_checkpoint", dest="load_checkpoint", type=bool, default=True, help="true of false"

    "-activation", dest="activation", type=str, default='relu', help="activation function"
    "-channel_out1", dest='channel_out1', type=int, default=64, help="number of channels"
    "-channel_out2", dest='channel_out2', type=int, default=64, help="number of channels"
    "-k_size", dest='k_size', type=int, default=4, help="size of filter"
    "-pooling_size", dest='pooling_size', type=int, default=2, help="size for max pooling"
    "-stride", dest='stride', type=int, default=1, help="stride for filter"
    "-max_stride", dest='max_stride', type=int, default=2, help="stride for max pooling"
    "-ckp_path", dest='ckp_path', type=str, default="checkpoint", help="path of checkpoint"
    """
    args = argparse.Namespace(
        mode='train',
        num_epochs=1,
        fc_hidden1=512,
        fc_hidden2=128,
        learning_rate=0.01, # prev 0.002
        decay=0.5,
        batch_size=100,
        dropout=0.4,
        rotation=10,
        load_checkpoint=False,
        activation='relu',
        channel_out1=128,
        channel_out2=64, # change these values
        stride=1,
        max_stride=2,
        ckp_path='checkpoint',
        k_size=4, # I can change kernel size as you move across the layers.
        pooling_size=2,
    )
    # For hidden layers and convolutional layers, I want to vary the number of hidden units or output channels
    # e.g, decreasing in each layer, or decrease then increase the decrease, etc. (repeating this pattern)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # use_mps = torch.backends.mps.is_available()
    # device = torch.device("mps" if use_mps else "cpu")
    print(f"device: {device}")

    train_loader, test_loader = load_data("C:/Users/aqwan/GitHub/CMSC421-FALL23-Students/Assignment_1/Convolutional_Neural_Networks/data/sampled_CINIC_10/", args.batch_size)

    model = CNNModel(args=args).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.num_epochs):
        adjust_learning_rate(args.learning_rate, optimizer, epoch, args.decay)
        train_one_epoch(model, optimizer, train_loader, device)
        test_accuracy = test_model(model, test_loader, device)
        print(f"Epoch {epoch + 1}, Test Accuracy: {test_accuracy}")

        # Optionally, save model checkpoint here

    PATH = './CINIC_10_CNN_weights.pth'
    torch.save(model.state_dict(), PATH)


    print("Training Complete!")


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Running time: {(end_time - start_time) / 60.0:.2f} mins")