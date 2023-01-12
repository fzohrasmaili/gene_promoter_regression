import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
from random import randint
import numpy as np
import time
import os
import sys
import numpy
import sklearn
import sklearn.metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

##########################################################################
# Define functions
# a. Define suequence processing function
def decimal_to_binary_tensor(value_list, width):
    binary_list =[]
    for value in value_list:
        binary =np.zeros(4)
        if value == 1:
            binary = np.array([1, 0, 0, 0]) #Nucleotide A
        if value == 2:
            binary = np.array([0, 1, 0, 0]) #Nucleotide T
        if value == 3:
            binary = np.array([0, 0, 1, 0]) # Nucleotide C
        if value == 4:
            binary = np.array([0, 0, 0, 1]) # Nucleotide G
        if value == 0:
            binary = np.array([0, 0, 0, 0])
        binary_list.append(binary)
    return torch.tensor(np.asarray(binary_list), dtype=torch.uint8)
# b. Define file to list function
def file_to_list (filename):
    file_list=[]
    file=open(filename, 'r')
    cnt=1
    for line in file:
        line =line.strip().replace('n','0')
        lst=list(map(int,line))
        new_tens = decimal_to_binary_tensor(lst, width=4)
        file_list.append(new_tens)
        cnt=cnt+1
    stacked_tensor = torch.stack(file_list)
    return stacked_tensor
def evaluation_regression(actual_tsr, predicted_tsr):
    from sklearn.metrics import mean_squared_error
    import math
    actual = actual_tsr.squeeze().detach().numpy()
    predicted  = predicted_tsr.squeeze().detach().numpy()
    mse = mean_squared_error(actual, predicted)
    rmse = math.sqrt(mse)
    nrmse = rmse/np.mean(actual)
    # Relative Squared Error (RSE)
    err= predicted - actual
    err2 = actual - np.mean(actual)
    RSE = np.sum(np.multiply(err,err)) / np.sum(np.multiply(err2,err2))
    return mse, rmse, nrmse, RSE
def plot_contrast_distribution (actual_scores, predicted_scores, i):
    plt.clf()
    plt.figure(i)
    plt.hist(actual_scores, bins=50, alpha=0.5, label='real scores')
    plt.hist(predicted_scores, bins=50, alpha=0.5, label='predicted scores')
    plt.legend(loc='upper right')
    MI = sklearn.metrics.mutual_info_score (actual_scores, predicted_scores)
    plt.title(f'LV_{i}  MI score = {MI}')
    #plt.show()
    plt.savefig(f"LV_{i}_distri_RESCONV.png")
def scatter_plot (actual_scores, predicted_scores, i, prefix=""):
    plt.clf()
    fig, ax = plt.subplots()
    ax.scatter(actual_scores, predicted_scores,  alpha=0.5)
    lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]   
    # now plot both limits against eachother
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)  
    ax.legend(loc='upper right')
    pearson_correlation = pearsonr (actual_scores, predicted_scores)
    plt.title(f'LV_{i} Pearson corr = {pearson_correlation}')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    #plt.show()
    fig.savefig(f"test_single/{prefix}LV_{i}_scatter_RESCONV.png")
