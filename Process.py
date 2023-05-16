import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date, time
import pandas as pd

from scipy.io import loadmat  # SciPy module that loads mat-files
from scipy.ndimage import gaussian_filter1d

from pyEDM import *
from causal_ccm import *
from tqdm import tqdm # for showing progress bar in for loops
from operator import add

#get single trials spike train, index = trial #
def get_single_trial_data(spikes, index):

    trial_spikes = []
    for train in spikes:
        trial_spikes.append(train[index])
        
    return trial_spikes

def average_single_trial_data(spikes, index1, index2):

    sum = np.empty([585, 201])
    for i in range(index1, index2):
        sum += numpy.asarray(get_single_trial_data(spikes, i))
    tot = index2-index1-1
    avg = sum/tot
    
    return avg

# Function for applying Gaussian Kernel as a Filter with Standard Deviation Sigma 
def gaussian_filter(spikes, sigma):
    smoothed = np.empty([585, 386523])
    i = 0
    for train in spikes:
        smoothed[i] = gaussian_filter1d(train, sigma)
        i+=1
        
    return smoothed
 
def filter_waldo_voltaire(neurons):

    waldo = []
    voltaire = []
    for n in range(0,584):
        if neurons[n][0] == 'W':
            waldo.append(n)
        if neurons[n][0] == 'v':
            voltaire.append(n)

    return waldo, voltaire

# Only include time bins 101 through 181
def filter_time_bins(spikes):

    relevant_spikes = []
    for train in spikes:
        spike_train = []
        for trial in train:
            spike_train.extend(trial[101:181].tolist())
        relevant_spikes.append(spike_train)

    return relevant_spikes

# Function for Applying Moving Average Time Series Smoother
def moving_average(spikes, kernel_size):
    smoothed = np.empty([585, spikes.shape[1]])
    i = 0
    for train in spikes:
        kernel = np.ones(kernel_size) / kernel_size
        smoothed[i] = np.convolve(train, kernel, mode='same')
        i+=1
    return smoothed

#Get only spike trains from the last 30 trials after 20 trials of uncertainty
def truncate(spikes):
    truncated = np.empty([585,2400])
    i = 0 
    for train in spikes: 
        truncated[i] = train[4000:6400]
        i+=1
    return truncated 

def filter_based_on_sparsity(spikes, threshold):
    trial_spikes = get_single_trial_data(spikes, 0)
    index = 0
    for neuron in trial_spikes:
        if sum(neuron) <= threshold:
            del trial_spikes[index]
        index+=1
    return trial_spikes

def get_neuron_groups(neurons):

    aNeurons = []
    bNeurons = []
    cNeurons = []
    dNeurons = []
    eNeurons = []
    fNeurons = []
    gNeurons = []
    hNeurons = []

    for n in range(0,584):
        if neurons[n][1][0] == 'A':
            aNeurons.append(n)
        if neurons[n][1][0] == 'B':
            bNeurons.append(n)
        if neurons[n][1][0] == 'C':
            cNeurons.append(n)
        if neurons[n][1][0] == 'D':
            dNeurons.append(n)
        if neurons[n][1][0] == 'E':
            eNeurons.append(n)
        if neurons[n][1][0] == 'F':
            fNeurons.append(n)
        if neurons[n][1][0] == 'G':
            gNeurons.append(n)
        if neurons[n][1][0] == 'H':
            hNeurons.append(n)

    return aNeurons, bNeurons, cNeurons, dNeurons, eNeurons, fNeurons, gNeurons, hNeurons

#Function to create dataframe for CCM from spike train list 
def create_spike_dataframe(spikes):

    relevant_df = pd.DataFrame()
    relevant_df = relevant_df.T
    relevant = relevant[~np.all(relevant == 0, axis=1)]
    relevant_df['time'] = range(1, len(relevant_df) + 1)
    relevant_df.columns = relevant_df.columns.astype(str)
    cols = list(relevant_df.columns)
    cols = [cols[-1]] + cols[:-1]
    relevant_df = relevant_df[cols]

    return relevant_df
