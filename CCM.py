from causal_ccm import ccm
import os
import scipy
import mat73

import numpy as np
from scipy.io import loadmat  # this is the SciPy module that loads mat-files
import matplotlib.pyplot as plt
from datetime import datetime, date, time
import pandas as pd
from copent import transent

def load_all_files():

    data = []
    # iterate over files in
    # that directory

    files = [f for f in os.listdir('.') if os.path.isfile(f)]

    for f in files:
        try :
            data.append(loadmat(f))
        except :
            data.append(mat73.loadmat(f))

def create_transent_matrix(spikes):

    transfer_entropy = np.empty([586, 586])
    for i in range(1,586):
        for j in range(1,586):
                transfer_entropy[i][j]= transent(spikes.iloc[:,i], relevant_df.iloc[:,j])

    return transfer_entropy

def create_CCM_matrix(spikes, region_1,region_2, dim):

    tau = 1
    E = dim
    L = 80
    #find indexes and lengths of 2 regions
    #fill in here

    corr_matrix= np.empty([110,77])
    i = 0
    for k in range(0,110):
        j = 0
        for l in range(306,383):
            X = spikes[k]
            Y = spikes[l]
            ccm_XY = ccm(X, Y, tau, E, L).causality()[0] # define new ccm object # Testing for X -> Y
            ccm_YX = ccm(Y, X, tau, E, L).causality()[0] # define new ccm object # Testing for Y -> X    
            corr_matrix[i][j]= max(ccm_XY, ccm_YX)
            j+=1
        i+=1

    return corr_matrix

def CCM_search(spikes1, spikes2, df):

    CCM_df = pd.DataFrame()
    for i in range(2,len(spikes1)):
        for j in range(2,len(spikes2)):
            df = CCM(dataFrame = df, E = 4, columns = i, target = j, libSizes = "4 198 1", sample = 200, showPlot = False) 
            to_append = df.iloc[:,1:3]
            CCM_df = pd.concat([CCM_df,to_append],axis=1) 

    return CCM_df

def plot_CCM_matrix(matrix, title):

    matrix[np.isnan(matrix)] = 0
    fig, ax = plt.subplots()
    c = plt.pcolor(matrix)
    fig.colorbar(c)
    plt.title(title)
    plt.show()

def top_neurons(CCM_df, neurons):
    # return top N neuron indexes based on CCM for input CCM dataframe
    best_rho = CCM_df.max().nlargest(n=neurons)
    best_rho = pd.DataFrame(best_rho)
    top_indices = best_rho.index.drop_duplicates()

    return top_indices  
