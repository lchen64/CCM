#!/usr/bin/env python
# coding: utf-8

# In[52]:


from causal_ccm import ccm
import os
import scipy
import mat73

import numpy as np
from scipy.io import loadmat  # this is the SciPy module that loads mat-files
import matplotlib.pyplot as plt
from datetime import datetime, date, time
import pandas as pd



path = os.path.join('Users', 'yoolab', 'Desktop', 'PFCInformationFlowData', 'Neurophysiology')


cd Desktop


cd Neurophysiology


ls


data_dict = mat73.loadmat('V20160929_neurons.mat')



data = []
# assign directory
directory = path
 
# iterate over files in
# that directory


#mat = loadmat('measured_data.mat')  # load mat-file
#mdata = mat['measuredData']

files = [f for f in os.listdir('.') if os.path.isfile(f)]

for f in files:
    try :
        data.append(loadmat(f))
    except :
        data.append(mat73.loadmat(f))

            

data[0]


ccm_out = ccm(X, Y, tau, E=2, L) # define ccm with X, Y time series



ccm_out.causality()

