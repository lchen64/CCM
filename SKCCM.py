# %%
import skccm.data as data
import matplotlib

rx1 = 3.72 #determines chaotic behavior of the x1 series
rx2 = 3.72 #determines chaotic behavior of the x2 series
b12 = 0.2 #Influence of x1 on x2
b21 = 0.01 #Influence of x2 on x1
ts_length = 1000
x1,x2 = data.coupled_logistic(rx1,rx2,b12,b21,ts_length)



# %%

plt.plot(x1)
plt.xlim(0, 100)
plt.title("x1 in the coupled logistic data")

# %%
plt.plot(x2)
plt.xlim(0, 100)
plt.title("x2 in the coupled logistic data")

# %%
import skccm as ccm
lag = 1
embed = 2
e1 = ccm.Embed(x1)
e2 = ccm.Embed(x2)
X1 = e1.embed_vectors_1d(lag,embed)
X2 = e2.embed_vectors_1d(lag,embed)

# %%
from skccm.utilities import train_test_split
import numpy as np

#split the embedded time series
x1tr, x1te, x2tr, x2te = train_test_split(X1,X2, percent=.75)

CCM = ccm.CCM() #initiate the class

#library lengths to test
len_tr = len(x1tr)
lib_lens = np.arange(10, len_tr, len_tr/20, dtype='int')

#test causation
CCM.fit(x1tr,x2tr)
x1p, x2p = CCM.predict(x1te, x2te,lib_lengths=lib_lens)

sc1,sc2 = CCM.score()

# %%
import matplotlib.pyplot as plt
sc1,sc2

# %%


x = np.arange(0,1000)


fig, ax = plt.subplots()

ax.plot(lib_lens, sc1)
ax.plot(lib_lens, sc2)
plt.title("CCM of Coupled Logistic Data Shows that the 'Causation' is from X2 to X1, which makes sense given the time series plots")
plt.show()


