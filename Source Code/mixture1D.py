import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm 

n_A = 400
n_B = 600
n = n_A + n_B
Ones = np.ones((n)) # array with 1's
p = 1  # p = 1 works a lot better than p = 2!
np.random.seed(438713)
min_θ_A =  99999999
min_θ_B =  99999999
max_θ_A = -99999999
max_θ_B = -99999999
CR_x=[]  # confidence region for (best_θ_A, best_θ_A), 1st coordinate  
CR_y=[]  # confidence region for (best_θ_A, best_θ_A), 2nd coordinate 

for sample in range(2):

    W_A  = np.random.normal(0.5, 0.3, size=n_A)
    W_B  = np.random.normal(1.0, 0.2, size=n_B)
    W    = np.concatenate((W_A, W_B))
    minError=99999999
    print('Sample %1d:' %(sample))

    for iter in range(100000):

        θ_A = -1 + 3*np.random.rand()
        θ_B = -1 + 3*np.random.rand()
        Error = (1/n) * np.sum(abs( (W - θ_A * Ones) * (W - θ_B * Ones) )**p)
        if Error < minError:
            minError=Error
            print('Iter = %5d  θ_A = %+.3f  θ_B = %+.3f  Error = %+.3f' %(iter,θ_A ,θ_B, Error))
            best_θ_A = min(θ_A, θ_B)
            best_θ_B = max(θ_A, θ_B)

    if best_θ_A < min_θ_A:
        min_θ_A = best_θ_A
    if best_θ_A > max_θ_A:
        max_θ_A = best_θ_A
    if best_θ_B < min_θ_B:
        min_θ_B = best_θ_B
    if best_θ_B > max_θ_B:
        max_θ_B = best_θ_B
    CR_x.append(best_θ_A) 
    CR_y.append(best_θ_B) 
    print()
 
print('95 %% range for min(θ_A, θ_B): [%+.5f, %+.5f]' %(min_θ_A ,max_θ_A))
print('95 %% range for max(θ_A, θ_B): [%+.5f, %+.5f]' %(min_θ_B ,max_θ_B))

# intialize plotting parameters
plt.rcParams['axes.linewidth'] = 0.2
plt.rc('axes',edgecolor='black') # border color
plt.rc('xtick', labelsize=7) # font size, x axis  
plt.rc('ytick', labelsize=7) # font size, y axis

# plotting histogram and density
bins=np.linspace(min(W), max(W), num=100)
plt.hist(W_A, color = "blue", alpha=0.3, edgecolor='blue',bins=bins) 
plt.hist(W_B, color = "red", alpha=0.3, edgecolor='red',bins=bins) 
plt.plot(bins, 8*norm.pdf(bins,0.5,0.3),color='blue',linewidth=0.6) 
plt.plot(bins, 12*norm.pdf(bins,1,0.2),color='red',linewidth=0.6) 
plt.show()

# plotting confidence region
plt.scatter(CR_x,CR_y,s=6,alpha=0.3) 
plt.show() 

