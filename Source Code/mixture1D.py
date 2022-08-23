import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm 
from sklearn.cluster import KMeans

N_tests=5    # number of data sets being tested
n_A = 1500  # number of points in cluster A
n_B = 8500  # number of points in cluster B
n = n_A + n_B
Ones = np.ones((n)) # array with 1's
p_A = 3
p_B = 1
np.random.seed(438713)
min_θ_A =  99999999
min_θ_B =  99999999
max_θ_A = -99999999
max_θ_B = -99999999
CR_x=[]  # confidence region for (best_θ_A, best_θ_A), 1st coordinate  
CR_y=[]  # confidence region for (best_θ_A, best_θ_A), 2nd coordinate 

def compute_MSE(θ_A, θ_B, p_A, p_B, W):
    n = W.size
    MSE = (1/n) * np.sum((abs(W - θ_A * Ones)**p_A) * (abs(W - θ_B * Ones)**p_B))
    return MSE

for sample in range(N_tests):   # new dataset at each iteration

    # W_A  = np.random.normal(0.5, 2, size=n_A)
    # W_B  = 1 + np.random.gamma(8, 5, size=n_B)/4
    W_A  = np.random.normal(0.5, 0.1, size=n_A)
    W_B  = np.random.normal(1.0, 0.2, size=n_B)
    W    = np.concatenate((W_A, W_B))
    min_MSE=99999999
    print('Sample %1d:' %(sample))

    for iter in range(10000):

        θ_A = np.amin(W) + (np.amax(W)-np.amin(W))*np.random.rand()
        θ_B = np.amin(W) + (np.amax(W)-np.amin(W))*np.random.rand()
        MSE = compute_MSE(θ_A, θ_B, p_A, p_B, W)   # MSE for my method
        if MSE < min_MSE:
            min_MSE=MSE
            print('Iter = %5d  θ_A = %+8.4f  θ_B = %+8.4f  MSE = %+12.4f' %(iter,θ_A ,θ_B, MSE))
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

    # get centers from Kmeans method (for comparison purposes)  
    V    = W.copy()  
    km = KMeans(n_clusters=2) 
    km.fit(V.reshape(-1,1))   
    centers_kmeans=km.cluster_centers_ 
    kmeans_A=min(centers_kmeans[0,0],centers_kmeans[1,0])
    kmeans_B=max(centers_kmeans[0,0],centers_kmeans[1,0])

    MSE_kmeans = compute_MSE(centers_kmeans[0,0], centers_kmeans[1,0], p_A, p_B, V)  
    centroid=(1/n)*np.sum(W) 
    centroid_A=(1/n_A)*np.sum(W_A) 
    centroid_B=(1/n_B)*np.sum(W_B) 
    median_A=np.median(W_A)
    median_B=np.median(W_B)
    MSE_base = compute_MSE(centroid, centroid, p_A, p_B, W)  # MSE for base model
    MSE_tc1 = compute_MSE(centroid_A, centroid_B, p_A, p_B, W)  
    MSE_tc2 = compute_MSE(centroid_B, centroid_A, p_A, p_B, W)  
    MSE_true_centers = min(MSE_tc1,MSE_tc2)  
    MSE_tm1 = compute_MSE(median_A, median_B, p_A, p_B, W)  
    MSE_tm2 = compute_MSE(median_B, median_A, p_A, p_B, W)  
    MSE_true_medians = min(MSE_tm1,MSE_tm2)  # MSE for base model

    print('True centers  θ_A = %+8.4f  θ_B = %+8.4f  MSE = %+12.4f' %(centroid_A,centroid_B,MSE_true_centers)) 
    print('model (%1d,%1d)   θ_A = %+8.4f  θ_B = %+8.4f  MSE = %+12.4f' %(p_A,p_B,best_θ_A,best_θ_B,min_MSE)) 
    print('Kmeans        θ_A = %+8.4f  θ_B = %+8.4f  MSE = %+12.4f' %(kmeans_A,kmeans_B,MSE_kmeans)) 
    print('True medians  θ_A = %+8.4f  θ_B = %+8.4f  MSE = %+12.4f' %(median_A,median_B,MSE_true_medians)) 
    print('Base          θ_A = %+8.4f  θ_B = %+8.4f  MSE = %+12.4f' %(centroid,centroid,MSE_base)) 
    print()
 
print('95 %% range for min(θ_A, θ_B): [%+8.4f, %+8.4f]' %(min_θ_A ,max_θ_A))
print('95 %% range for max(θ_A, θ_B): [%+8.4f, %+8.4f]' %(min_θ_B ,max_θ_B))

# intialize plotting parameters
plt.rcParams['axes.linewidth'] = 0.2
plt.rc('axes',edgecolor='black') # border color
plt.rc('xtick', labelsize=7) # font size, x axis  
plt.rc('ytick', labelsize=7) # font size, y axis

# plotting histogram and density
bins=np.linspace(min(W), max(W), num=100)
plt.hist(W_A, color = "blue", alpha=0.2, edgecolor='blue',bins=bins) 
plt.hist(W_B, color = "red", alpha=0.3, edgecolor='red',bins=bins) 
plt.hist(W, color = "green", alpha=0.1, edgecolor='green',bins=bins) 
# plt.plot(bins, 8*norm.pdf(bins,0.5,0.3),color='blue',linewidth=0.6) 
# plt.plot(bins, 12*norm.pdf(bins,1,0.2),color='red',linewidth=0.6) 
plt.show()

# plotting confidence region
if N_tests > 50:
    plt.scatter(CR_x,CR_y,s=6,alpha=0.3) 
    plt.show() 
