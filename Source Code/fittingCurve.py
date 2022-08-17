import numpy as np
import matplotlib as mpl
from scipy.optimize import curve_fit
from matplotlib import pyplot, rc

# initializations, define functions

def fpred(x, θ1, θ2, θ4, θ5):
  y = θ1*np.cos(θ2*x)+ θ4*np.cos(θ5*x) 
  return y

def fobs(x,a1,a2,a4,a5,a7,a8):    
  y = a1*np.cos(a2*xobs)+a4*np.cos(a5*xobs)+a7*np.cos(a8*xobs)  
  return y

n=800
n_training=200  # first n_training points is training set
x=[]
y_obs=[]
y_pred=[]
y_exact=[]

# create data set (observations)   

a1=0.5 
a2=np.sqrt(2)
a4=-0.7 
a5=2
a7=0.2 # noise (e=0 means no noise)
a8=np.log(2)

for k in range(n):
  xobs=k/20.0
  x.append(xobs)
  y_obs.append(fobs(xobs,a1,a2,a4,a5,a7,a8))  

# curve fitting between f and data, on training set

θ_bounds=((-2.0, -2.5, -1.0, -2.5),(2.0, 2.5, 1.0, 2.5))
θ_start=(0.0, 1.0, 0.0, 1.8)
popt, _ = curve_fit(fpred, x[0:n_training], y_obs[0:n_training],\
    method='trf',bounds=θ_bounds,p0=θ_start) 
θ1, θ2, θ4, θ5 = popt
print('Estimates  : θ1=%.5f θ2=%.5f θ4=%.5f θ5=%.5f' % (θ1, θ2, θ4, θ5))
print('True values: θ1=%.5f θ2=%.5f θ4=%.5f θ5=%.5f' % (a1, a2, a4, a5))        
print('Initial val: θ1=%.5f θ2=%.5f θ4=%.5f θ5=%.5f' % \
   (θ_start[0], θ_start[1], θ_start[2], θ_start[3]))

# predictions  

for k in range(n):
  xobs=x[k]
  y_pred.append(fpred(xobs, θ1, θ2, θ4, θ5))
  y_exact.append(fpred(xobs, a1, a2, a4, a5))    

# show plot

mpl.rcParams['axes.linewidth'] = 0.5
rc('axes',edgecolor='black') # border color
rc('xtick', labelsize=6) # font size, x axis 
rc('ytick', labelsize=6) # font size, y axis
pyplot.scatter(x[0:n_training],y_obs[0:n_training],s=0.5,color='red')
pyplot.scatter(x[n_training:n],y_obs[n_training:n],s=0.5,color='orange')
pyplot.plot(x, y_pred, color='blue',linewidth=0.5)
pyplot.plot(x, y_exact, color='gray',linewidth=0.5)
pyplot.show()
