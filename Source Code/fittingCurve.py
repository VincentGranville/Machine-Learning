import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot

# initializations, define functions

def fpred(x, θ1, θ2, θ4, θ5):
  y = θ1*np.cos(θ2*x)+ θ4*np.cos(θ5*x) 
  return y

def fobs(x,a,b,c,d,e,f):
  y = a*np.cos(b*xobs)+c*np.cos(d*xobs)+e*np.cos(f*xobs)
  return y

n=800
n_training=200  # first n_training points is training set
x=[]
y_obs=[]
y_pred=[]
y_exact=[]

# create data set (observations)

a=0.5 
b=np.sqrt(2)
c=-0.7 
d=2
e=0.2 # noise (e=0 means no noise)
f=np.log(2)

for k in range(n):
  xobs=k/20.0
  x.append(xobs)
  y_obs.append(fobs(xobs,a,b,c,d,e,f))

# curve fitting between f and data, on training set

θ_bounds=((-2.0, -2.5, -1.0, -2.5),(2.0, 2.5, 1.0, 2.5))
θ_start=(0.0, 1.0, 0.0, 1.8)
popt, _ = curve_fit(fpred, x[0:n_training], y_obs[0:n_training],\
    method='trf',bounds=θ_bounds,p0=θ_start) 
θ1, θ2, θ4, θ5 = popt
print('Estimates  : θ1=%.5f θ2=%.5f θ4=%.5f θ5=%.5f' % (θ1, θ2, θ4, θ5))
print('True values: θ1=%.5f θ2=%.5f θ4=%.5f θ5=%.5f' % (a, b, c, d))
print('Initial val: θ1=%.5f θ2=%.5f θ4=%.5f θ5=%.5f' % \
   (θ_start[0], θ_start[1], θ_start[2], θ_start[3]))

# predictions  

for k in range(n):
  xobs=x[k]
  y_pred.append(fpred(xobs, θ1, θ2, θ4, θ5))
  y_exact.append(fpred(xobs, a, b, c, d))

# show plot

pyplot.plot(x, y_obs, color='red')
pyplot.plot(x[0:n_training], y_pred[0:n_training], '--', color='blue')
pyplot.plot(x[n_training:n], y_pred[n_training:n], '--', color='green')
pyplot.show()
