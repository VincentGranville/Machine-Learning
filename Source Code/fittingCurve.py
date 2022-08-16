import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot

a=2
b=np.sqrt(2)
c=np.log(2)

def f(x, θ1, θ2, θ4, θ5):
  y = θ1*np.cos(θ2*x)+ θ4*np.cos(θ5*x) 
  return y

def fobs(x):
  y = 0.5*np.cos(a*xobs)-0.7*np.cos(b*xobs)+0.001*np.cos(c*xobs)
  return y

n=200
x=[]
y=[]
ypred=[]

# create data set (observations)
a=2
b=np.sqrt(2)
c=np.sqrt(5) # np.sqrt(61)/2
for k in range(n):
  xobs=k/20.0
  x.append(xobs)
  y.append(fobs(xobs))
  ## print(xobs,y)

# curve fit between f and data
popt, _ = curve_fit(f, x, y)
θ1, θ2, θ4, θ5 = popt
print('θ1=%.5f θ2=%.5f θ4=%.5f θ5=%.5f' % (θ1, θ2, θ4, θ5))

# add points outside the training set, to the right 
for k in range(2*n):
  xobs=k/20.0
  ypred.append(f(xobs, θ1, θ2, θ4, θ5))
  if k>=n:
    x.append(xobs)
    y.append(fobs(xobs))

# create and output plot
pyplot.scatter(x, y)
pyplot.plot(x, y, '--', color='red')
pyplot.plot(x, ypred, '--', color='blue')
pyplot.show()
