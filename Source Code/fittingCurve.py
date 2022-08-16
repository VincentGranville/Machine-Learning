import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot

def f(x, θ0, θ1, θ2, θ4, θ5):
  θ0=1
  y = θ0 + θ1*np.cos(θ2*x) + θ4*np.cos(θ5*x) 
  return y

n=200
x=[]
y=[]

# create data set (observations)
a=2
b=np.sqrt(2)
c=np.sqrt(5) # np.sqrt(61)/2
for k in range(n):
  xobs=k/20.0
  x.append(xobs)
  yobs=1+0.5*np.cos(a*xobs)-0.7*np.cos(b*xobs) ### +0.05*np.cos(c*xobs)
  y.append(yobs)
  print(xobs,yobs)

# curve fit between f and data
popt, _ = curve_fit(f, x, y)
θ0, θ1, θ2, θ4, θ5 = popt
print('θ0=%.5f θ1=%.5f θ2=%.5f θ4=%.5f θ5=%.5f' % (θ0, θ1, θ2, θ4, θ5))

# create and output plot
pyplot.scatter(x, y)
pyplot.plot(x, y, '--', color='red')
pyplot.show()
