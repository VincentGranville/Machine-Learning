# Kriging-style spatial regression / inverse distance interpolation
#
# Highlights of this "fuzzy regression" code:
#
# 1. Model-free; produces big output file to compute prediction intervals
# 2. Bivariate case, featuring nearest-neighbor approach (the weights)
# 3. Math-free (no matrix algrebra, square root or calculus)
# 4. Statistics-free (no statistical science involved at all)
# 5. Requires no technical knowledge beyond high school, but far from trivial!
# 6. Exact predictions for training set, yet robust (no overfitting)
# 7. Increasing M is "lazy way" to boost performance, but it slows speed
#
# By Vincent Granville, www.MLTechniques.com

import numpy as np
import math
import random
random.seed(100)

# Hyperparameters

# n (number of obs) set after reading input file [n=1000 here]

P=0.8         # proportion of data allocated to training
              # the remaining is for validation
M=5000        # max number of splines used per point
              # M=5000 offers modest gain over M=800
r=2           # number of points defining a spline
              # also works with r=1 or larger r
smoother=1.5  # smoothing param used in weighted predictions 
              # try 0.5 for more smoothing (0 = max smoothing)
thresh1=25.0  # max distance allowed to nearby spline  
              # increase to eliminate points with no predictions
              # decrease to narrow (improve) confidence intervals
thresh2=1.5   # max outlier level allowed for predicted values 
              # if < 1, predicted can't be more extreme than observed
              # if too low, may increase number of points with no prediction
              # if too large, may produce a few strong outlier predictions
thresh3=0.001 # control numerical stability (keep 0.001)

# Global indicators (defined later)
#
# missing       # number of points not assigned a prediction

# Output var (defined later) 
#
# count       # actual number of splines used for a specific point
# error       # code telling why a point is not assigned a prediction
# weight      # weight assigned to a spline, for a given point
# zpred       # predicted value for a point zz = (xx, yy)
# zpredw      # weighted predicted value

# Input var (defined later)
#
# xx, yy, zz: coordinates of a point 

#----------------------------------------------------
# Reading input file 

x=[]
y=[]
z=[]

file=open('fuzzy2b.txt',"r")
lines=file.readlines()
for aux in lines:
    x.append(aux.split('\t')[0])
    y.append(aux.split('\t')[1])
    z.append(aux.split('\t')[2])
file.close()

x = list(map(float, x))
y = list(map(float, y))
z = list(map(float, z))

zmin=np.min(z)
zmax=np.max(z)
zavg=np.mean(z)
zdev=max(abs(zmin-zavg),abs(zmax-zavg))

n=len(x)

#----------------------------------------------------
def F(xx,yy,r):

  zz=0 
  distmin=1
  error=0

  idx=[]
  A=[]
  B=[]

  for i in range(0,r):
    idx.insert(i,int(n*P*random.random()))

  prod=1.0;
  for i in range(0,r): 
    for j in range(i+1,r): 
      prod*=(x[idx[i]]-x[idx[j]])*(y[idx[i]]-y[idx[j]])
  if abs(prod)>thresh3:
    for i in range(0,r): 
      A.insert(i,1.0)
      B.insert(i,1.0)
      for j in range(0,r): 
        if j != i:
          A[i]*=(xx-x[idx[j]])/(x[idx[i]]-x[idx[j]])
          B[i]*=(yy-y[idx[j]])/(y[idx[i]]-y[idx[j]])
      zz+=z[idx[i]]*(A[i]+B[i])/2
      distmin*=max(abs(xx-x[idx[i]]),abs(yy-y[idx[i]]))
    distmin=pow(distmin,1/r)
  else:
    error=1; 
 
  return [zz,distmin,error]

#----------------------------------------------------
# Main step: predictions for points in validation set
# For training set points, change range(int(P*n),n) to range(0,int(P*n))

file_small=open("fuzzy_small.txt","w")
file_big=open("fuzzy_big.txt","w")

for j in range(int(P*n),n): # loop over all validation points 

  xx=x[j]
  yy=y[j]
  zobs=z[j]
  count=0
  missing=0
  sweight=0.0
  zpredw=0.0
  zpred=0.0

  for k in range(0,M): # inner loop over all splines

    list=F(xx,yy,r)
    zz=list[0]
    distmin=list[1]
    error=list[2]
    weight=math.exp(-smoother*distmin)  
    zzdevratio=abs(zz-zavg)/zdev

    if distmin<thresh1 and zzdevratio<thresh2 and error==0: 
      count+=1
      sweight+=weight
      zpredw+=zz*weight
      zpred+=zz
      row=[j,xx,yy,zobs,zz,distmin,weight,zzdevratio]
      for field in row:
        file_big.write(str(field)+"\t")
      file_big.write("\n")

  if count>0:
    zpredw=zpredw/sweight
    zpred=zpred/count
  else:
    missing+=1
    zpredw=""
    zpred=""

  row=[j,count,xx,yy,zobs,zpred,zpredw] 
  for field in row:
    file_small.write(str(field)+"\t")
  file_small.write("\n")

file_big.close()
file_small.close()
print(missing,"ignored points\n")

