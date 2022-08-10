import math

epsilon=0.05 
beta=0.45
alpha=1.00 
nMax=5001

Prob={}
Exp={}
Var={}
Prob[(0,0)] =1
Prob[(0,-1)]=0 
Prob[(0,1)] =0 
Prob[(0,-2)]=0 
Prob[(0,2)] =0 

def G(n):
 return(alpha*(n**beta))

def psi(n,m):
  p=0.0
  if m>G(n): 
    p=-1
  if m<-G(n):  
    p=1
  return(p)

Exp[0]=0
Var[0]=0
OUT=open("rndproba.txt","w")
for n in range(1,nMax):
  Exp[n]=0
  Var[n]=0
  delta=0
  for m in range(-n-2,n+3,1):
    Prob[(n,m)]=0
  for m in range(-n,n+1,1):
    Prob[(n,m)]=(0.5+epsilon*psi(n-1,m-1))*Prob[(n-1,m-1)]\
        +(0.5-epsilon*psi(n-1,m+1))*Prob[(n-1,m+1)]
    Exp[n]=Exp[n]+m*Prob[(n,m)]
    Var[n]=Var[n]+m*m*Prob[(n,m)]
    if m>G(n-1) and m<n: 
      delta=delta+8*epsilon*m*Prob[(n-1,m)]
  var1=Var[n]
  var2=Var[n-1]+1-delta
  string1=("%5d %.6f %.6f %.6f" % (n,var1,var2,delta))
  string2=("%5d\t%.6f\t%.6f\t%.6f\n" % (n,var1,var2,delta))
  print(string1)
  OUT.write(string2) 
OUT.close() 
