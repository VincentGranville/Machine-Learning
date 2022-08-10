import random
import math
random.seed(1) 

n=50000
Nsample=1
deviations='Large'
mode='Power' 

if deviations=='Large':
  eps=0.01
  beta=0.54
  alpha=0.3
elif deviations=='Small':
  eps=0.05
  beta=0.35 #beta = 1 for log
  alpha=1

def G(n):
  if mode=='Power':
    return(alpha*(n**beta))
  elif mode=='Log' and n>0:
    return(alpha*(math.log(n)**beta))
  else:
    return(0)

OUT=open("rndtest.txt","w")
for sample in range(Nsample):
  print("Sample: ",sample)
  S=0
  for k in range(1,n):
    x=1
    rnd=random.random()
    M=G(k)
    if deviations=='Large':
      if ((S>=-M and S<0 and rnd<0.5+eps) or (S<=M and S>0 and rnd<0.5-eps) or 
        (abs(S)>=M and rnd<0.5) or (S==0 and rnd<0.5)):
        x=-1
    elif deviations=='Small':
      if (S<-M and rnd<0.5-eps) or (S>M and rnd<0.5+eps) or (abs(S)<=M and rnd<0.5):
        x=-1
    print(k,M,S,x)
    S=S+x
    line=str(sample)+"\t"+str(k)+"\t"+str(S)+"\t"+str(x)+"\n" 
    OUT.write(line)   
OUT.close()      
