import numpy as np, time
from chain_v3 import build
from chain_v2 import mu,gam,nu,beta,rho,Mf,kap_R0

def run(ch,kap,a0,N0,rng,thr,Tmax=2e4):
    N=ch['N']; w=ch['w']; ph=ch['ph']
    A=int(round(a0*N0)); S=N0-A; R=np.zeros(N); t=0.0
    nr=2+2*N+3+N
    rates=np.zeros(2+N+N+3+N)
    om=2; orl=2+N; od=orl+N
    while t<Tmax:
        Rs=R.sum(); Np=S+A+Rs
        if A==0: return 0
        if A>thr*Np: return 1
        aN=A/Np; g=kap/(1+nu*Rs/Np); M=1.0/(1-aN*(1-rho))
        rates[0]=beta*g*M*S*A/Np
        rates[1]=gam*A
        rates[om:om+N]=w*R                 # maturation out of each bin
        rates[orl:orl+N]=ph*aN*R           # relapse from each bin
        rates[od]=mu*Np; rates[od+1]=mu*S; rates[od+2]=mu*A
        rates[od+3:od+3+N]=mu*R
        tot=rates.sum()
        if tot<=0: return 0
        t+=rng.exponential(1.0/tot)
        k=np.searchsorted(np.cumsum(rates),rng.random()*tot)
        if k==0: S-=1; A+=1
        elif k==1: A-=1; R[0]+=1
        elif k<orl:
            i=k-om
            if i==N-1: R[i]-=1          # ages out of the tracked window (still reformed->exits)
            else: R[i]-=1; R[i+1]+=1
        elif k<od: R[k-orl]-=1; A+=1
        elif k==od: S+=1
        elif k==od+1: S-=1
        elif k==od+2: A-=1
        else: R[k-od-3]-=1
    return 1 if A>0.05*(S+A+R.sum()) else 0

def frac(kind,Nstage,a0,n,thr,N0=1500,seed=0):
    ch=build(kind,Nstage); kap=kap_R0(0.95); rng=np.random.default_rng(seed)
    return np.mean([run(ch,kap,a0,N0,rng,thr) for _ in range(n)])
def wilson(p,n,z=1.96):
    d=1+z*z/n; c=p+z*z/(2*n); h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)); return ((c-h)/d,(c+h)/d)

