import numpy as np, json, os
from gill_core import run
from chain_v3 import build
from chain_v2 import kap_R0
def frac(kind,Ns,a0,n,thr,hmin,N0=1500,seed=0):
    ch=build(kind,Ns,h_min=hmin); kap=kap_R0(0.95); rng=np.random.default_rng(seed)
    return float(np.mean([run(ch,kap,a0,N0,rng,thr,Tmax=2e4) for _ in range(n)]))
def wil(p,n,z=1.96):
    d=1+z*z/n; c=p+z*z/(2*n); h=z*np.sqrt(max(p*(1-p),0)/n+z*z/(4*n*n)); return ((c-h)/d,(c+h)/d)
def save(key,val,path='gres.json'):
    d=json.load(open(path)) if os.path.exists(path) else {}
    d[key]=val; json.dump(d,open(path,'w'),indent=1)
