import sys, time, json, os, numpy as np
from g_lib import frac, wil
a0=float(sys.argv[1]); n=500; t0=time.time()
path='g500.json'
d=json.load(open(path)) if os.path.exists(path) else {}
for k in ["early","const","late"]:
    p=frac(k,20,a0,n,0.17,400.,seed=abs(hash((k,a0,'v2')))%2**31)
    lo,hi=wil(p,n); d[f"{k}|{a0}"]=[p,lo,hi]
    print(f"  {k}: {p:.3f} [{lo:.3f},{hi:.3f}]  ({time.time()-t0:.0f}s)",flush=True)
json.dump(d,open(path,'w'),indent=1)
print(f"a0={a0} done")
