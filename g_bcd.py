import time
from g_lib import frac,save
t0=time.time(); n=60; K=["early","const","late"]
print("(b) MESH SENSITIVITY [a0=0.07, thr=0.17]",flush=True)
b={}
for Ns,hm in [(20,400.),(40,200.),(60,100.)]:
    v=[frac(k,Ns,0.07,n,0.17,hm,seed=abs(hash((k,Ns,'m')))%2**31) for k in K]
    b[f"N{Ns}"]=v; print(f"  N={Ns:2d},h_min={hm:5.0f}: early={v[0]:.2f} const={v[1]:.2f} late={v[2]:.2f}  [{time.time()-t0:.0f}s]",flush=True)
save("mesh",b)
print("(c) THRESHOLD SENSITIVITY [N=20,a0=0.07]",flush=True)
c={}
for thr in [0.12,0.17,0.25]:
    v=[frac(k,20,0.07,n,thr,400.,seed=abs(hash((k,thr)))%2**31) for k in K]
    c[str(thr)]=v; print(f"  thr={thr:.2f}: early={v[0]:.2f} const={v[1]:.2f} late={v[2]:.2f}  [{time.time()-t0:.0f}s]",flush=True)
save("threshold",c)
print("(d) FINITE-SIZE [N=20,a0=0.07,thr=0.17]",flush=True)
d={}
for N0 in [750,1500,3000]:
    v=[frac(k,20,0.07,n,0.17,400.,N0=N0,seed=abs(hash((k,N0)))%2**31) for k in K]
    d[str(N0)]=v; print(f"  N0={N0:4d}: early={v[0]:.2f} const={v[1]:.2f} late={v[2]:.2f}  [{time.time()-t0:.0f}s]",flush=True)
save("finitesize",d)
