import numpy as np
from scipy.integrate import solve_ivp
from chain_v2 import (mu,gam,phi0,nu,beta,rho,Mf,kap_R0,make_profile,
                      exact_upper_a,stil_exact,R0_of_a)

# GRADED age mesh: geometric from h_min near tau=0 out to span, so a 400-d feature
# and a 100,000-d domain are both resolved with modest N.
def graded_edges(N,span,h_min):
    # geometric ratio r solving h_min*(r^N-1)/(r-1)=span
    lo,hi=1.0+1e-12,2.0
    f=lambda r: h_min*(r**N-1)/(r-1)-span
    if f(1.0000001)>0: r=1.0
    else:
        for _ in range(200):
            mid=0.5*(lo+hi)
            if f(mid)>0: hi=mid
            else: lo=mid
        r=0.5*(lo+hi)
    h=h_min*r**np.arange(N)
    return np.concatenate(([0.0],np.cumsum(h))), h

def build(kind,N,span_mu=15.0,h_min=10.0,sigma=400.0):
    Phi,_,_=make_profile(kind,sigma)
    span=span_mu/mu
    edges,h=graded_edges(N,span,h_min)
    phihat=(Phi(edges[1:])-Phi(edges[:-1]))/h     # discretize Phi, not phi
    return dict(N=N,w=1.0/h,ph=phihat,edges=edges)

def rhs(t,y,ch,kap):
    N=ch['N']; w=ch['w']; ph=ch['ph']
    s,a=y[0],y[1]; R=y[2:2+N]; st=R.sum()
    F=beta*(kap/(1+nu*st))*Mf(a)*s*a; rel=a*np.dot(ph,R)
    ds=mu-F-mu*s; da=F+rel-(mu+gam)*a
    out=w*R                      # exit rate from each bin (variable width)
    dR=np.empty(N)
    dR[0]=gam*a-out[0]-(mu+ph[0]*a)*R[0]
    dR[1:]=out[:-1]-out[1:]-(mu+ph[1:]*a)*R[1:]
    return np.concatenate(([ds,da],dR))

def chain_a(ch,kap,a0=0.5,T=4e6):
    N=ch['N']; y0=np.zeros(2+N); y0[0]=1-a0; y0[1]=a0
    s=solve_ivp(rhs,[0,T],y0,args=(ch,kap),method="LSODA",rtol=1e-10,atol=1e-13)
    return s.y[1,-1]

if __name__=="__main__":
    R0=1.1; kap=kap_R0(R0)
    print("GRADED-MESH SCHEME: chain a* vs EXACT continuum (R0=1.1)")
    Ns=[40,80,160,320]
    print(f"{'profile':7s} {'exact':>9} | " + " ".join(f"N={n:<7d}" for n in Ns) + "  rel.err@max N")
    for kind in ["const","early","late"]:
        Phi,_,_=make_profile(kind); ex=exact_upper_a(R0,Phi)
        row=[chain_a(build(kind,N),kap) for N in Ns]
        print(f"{kind:7s} {ex:9.5f} | " + " ".join(f"{v:9.5f}" for v in row) +
              f"   {abs(row[-1]-ex)/ex*100:.3f}%")
    print("\nSpan-independence (N=320, graded):")
    for kind in ["const","early","late"]:
        vals=[chain_a(build(kind,320,span_mu=sm),kap) for sm in [10.,15.,25.,40.]]
        print(f"  {kind:6s} -> " + " ".join(f"{v:.5f}" for v in vals))
    print("\nh_min-independence (N=320, span=15/mu):")
    for kind in ["early"]:
        vals=[chain_a(build(kind,320,h_min=hm),kap) for hm in [40.,20.,10.,5.]]
        print(f"  {kind:6s} h_min=40,20,10,5 -> " + " ".join(f"{v:.5f}" for v in vals))
