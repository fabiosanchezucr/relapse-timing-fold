import numpy as np
from exact_model import P
mu,gam,phi0,nu,beta=P['mu'],P['gamma'],P['phi'],P['nu'],P['beta']; rho=0.667
def Mf(a): return 1.0/(1-a*(1-rho))
L0=phi0/mu
TAU=np.linspace(0,40.0/mu,200001); EMU=np.exp(-mu*TAU)

def Phi_reg(k,eps):
    """phi_eps(tau)=c(eps+tau)^{k-1}; Phi=c[(eps+tau)^k-eps^k]/k ; c set so L=L0."""
    if eps==0:
        base=lambda t: t**k/k          # c*(tau^k)/k
    else:
        base=lambda t: ((eps+t)**k-eps**k)/k
    B=base(TAU)
    Lb=mu*np.trapezoid(B*EMU,TAU)      # L for c=1
    c=L0/Lb
    return (lambda t: c*base(t)), c

def stil(a,Phi): return gam*a*np.trapezoid(EMU*np.exp(-a*Phi(TAU)),TAU)
def R0a(a,Phi):
    st=stil(a,Phi); s=1-a-st
    return mu*(1+nu*st)*(a+st)/(a*(mu+gam)*Mf(a)*s) if s>1e-12 else np.nan
def fold(Phi,n=500):
    ag=np.linspace(0.004,0.85,n); v=np.array([R0a(a,Phi) for a in ag])
    i=np.nanargmin(v); return v[i],ag[i]

print("S4: does the unbounded hazard at tau=0 drive the result?")
print("  phi_eps(tau)=c(eps+tau)^{k-1}, k=0.34, c chosen so L=L0 (fixed intensity).")
print(f"{'eps (d)':>9} {'phi(0) /d':>12} {'P(relapse<1d)':>14} {'fold R_c':>10} {'a*':>7}")
k=0.34
for eps in [0.0,0.5,1.0,7.0,30.0,90.0,365.0,1000.0]:
    Phi,c=Phi_reg(k,eps)
    ph0 = np.inf if eps==0 else c*eps**(k-1)
    # prob of relapsing within 1 day at the fold prevalence, given hazard a*phi
    Rc,ast=fold(Phi)
    p1=1-np.exp(-ast*Phi(np.array([1.0]))[0])
    print(f"{eps:9.1f} {ph0:12.4g} {p1:14.4f} {Rc:10.4f} {ast:7.3f}")
print("\nBounded comparison families (all fixed L):")
def Phi_exp(sig):
    A=L0*(1.0/sig+mu); return lambda t: A*sig*(1-np.exp(-t/sig))
def Phi_step(T,hi_frac=0.95):
    # hazard h1 on [0,T], h2 after, with h1 finite; match L
    def mk(h1):
        h2=0.0
        return lambda t: np.where(t<T,h1*t,h1*T)
    # choose h1 so L matches
    f=lambda h1: mu*np.trapezoid(mk(h1)(TAU)*EMU,TAU)-L0
    lo,hi=1e-6,10.0
    for _ in range(200):
        m=0.5*(lo+hi)
        if f(m)>0: hi=m
        else: lo=m
    return mk(0.5*(lo+hi)), 0.5*(lo+hi)
for sig in [180.,365.,730.]:
    Rc,ast=fold(Phi_exp(sig)); print(f"  exponential decay sigma={sig:5.0f}d (phi(0) finite): R_c={Rc:.4f} a*={ast:.3f}")
for T in [90.,365.,730.]:
    Ph,h1=Phi_step(T); Rc,ast=fold(Ph)
    print(f"  step: constant hazard {h1:.5f}/d on first {T:5.0f}d then 0 : R_c={Rc:.4f} a*={ast:.3f}")
print(f"\n  reference: constant phi over all ages          : R_c={fold(lambda t: phi0*t)[0]:.4f}")
