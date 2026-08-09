import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from exact_model import P
mu,gam,phi0,nu,beta=P['mu'],P['gamma'],P['phi'],P['nu'],P['beta']; rho=0.667
def Mf(a): return 1.0/(1-a*(1-rho))
def kap_R0(R0): return R0*(mu+gam)/beta
L0=phi0/mu                      # exact lifetime relapse intensity of the baseline

# ---------------- profiles: ANALYTIC Phi and analytic L on [0,inf) ----------------
# early : phi = A e^{-tau/sigma}          -> L = A/(1/sigma+mu),  Phi = A*sigma(1-e^{-tau/sigma})
# const : phi = phi0                      -> L = phi0/mu,         Phi = phi0*tau
# late  : phi = A(1-e^{-tau/sigma})       -> L = A[1/mu - sigma/(1+mu*sigma)]
def make_profile(kind,sigma=400.0,L=L0):
    if kind=="const":
        A=L*mu
        return (lambda t: A*t), (lambda t: A*np.ones_like(t)), A
    if kind=="early":
        A=L*(1.0/sigma+mu)
        return (lambda t: A*sigma*(1-np.exp(-t/sigma))), (lambda t: A*np.exp(-t/sigma)), A
    if kind=="late":
        sg=2500.0; A=L/(1.0/mu-sg/(1+mu*sg))
        return (lambda t: A*(t-sg*(1-np.exp(-t/sg)))), (lambda t: A*(1-np.exp(-t/sg))), A
    raise ValueError

# ---------------- EXACT continuum equilibrium (gold standard) ----------------
TAU=np.linspace(0,60.0/mu,400001)          # 60/mu: truncation error ~ e^{-60}
def stil_exact(a,Phi):
    return gam*a*np.trapezoid(np.exp(-mu*TAU-a*Phi(TAU)),TAU)
def R0_of_a(a,Phi):
    st=stil_exact(a,Phi); s=1-a-st
    return mu*(1+nu*st)*(a+st)/(a*(mu+gam)*Mf(a)*s) if s>1e-12 else np.nan
def exact_upper_a(R0,Phi,lo=1e-4,hi=0.98):
    # upper branch: largest root of R0_of_a(a)=R0
    ag=np.linspace(lo,hi,4000); vals=np.array([R0_of_a(a,Phi) for a in ag])
    ok=np.isfinite(vals); ag,vals=ag[ok],vals[ok]
    d=vals-R0; idx=np.where(np.sign(d[:-1])!=np.sign(d[1:]))[0]
    if len(idx)==0: return None
    i=idx[-1]
    return brentq(lambda a: R0_of_a(a,Phi)-R0, ag[i], ag[i+1])

# ---------------- corrected chain: discretize Phi (bin-average hazard) ----------
def build(kind,N,span_mu=15.0,sigma=400.0):
    Phi,phi,_=make_profile(kind,sigma)
    T=span_mu/mu; h=T/N; w=1.0/h
    edges=np.arange(N+1)*h
    phihat=(Phi(edges[1:])-Phi(edges[:-1]))/h        # <-- exact cumulative-hazard increments
    return dict(N=N,w=w,ph=phihat,Phi=Phi)

def rhs(t,y,ch,kap):
    N,w,ph=ch['N'],ch['w'],ch['ph']
    s,a=y[0],y[1]; R=y[2:2+N]; st=R.sum()
    F=beta*(kap/(1+nu*st))*Mf(a)*s*a; rel=a*np.dot(ph,R)
    ds=mu-F-mu*s; da=F+rel-(mu+gam)*a
    dR=np.empty(N)
    dR[0]=gam*a-(w+mu+ph[0]*a)*R[0]
    dR[1:N-1]=w*R[0:N-2]-(w+mu+ph[1:N-1]*a)*R[1:N-1]
    dR[N-1]=w*R[N-2]-(w+mu+ph[N-1]*a)*R[N-1]   # last stage also exits (no pile-up)
    return np.concatenate(([ds,da],dR))

def chain_a(ch,kap,a0=0.5,T=4e6):
    N=ch['N']; y0=np.zeros(2+N); y0[0]=1-a0; y0[1]=a0
    s=solve_ivp(rhs,[0,T],y0,args=(ch,kap),method="LSODA",rtol=1e-10,atol=1e-13)
    return s.y[1,-1]

if __name__=="__main__":
    R0=1.1; kap=kap_R0(R0)
    print("CORRECTED SCHEME: chain equilibrium a* vs EXACT continuum value")
    print(f"{'profile':8s} {'exact a*':>10} | " + " ".join(f"N={n:<5d}" for n in [20,40,80,160,320]))
    for kind in ["const","early","late"]:
        Phi,_,_=make_profile(kind)
        ex=exact_upper_a(R0,Phi)
        row=[]
        for N in [20,40,80,160,320]:
            row.append(chain_a(build(kind,N),kap))
        print(f"{kind:8s} {ex:10.5f} | " + " ".join(f"{v:7.5f}" for v in row))
    print("\nSpan-independence check (N=320):")
    for kind in ["const","early","late"]:
        vals=[chain_a(build(kind,320,span_mu=sm),kap) for sm in [10.,15.,25.,40.]]
        print(f"  {kind:6s} span/mu=10,15,25,40 -> " + " ".join(f"{v:.5f}" for v in vals))
