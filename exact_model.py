import numpy as np
from scipy.optimize import brentq, fsolve

# ---- Manuscript's exact classical SAR (scaled, simplex s+a+stil=1) [Eq.1-2] ----
# sdot = mu - beta g(stil) s a - mu s
# adot = beta g(stil) s a + phi stil a - (mu+gamma) a
# stildot = gamma a - phi stil a - mu stil
# g(stil) = kappa/(1+nu stil)
# Behavioral layer -> scalar mixing factor M(a) on the force of infection [Lemma 4.3, Eq.17]
#   static conditional-proportional:  M(a) = 1/(1 - a(1-rho)),  rho=b^a/b^s
# R0 = beta kappa/(mu+gamma); swept via kappa.

P = dict(mu=0.00015, gamma=0.0027, phi=0.0044, nu=0.8, beta=0.009, rho=0.667)

def stil_star(a,p):  return p['gamma']*a/(p['phi']*a+p['mu'])
def s_star(a,p):     return 1 - a - stil_star(a,p)
def abar(p):
    # unique root of s*(a)=0 on (0,1)
    return brentq(lambda a: s_star(a,p), 1e-9, 1-1e-9)

def R_cl(a,p):  # classical branch curve  R0(a) with M=1  [Lemma 5.1]
    st=stil_star(a,p); s=1-a-st
    return (1+p['nu']*st)*((p['mu']+p['gamma'])-p['phi']*st)/((p['mu']+p['gamma'])*s)

def M_static(a,p): return 1.0/(1 - a*(1-p['rho']))

def fold(p, use_M=False):
    ab=abar(p); agrid=np.linspace(1e-5, ab-1e-5, 200000)
    R=R_cl(agrid,p)
    if use_M: R=R/M_static(agrid,p)
    # restrict to region where branch is meaningful (R>0)
    k=np.nanargmin(R)
    return R[k], agrid[k]

Rphi = P['phi']/(P['mu']+P['gamma'])
print(f"R_phi = {Rphi:.3f}  (backward bif requires >1)")
print(f"abar   = {abar(P):.3f}   (paper: 0.431)")
Rc_cl,acl = fold(P, use_M=False)
Rc_b, ab_ = fold(P, use_M=True)
print(f"classical fold  R_c^cl = {Rc_cl:.4f} at a*={acl:.3f}   (paper: 0.9014)")
print(f"adaptive  fold  R_c^b  = {Rc_b:.4f} at a*={ab_:.3f}   (paper: ~0.865-0.866)")
print(f"fold shift dR0 = {Rc_b-Rc_cl:+.4f}   (paper: -0.035 to -0.037)")

# cross-check nu=0 closed-form fold (paper: 0.64955)
P0=dict(P); P0['nu']=0.0
print(f"nu=0 classical fold = {fold(P0)[0]:.5f}   (paper: 0.64955)")
