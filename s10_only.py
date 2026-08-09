import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.special import gamma as G
from exact_model import P
mu,gam,phi0,nu,beta=P['mu'],P['gamma'],P['phi'],P['nu'],P['beta']
TAU=np.linspace(0,40.0/mu,60001); EMU=np.exp(-mu*TAU)
L0=phi0/mu
def Mrho(a,r): return 1.0/(1-a*(1-r))
def lam_of(k,L=L0): return (1.0/mu)*(G(k+1)/L)**(1.0/k)
def stil_k(a,k,L=L0): return gam*a*np.trapezoid(EMU*np.exp(-a*(TAU/lam_of(k,L))**k),TAU)
def stil_const(a): return gam*a/(mu+phi0*a)
def R0(a,st,r=0.667,MU=mu,GAM=gam,NU=nu):
    s=1-a-st
    return MU*(1+NU*st)*(a+st)/(a*(MU+GAM)*Mrho(a,r)*s) if s>1e-12 else np.nan
AG=np.linspace(0.004,0.85,340)
def fold_k(k,r=0.667): 
    v=np.array([R0(a,stil_k(a,k),r) for a in AG]); i=np.nanargmin(v); return v[i],AG[i]
def fold_const(r=0.667):
    v=np.array([R0(a,stil_const(a),r) for a in AG]); return np.nanmin(v)

INK="#1b1b1b"; ACC="#c1440e"; BLU="#2f6690"; GRN="#3a7d44"
plt.rcParams.update({"font.family":"DejaVu Sans","axes.edgecolor":"#444","axes.linewidth":0.8,
 "axes.grid":True,"grid.color":"#ececec","axes.axisbelow":True,"font.size":10.5})

# ---------- S10: robustness surface -- enlargement ratio vs k, across parameters ----------
def fold_generic(k,MU,GAM,PH,NU,r):
    L=PH/MU; lam=(1.0/MU)*(G(k+1)/L)**(1.0/k)
    TA=np.linspace(0,40.0/MU,40001); E=np.exp(-MU*TA)
    def st(a): return GAM*a*np.trapezoid(E*np.exp(-a*(TA/lam)**k),TA)
    def stc(a): return GAM*a/(MU+PH*a)
    v=np.array([R0(a,st(a),r,MU,GAM,NU) for a in AG]); i1=np.nanargmin(v)
    vc=np.array([R0(a,stc(a),r,MU,GAM,NU) for a in AG]); i2=np.nanargmin(vc)
    # interior minimum required: otherwise the branch is monotone (no fold)
    if i1 in (0,len(AG)-1) or i2 in (0,len(AG)-1): return np.nan
    return vc[i2]/v[i1]
variants=[("baseline",dict()),(r"$\nu\times0.5$",dict(NU=nu*0.5)),(r"$\rho=1$",dict(r=1.0)),
          (r"$\phi\times0.5$",dict(PH=phi0*0.5)),(r"$\phi\times2$",dict(PH=phi0*2)),
          (r"$\gamma\times0.5$",dict(GAM=gam*0.5)),(r"$\gamma\times2$",dict(GAM=gam*2)),
          (r"$\mu\times0.5$",dict(MU=mu*0.5)),(r"$\mu\times2$",dict(MU=mu*2))]
kk=np.array([0.30,0.40,0.50,0.65,0.80,0.90])
fig,ax=plt.subplots(figsize=(7.6,4.8)); fig.subplots_adjust(left=0.11,right=0.97,top=0.9,bottom=0.13)
cmap=plt.cm.viridis(np.linspace(0,0.9,len(variants)))
for (lab,kw),c in zip(variants,cmap):
    base=dict(MU=mu,GAM=gam,PH=phi0,NU=nu,r=0.667); base.update(kw)
    ys=np.array([fold_generic(k,**base) for k in kk])
    m=np.isfinite(ys)
    ax.plot(kk[m],ys[m],"o-",color=c,lw=1.6,ms=4,label=lab)
ax.axhline(1.0,color="#444",lw=1.2,ls="--")
ax.set_yscale("log"); ax.set_xlabel(r"relapse-hazard shape  $k$")
ax.set_ylabel(r"basin enlargement  $\mathcal{R}_c^{\rm const}/\mathcal{R}_c(k)$")
ax.set_title("Enlargement across the whole $k<1$ range and all parameters",loc="left",fontweight="bold",color=INK)
ax.legend(frameon=False,fontsize=8,ncol=3,loc="upper right")
fig.savefig("figs/robust_surface.pdf",facecolor="white")
fig.savefig("robust_surface.png",dpi=160,facecolor="white")
print("robustness surface done")
