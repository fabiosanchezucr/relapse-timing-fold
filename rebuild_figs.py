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

# ---------- joint fold map (sigma-equivalent: use k as the timing axis) ----------
ks=np.linspace(0.28,1.0,22); rhos=np.linspace(0.5,1.0,18)
Z=np.zeros((len(rhos),len(ks))); Ast=np.zeros(len(ks))
for j,k in enumerate(ks):
    st=np.array([stil_k(a,k) for a in AG])
    for i,r in enumerate(rhos):
        v=np.array([R0(a,st[m],r) for m,a in enumerate(AG)]); Z[i,j]=np.nanmin(v)
    v0=np.array([R0(a,st[m],0.667) for m,a in enumerate(AG)]); Ast[j]=AG[np.nanargmin(v0)]
fig,(a1,a2)=plt.subplots(1,2,figsize=(11.4,4.5))
fig.subplots_adjust(left=0.07,right=0.985,wspace=0.3,top=0.87,bottom=0.15)
cf=a1.contourf(ks,rhos,Z,levels=np.linspace(0,1.0,21),cmap="viridis")
cs=a1.contour(ks,rhos,Z,levels=[0.1,0.3,0.5,0.7],colors="w",linewidths=0.8); a1.clabel(cs,fmt="%.1f",fontsize=8)
a1.invert_yaxis(); a1.plot([1.0],[0.667],"*",color=ACC,ms=15)
a1.set_xlabel(r"relapse-hazard shape  $k$"); a1.set_ylabel(r"activity gap  $\rho$ (behavior)")
a1.set_title(r"A  Joint fold $\mathcal{R}_c(k,\rho)$",loc="left",fontweight="bold",color=INK)
fig.colorbar(cf,ax=a1,pad=0.02).set_label(r"$\mathcal{R}_c$",fontsize=9)
Roff=np.array([fold_k(k,1.0)[0] for k in ks]); Ron=np.array([fold_k(k,0.667)[0] for k in ks])
a2.plot(ks,Ast,color=GRN,lw=1.9,label=r"fold location $a^*$")
a2.set_xlabel(r"relapse-hazard shape  $k$"); a2.set_ylabel(r"fold prevalence $a^*$",color=GRN)
a2.tick_params(axis='y',colors=GRN); a2b=a2.twinx(); a2b.grid(False)
a2b.plot(ks,-np.log(Ron/Roff),color=ACC,lw=1.9)
a2b.set_ylabel(r"$-\Delta\log\mathcal{R}_c$ from behavior",color=ACC); a2b.tick_params(axis='y',colors=ACC)
a2.set_title("B  Behaviour bites hardest where prevalence is high",loc="left",fontweight="bold",color=INK)
fig.suptitle("Behaviour $\\times$ relapse timing: substitutes on the absolute threshold, complements multiplicatively",
             x=0.07,ha="left",fontweight="bold",fontsize=11.5,color=INK)
fig.savefig("figs/joint_fold_map.pdf",facecolor="white"); plt.close(fig)
print("joint map done")

# ---------- S10: robustness surface -- enlargement ratio vs k, across parameters ----------
def fold_generic(k,MU,GAM,PH,NU,r):
    L=PH/MU; lam=(1.0/MU)*(G(k+1)/L)**(1.0/k)
    TA=np.linspace(0,40.0/MU,40001); E=np.exp(-MU*TA)
    def st(a): return GAM*a*np.trapezoid(E*np.exp(-a*(TA/lam)**k),TA)
    def stc(a): return GAM*a/(MU+PH*a)
    v=np.array([R0(a,st(a),r,MU,GAM,NU) for a in AG]); f1=np.nanmin(v)
    vc=np.array([R0(a,stc(a),r,MU,GAM,NU) for a in AG]); f2=np.nanmin(vc)
    return f2/f1
variants=[("baseline",dict()),(r"$\nu\times0.5$",dict(NU=nu*0.5)),(r"$\rho=1$",dict(r=1.0)),
          (r"$\phi\times0.5$",dict(PH=phi0*0.5)),(r"$\phi\times2$",dict(PH=phi0*2)),
          (r"$\gamma\times0.5$",dict(GAM=gam*0.5)),(r"$\gamma\times2$",dict(GAM=gam*2)),
          (r"$\mu\times0.5$",dict(MU=mu*0.5)),(r"$\mu\times2$",dict(MU=mu*2))]
kk=np.array([0.30,0.40,0.50,0.65,0.80,0.90])
fig,ax=plt.subplots(figsize=(7.6,4.8)); fig.subplots_adjust(left=0.11,right=0.97,top=0.9,bottom=0.13)
cmap=plt.cm.viridis(np.linspace(0,0.9,len(variants)))
for (lab,kw),c in zip(variants,cmap):
    base=dict(MU=mu,GAM=gam,PH=phi0,NU=nu,r=0.667); base.update(kw)
    ys=[fold_generic(k,**base) for k in kk]
    ax.plot(kk,ys,"o-",color=c,lw=1.6,ms=4,label=lab)
ax.axhline(1.0,color="#444",lw=1.2,ls="--")
ax.set_yscale("log"); ax.set_xlabel(r"relapse-hazard shape  $k$")
ax.set_ylabel(r"basin enlargement  $\mathcal{R}_c^{\rm const}/\mathcal{R}_c(k)$")
ax.set_title("Enlargement across the whole $k<1$ range and all parameters",loc="left",fontweight="bold",color=INK)
ax.legend(frameon=False,fontsize=8,ncol=3,loc="upper right")
fig.savefig("figs/robust_surface.pdf",facecolor="white")
fig.savefig("robust_surface.png",dpi=160,facecolor="white")
print("robustness surface done")
