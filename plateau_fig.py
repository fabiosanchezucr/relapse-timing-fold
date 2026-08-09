import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.special import gamma as G
from exact_model import P
mu,gam,phi0,nu=P['mu'],P['gamma'],P['phi'],P['nu']
def M(a,rho=0.667): return 1.0/(1-a*(1-rho))
L0=phi0/mu; TAU=np.linspace(0,150000,150001)
def lam_of_k(k): return (1.0/mu)*(G(k+1)/L0)**(1.0/k)
def stil_ex(a,k):
    lam=lam_of_k(k); return gam*a*np.trapezoid(np.exp(-mu*TAU-a*(TAU/lam)**k),TAU)
def stil_as(a,k): return gam*lam_of_k(k)*G(1+1.0/k)*a**(1-1.0/k)
def R0(a,st):
    s=1-a-st
    return mu*(1+nu*st)*(a+st)/(a*(mu+gam)*M(a)*s) if s>1e-9 else np.nan
def fold(k,exact=True,n=400):
    ag=np.linspace(0.005,0.60,n)
    R=np.array([R0(a,(stil_ex(a,k) if exact else stil_as(a,k))) for a in ag])
    i=np.nanargmin(R); return R[i],ag[i]
def fold_const():
    st=lambda a: gam*a/(mu+phi0*a); ag=np.linspace(0.002,0.6,4000)
    R=np.array([R0(a,st(a)) for a in ag]); i=np.nanargmin(R); return R[i]

INK="#1b1b1b"; ACC="#c1440e"; BLU="#2f6690"; GRN="#3a7d44"
plt.rcParams.update({"font.family":"DejaVu Sans","axes.edgecolor":"#444","axes.linewidth":0.8,
 "axes.grid":True,"grid.color":"#ececec","axes.axisbelow":True,"font.size":10.5})
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11.4,4.5))
fig.subplots_adjust(left=0.075,right=0.985,wspace=0.27,top=0.86,bottom=0.15)

ks=np.array([0.25,0.30,0.34,0.40,0.45,0.55,0.70,0.85,1.0])
Fe=[];Fa=[]
for k in ks:
    Fe.append(fold(k,True,300)[0]); Fa.append(fold(k,False,3000)[0])
Fe=np.array(Fe);Fa=np.array(Fa); Fc=fold_const()
ax1.plot(ks,Fe,"o-",color=INK,lw=2,ms=6,label="exact (quadrature)")
ax1.plot(ks,Fa,"--",color=ACC,lw=1.8,label=r"asymptotic $\tilde s^*\!\sim\!\gamma\Lambda_k a^{1-1/k}$")
ax1.plot([1.0],[Fc],"s",color=BLU,ms=9,label=f"constant $\\phi$ (memoryless): {Fc:.2f}")
subs={"k=0.34":0.34,"k=0.30":0.30,"k=0.45":0.45}
for n_,k in subs.items():
    f=fold(k,True,300)[0]; ax1.plot([k],[f],"o",color=GRN,ms=9)
    ax1.annotate(n_,(k,f),textcoords="offset points",xytext=(4,7),fontsize=8.3,color=GRN)
ax1.set_xlabel(r"relapse-hazard shape  $k$   (Weibull; $k<1$ declining)")
ax1.set_ylabel(r"saddle-node fold  $\mathcal{R}_c$")
ax1.set_title("A  Closed form captures the fold across shapes",loc="left",fontweight="bold",color=INK)
ax1.legend(frameon=False,fontsize=8.3,loc="upper left")

# Panel B: robustness — ratio R_c(const)/R_c(k=0.34) across parameter variations
labels=["baseline",r"$\nu\!\times\!0.5$",r"$\rho=1$ (no behav.)",r"$\phi\!\times\!0.5$",
        r"$\phi\!\times\!2$",r"$\gamma\!\times\!0.5$",r"$\gamma\!\times\!2$",r"$\mu\!\times\!0.5$",r"$\mu\!\times\!2$"]
ratios=[10.6,10.6,10.0,7.5,4.2,3.4,22.6,16.2,5.0]
y=np.arange(len(labels))
ax2.barh(y,ratios,color=[ACC if r>=10 else BLU for r in ratios],alpha=0.85,height=0.62)
ax2.axvline(1.0,color="#444",lw=1.2)
ax2.set_yticks(y); ax2.set_yticklabels(labels,fontsize=8.6); ax2.invert_yaxis()
ax2.set_xlabel(r"basin enlargement  $\mathcal{R}_c^{\rm const}/\mathcal{R}_c^{\rm early}$")
ax2.set_title("B  Enlargement is robust to all parameters",loc="left",fontweight="bold",color=INK)
for i,r in enumerate(ratios): ax2.text(r+0.3,i,f"{r:.1f}×",va="center",fontsize=8.2,color="#333")
ax2.set_xlim(0,25); ax2.grid(axis="y",visible=False)
fig.suptitle("The enlarged-basin plateau: an asymptotic closed form explains it, and it survives every parameter variation",
             x=0.075,ha="left",fontweight="bold",fontsize=11.5,color=INK)
fig.savefig("figs/plateau_asymptotic.pdf",facecolor="white")
fig.savefig("plateau_asymptotic.png",dpi=170,facecolor="white")
print("folds:", {n_:round(fold(k,True,300)[0],4) for n_,k in subs.items()}, "const:",round(Fc,4))
