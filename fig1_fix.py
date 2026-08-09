import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from chain_v2 import make_profile, TAU, mu, gam, nu, beta
rb=0.667
def M(a): return 1.0/(1-a*(1-rb))
def stil(a,Phi): return gam*a*np.trapezoid(np.exp(-mu*TAU-a*Phi(TAU)),TAU)
def R0a(a,Phi):
    st=stil(a,Phi); s=1-a-st
    return mu*(1+nu*st)*(a+st)/(a*(mu+gam)*M(a)*s) if s>1e-12 else np.nan
INK="#1b1b1b"; ACC="#c1440e"; BLU="#2f6690"; GRN="#3a7d44"
plt.rcParams.update({"font.family":"DejaVu Sans","axes.edgecolor":"#444","axes.linewidth":0.8,
 "axes.grid":True,"grid.color":"#eaeaea","axes.axisbelow":True,"font.size":10.5})
fig,ax=plt.subplots(figsize=(7.6,5.2)); fig.subplots_adjust(left=0.11,right=0.96,top=0.9,bottom=0.12)
ag=np.linspace(0.0008,0.44,1400)
for kind,c,lab in [("early",ACC,"early relapse (decaying hazard)"),
                   ("const",BLU,"constant $\\phi$ (companion model)"),
                   ("late",GRN,"late relapse (rising hazard)")]:
    Phi,_,_=make_profile(kind)
    R=np.array([R0a(a,Phi) for a in ag])
    ok=np.isfinite(R); a_,R_=ag[ok],R[ok]
    i=np.nanargmin(R_)
    interior = 0<i<len(R_)-1
    if interior:
        Rc,ac=R_[i],a_[i]
        ax.plot(R_[i:],a_[i:],color=c,lw=2.0,label=f"{lab}:  $\\mathcal{{R}}_c$={Rc:.2f}")
        ax.plot(R_[:i+1],a_[:i+1],color=c,lw=1.3,ls=":")
        ax.plot([Rc],[ac],"o",color=c,ms=6)
    else:
        ax.plot(R_,a_,color=c,lw=2.0,label=f"{lab}:  no interior minimum")
ax.axvline(1.0,color="#888",lw=0.9,ls="--"); ax.text(1.008,0.40,r"$\mathcal{R}_0=1$",color="#666",fontsize=9)
ax.axhline(0,color=INK,lw=1.2)
ax.set_xlim(0,1.25); ax.set_ylim(-0.01,0.44)
ax.set_xlabel(r"basic reproduction number  $\mathcal{R}_0$")
ax.set_ylabel(r"endemic addicted fraction  $a^*$")
ax.set_title("Relapse timing moves the fold across the whole bistable range",loc="left",fontweight="bold",color=INK)
ax.legend(frameon=False,fontsize=9,loc="upper left")
ax.annotate("earlier relapse:\nendemic state persists\nfar below threshold",xy=(0.10,0.24),xytext=(0.32,0.10),
            fontsize=8.6,color=ACC,arrowprops=dict(arrowstyle="->",color=ACC,lw=1))
fig.savefig("figs/relapse_bifurcation.pdf",facecolor="white")
print("saved; late branch min at grid edge (no fold) as expected")
