import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from s4_regularize import Phi_reg, fold, Phi_exp, Phi_step
INK="#1b1b1b"; ACC="#c1440e"; BLU="#2f6690"; GRN="#3a7d44"; MUT="#8a8a8a"
plt.rcParams.update({"font.family":"DejaVu Sans","axes.edgecolor":"#444","axes.linewidth":0.8,
 "axes.grid":True,"grid.color":"#ececec","axes.axisbelow":True,"font.size":10.5})
fig,(a1,a2)=plt.subplots(1,2,figsize=(11.4,4.5))
fig.subplots_adjust(left=0.075,right=0.985,wspace=0.26,top=0.86,bottom=0.15)

# Panel A: fold vs regularization eps, for the three calibrated shapes
eps=[0.5,1,3,7,15,30,60,90,180,365,1000]
for nm,k,c in [("shape $k$=0.30",0.30,BLU),("shape $k$=0.34",0.34,ACC),("shape $k$=0.45",0.45,GRN)]:
    v=[fold(Phi_reg(k,e)[0])[0] for e in eps]
    a1.plot(eps,v,"o-",color=c,lw=1.8,ms=4.5,label=nm)
a1.axhline(0.8634,color=MUT,lw=1.6,ls="--"); a1.text(0.7,0.79,"constant $\\phi$ (memoryless): 0.863",fontsize=8.4,color="#555")
a1.axvspan(7,90,color=ACC,alpha=0.07)
a1.text(9,0.52,"biologically\nplausible\nregularization",fontsize=8.2,color=ACC)
a1.set_xscale("log"); a1.set_xlabel(r"regularization  $\epsilon$  (days, log):  $\phi=c(\epsilon+\tau)^{k-1}$")
a1.set_ylabel(r"fold  $\mathcal{R}_c$"); a1.set_ylim(0,0.99)
a1.set_title("A  Removing the $\\tau\\!=\\!0$ singularity barely moves the fold",loc="left",fontweight="bold",color=INK)
a1.legend(frameon=False,fontsize=8.5,loc="center left")

# Panel B: bounded families vs singular Weibull -- all fixed L
names=[];vals=[];cols=[]
for sig,lab in [(180,"exp decay\n$\\sigma$=180d"),(365,"exp decay\n$\\sigma$=365d"),(730,"exp decay\n$\\sigma$=730d")]:
    names.append(lab); vals.append(fold(Phi_exp(sig))[0]); cols.append(BLU)
for T,lab in [(90,"step\n90d"),(365,"step\n365d"),(730,"step\n730d")]:
    names.append(lab); vals.append(fold(Phi_step(T)[0])[0]); cols.append(GRN)
for e,lab in [(0.0,"Weibull\n$\\epsilon$=0"),(30.0,"Weibull\n$\\epsilon$=30d")]:
    names.append(lab); vals.append(fold(Phi_reg(0.34,e)[0])[0]); cols.append(ACC)
x=np.arange(len(vals))
a2.bar(x,vals,color=cols,alpha=0.87,width=0.64)
a2.axhline(0.8634,color=MUT,lw=1.6,ls="--")
a2.text(0.05,0.80,"constant $\\phi$: 0.863",fontsize=8.4,color="#555")
for i,v in enumerate(vals): a2.text(i,v+0.012,f"{v:.3f}",ha="center",fontsize=7.9,color="#333")
a2.set_xticks(x); a2.set_xticklabels(names,fontsize=7.5); a2.set_ylim(0,0.95)
a2.set_ylabel(r"fold  $\mathcal{R}_c$")
a2.set_title("B  Bounded hazards give the same answer",loc="left",fontweight="bold",color=INK)
a2.grid(axis="x",visible=False)
fig.suptitle("The enlarged basin is produced by front-loading itself, not by the unbounded hazard at $\\tau=0$",
             x=0.075,ha="left",fontweight="bold",fontsize=11.5,color=INK)
fig.savefig("figs/regularization.pdf",facecolor="white")
fig.savefig("regularization.png",dpi=165,facecolor="white")
print("saved")
