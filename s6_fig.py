import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
INK="#1b1b1b"; ACC="#c1440e"; BLU="#2f6690"; MUT="#8a8a8a"
plt.rcParams.update({"font.family":"DejaVu Sans","axes.edgecolor":"#444","axes.linewidth":0.8,
 "axes.grid":True,"grid.color":"#ececec","axes.axisbelow":True,"font.size":10.5})
fig,(a1,a2)=plt.subplots(1,2,figsize=(11.2,4.4),gridspec_kw={"width_ratios":[1.15,1]})
fig.subplots_adjust(left=0.075,right=0.985,wspace=0.28,top=0.86,bottom=0.15)

# Panel A: the two fitted survival curves (log-time)
t=np.logspace(0,3.2,400)          # 1 day .. ~1600 d
k1,lam1=0.402,None                # untreated: from S(8d)=0.5,S(365)=0.04
lam1=8.0/(np.log(1/0.5))**(1/k1)
k2=0.891; lam2=40.5*30.44         # months -> days
a1.plot(t,np.exp(-(t/lam1)**k1),color=ACC,lw=2,label=f"untreated self-quitters  $k$={k1:.2f}")
a1.plot(t,np.exp(-(t/lam2)**k2),color=BLU,lw=2,label=f"methadone maintenance  $k$={k2:.2f}")
a1.plot([8,365],[0.50,0.04],"o",color=ACC,ms=7,mfc="white",mew=1.7)
tm=np.array([6,12,18,24,30])*30.44; Sm=np.array([0.83,0.72,0.62,0.53,0.46])
a1.plot(tm,Sm,"s",color=BLU,ms=6,mfc="white",mew=1.7)
a1.set_xscale("log"); a1.set_xlabel("days since quitting (log)"); a1.set_ylabel("fraction still abstinent")
a1.set_ylim(0,1.02); a1.legend(frameon=False,fontsize=8.6,loc="lower left")
a1.set_title("A  Two real relapse curves, very different shapes",loc="left",fontweight="bold",color=INK)
a1.text(1.4,0.16,"open markers = published\nsurvival / life-table points",fontsize=7.8,color="#666")

# Panel B: implied folds with CIs
labs=["untreated\nself-quitters\n$k$=0.40","methadone\nmaintenance\n$k$=0.89","constant\nhazard\n$k$=1"]
mid=[0.118,0.578,0.863]; lo=[0.111,0.512,0.863]; hi=[0.127,0.652,0.863]
err=[[m-l for m,l in zip(mid,lo)],[h-m for h,m in zip(hi,mid)]]
a2.bar(range(3),mid,yerr=err,capsize=5,color=[ACC,BLU,MUT],alpha=0.87,width=0.6)
for i,m in enumerate(mid): a2.text(i,hi[i]+0.03,f"{m:.2f}",ha="center",fontsize=9,color="#333")
a2.axhline(1.0,color="#999",lw=0.9,ls="--"); a2.text(2.35,1.02,"$\\mathcal{R}_0=1$",fontsize=8,color="#777",ha="right")
a2.set_xticks(range(3)); a2.set_xticklabels(labs,fontsize=8.2)
a2.set_ylabel(r"fold  $\mathcal{R}_c$"); a2.set_ylim(0,1.12); a2.grid(axis="x",visible=False)
a2.annotate("",xy=(1,0.70),xytext=(0,0.70),arrowprops=dict(arrowstyle="<->",color=INK,lw=1.3))
a2.text(0.5,0.73,"4.9$\\times$",ha="center",fontsize=9.5,fontweight="bold",color=INK)
a2.set_title("B  Implied eradication fold",loc="left",fontweight="bold",color=INK)
fig.suptitle("Relapse-curve shape is a property of treatment context, not of the substance --- and it moves the fold fivefold",
             x=0.075,ha="left",fontweight="bold",fontsize=11.3,color=INK)
fig.savefig("figs/empirical_curves.pdf",facecolor="white")
fig.savefig("empirical_curves.png",dpi=165,facecolor="white")
print("saved")
