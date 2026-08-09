import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
d=json.load(open('g500.json')); seeds=[0.03,0.07,0.11,0.15]
old=json.load(open('gres.json'))
INK="#1b1b1b"; ACC="#c1440e"; BLU="#2f6690"; GRN="#3a7d44"
plt.rcParams.update({"font.family":"DejaVu Sans","axes.edgecolor":"#444","axes.linewidth":0.8,
 "axes.grid":True,"grid.color":"#ececec","axes.axisbelow":True,"font.size":10.5})
fig,(ax,ax2)=plt.subplots(1,2,figsize=(11.2,4.4),gridspec_kw={"width_ratios":[1.25,1]})
fig.subplots_adjust(left=0.075,right=0.98,wspace=0.28,top=0.86,bottom=0.14)
for k,c,lab in [("early",ACC,"early relapse (declining hazard)"),
                ("const",BLU,"constant $\\phi$ (companion model)"),
                ("late",GRN,"late relapse (rising hazard)")]:
    p=[d[f"{k}|{s}"][0] for s in seeds]; lo=[max(d[f"{k}|{s}"][1],0) for s in seeds]
    hi=[min(d[f"{k}|{s}"][2],1) for s in seeds]
    ax.plot(seeds,p,"o-",color=c,lw=1.9,ms=6,label=lab); ax.fill_between(seeds,lo,hi,color=c,alpha=0.16)
ax.set_xlabel("initial addicted seed  $a(0)$"); ax.set_ylabel("establishment probability")
ax.set_ylim(-0.04,1.04); ax.legend(frameon=False,fontsize=8.8,loc="center right")
ax.set_title("A  Establishment at $\\mathcal{R}_0=0.95$ ($n=500$)",loc="left",fontweight="bold",color=INK)
cats=["mesh\nN=20","mesh\nN=40","mesh\nN=60","thr\n0.12","thr\n0.17","thr\n0.25","$N_0$\n750","$N_0$\n1500","$N_0$\n3000"]
vals=[old["mesh"]["N20"][1],old["mesh"]["N40"][1],old["mesh"]["N60"][1],
      old["threshold"]["0.12"][1],old["threshold"]["0.17"][1],old["threshold"]["0.25"][1],
      old["finitesize"]["750"][1],old["finitesize"]["1500"][1],old["finitesize"]["3000"][1]]
n=60; err=[1.96*np.sqrt(max(v*(1-v),1e-4)/n) for v in vals]; x=np.arange(len(vals))
ax2.bar(x,vals,yerr=err,capsize=3,color=BLU,alpha=0.85,width=0.62)
ax2.axhline(1.0,color=ACC,lw=1.6,ls="--"); ax2.text(0.1,1.03,"early relapse = 1.00 throughout",color=ACC,fontsize=8.2)
ax2.axhline(0.0,color=GRN,lw=1.6,ls=":"); ax2.text(0.1,-0.075,"late relapse = 0.00 throughout",color=GRN,fontsize=8.2)
ax2.set_xticks(x); ax2.set_xticklabels(cats,fontsize=7.6); ax2.set_ylim(-0.12,1.15)
ax2.set_ylabel("establishment prob. (constant $\\phi$)")
ax2.set_title("B  Robust to mesh, threshold, system size",loc="left",fontweight="bold",color=INK)
fig.suptitle("Stochastic establishment: ordering early $\\gg$ constant $\\gg$ late is discretization-independent",
             x=0.075,ha="left",fontweight="bold",fontsize=11.5,color=INK)
fig.savefig("figs/relapse_establishment_v2.pdf",facecolor="white")
print("const:",[round(d[f'const|{s}'][0],3) for s in seeds])
print("halfwidths:",[round((d[f'const|{s}'][2]-d[f'const|{s}'][1])/2,3) for s in seeds])
