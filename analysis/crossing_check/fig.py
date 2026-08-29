import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
d=json.load(open("res.json")); cal=d["cal"]; real=d["real"]
seps=sorted(int(k) for k in cal); hpfs=sorted(int(k) for k in real)
cf=[cal[str(s)]["f60"] for s in seps]; cm=[cal[str(s)]["med"] for s in seps]
rf=[real[str(h)]["f60"] for h in hpfs]; rm=[real[str(h)]["med"] for h in hpfs]
rper=[[p["f60"] for p in real[str(h)]["per"]] for h in hpfs]

fig,ax=plt.subplots(1,3,figsize=(13.5,4.1))
O,G,K="#C1622D","#2E7D64","#333333"

a=ax[0]
a.plot(seps,cf,"o-",color=K,lw=1.6,ms=5,label="synthetic, known separation")
for s,f in zip(seps,cf): a.annotate(f"{s}°",(s,f),textcoords="offset points",xytext=(4,-11),fontsize=8,color=K)
a.axhspan(min(rf[-1],rf[-3]),max(rf[-1],rf[-3]),color=G,alpha=.13)
a.axhline(rf[-1],color=G,lw=1.4)
a.annotate("traced 52 hpf = 0.43",(2,rf[-1]),textcoords="offset points",xytext=(2,5),fontsize=9,color=G)
a.axhline(rf[0],color=O,lw=1.4)
a.annotate("traced 32 hpf = 0.00",(2,rf[0]),textcoords="offset points",xytext=(2,5),fontsize=9,color=O)
a.set_xlabel("true separation between families (°)"); a.set_ylabel("fraction of crossings > 60°")
a.set_title("a  Calibration: 52 hpf tissue sits at 60–75°",fontsize=10,loc="left")
a.set_ylim(-.03,.6); a.legend(fontsize=8,frameon=False,loc="lower right")

a=ax[1]
a.plot(hpfs,rf,"o-",color=G,lw=1.8,ms=6,label="fraction > 60°  (stable)")
for h,ps in zip(hpfs,rper): a.plot([h]*len(ps),ps,"o",color=G,ms=3.5,alpha=.45)
a2=a.twinx()
a2.plot(hpfs,rm,"s--",color=O,lw=1.4,ms=5,label="median (°)  (unstable)")
for h in hpfs:
    ms=[p for p in [real[str(h)]["per"][i]["med"] for i in range(len(real[str(h)]["per"]))]]
    a2.plot([h]*len(ms),ms,"s",color=O,ms=3.5,alpha=.45)
a.set_xlabel("hpf"); a.set_ylabel("fraction > 60°",color=G); a2.set_ylabel("median crossing angle (°)",color=O)
a.set_title("b  The median is the unstable statistic",fontsize=10,loc="left")
h1,l1=a.get_legend_handles_labels(); h2,l2=a2.get_legend_handles_labels()
a.legend(h1+h2,l1+l2,fontsize=8,frameon=False,loc="upper left")

a=ax[2]
bins=np.linspace(0,90,19)
a.hist(cal["90"]["ang"],bins=bins,density=True,color=K,alpha=.30,label="synthetic, true 90°")
a.hist(real["52"]["ang"],bins=bins,density=True,histtype="step",color=G,lw=2,label="traced 52 hpf")
a.hist(real["32"]["ang"],bins=bins,density=True,histtype="step",color=O,lw=2,label="traced 32 hpf")
a.axvline(60,color="k",ls=":",lw=1)
a.set_xlabel("crossing angle (°)"); a.set_ylabel("density")
a.set_title("c  Bimodal: spur junctions fill the low mode",fontsize=10,loc="left")
a.legend(fontsize=8,frameon=False)
plt.tight_layout(); plt.savefig("crossing_check.png",dpi=170)
print("saved")
