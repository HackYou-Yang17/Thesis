import sys, glob, json
import numpy as np
from skimage.morphology import binary_dilation, disk
from skimage.draw import line as draw_line
from bias_check import real_crossings
from traced_dominance import load_trace, SCALE_UM, FIELD_UM, WIDTH_PX
from datapaths import ROOT as DATA_ROOT
N = int(round(FIELD_UM/SCALE_UM))
def synth(seed, sep, jit=25.0, pitch=2.80):
    rng=np.random.default_rng(seed); img=np.zeros((N,N),bool); j=np.deg2rad(jit)
    for fam in (0.0, np.deg2rad(sep)):
        for o in np.arange(-N,2*N,pitch/SCALE_UM):
            a=fam+rng.uniform(-j,j); c,s=np.cos(a),np.sin(a)
            px,py=o*(-s)+N/2,o*c+N/2; L=3*N
            rr,cc=draw_line(int(py-L*s),int(px-L*c),int(py+L*s),int(px+L*c))
            ok=(rr>=0)&(rr<N)&(cc>=0)&(cc<N)
            if ok.sum()<10: continue
            img[rr[ok],cc[ok]]=True
    return binary_dilation(img,disk(WIDTH_PX//2))

cal={}
for sep in (0,30,45,60,75,90):
    a=np.concatenate([real_crossings(synth(s,sep)) for s in range(5)])
    cal[sep]=dict(n=len(a), med=float(np.median(a)), f60=float(np.mean(a>60)), ang=a.tolist())
real={}
for hpf in (32,36,40,44,48,52):
    per=[]; acc=[]
    for f in sorted(glob.glob(DATA_ROOT + f"/{hpf}hpf/*- Copy.tif")):
        m,*_=load_trace(f); x=real_crossings(m); acc.append(x)
        per.append(dict(med=float(np.median(x)) if len(x) else None,
                        f60=float(np.mean(x>60)) if len(x) else None, n=len(x)))
    a=np.concatenate(acc)
    real[hpf]=dict(n=len(a), med=float(np.median(a)), f60=float(np.mean(a>60)),
                   per=per, ang=a.tolist())
json.dump(dict(cal={str(k):{kk:vv for kk,vv in v.items()} for k,v in cal.items()},
               real={str(k):v for k,v in real.items()}), open("res.json","w"))
print(f"{'hpf':>5s} {'n':>5s} {'median':>7s} {'>60deg':>7s}   per-heart >60")
for h,v in real.items():
    ph=", ".join(f"{p['f60']:.2f}" for p in v['per'])
    print(f"{h:>5d} {v['n']:>5d} {v['med']:>7.1f} {v['f60']:>7.2f}   [{ph}]")
