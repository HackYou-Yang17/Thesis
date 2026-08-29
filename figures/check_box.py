import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, itertools, runpy
runpy.run_path('make_box_fig.py', run_name='__main__')
fig=plt.gcf(); fig.canvas.draw(); texts=[]
for ax in fig.get_axes():
    items=list(ax.texts)+[ax.title,ax.xaxis.label,ax.yaxis.label]
    if ax.axison:
        x0,x1=sorted(ax.get_xlim()); y0,y1=sorted(ax.get_ylim())
        items+=[t for t in ax.get_xticklabels() if t.get_text().strip() and x0-1e-9<=t.get_position()[0]<=x1+1e-9]
        items+=[t for t in ax.get_yticklabels() if t.get_text().strip() and y0-1e-9<=t.get_position()[1]<=y1+1e-9]
    if ax.get_legend(): items+=ax.get_legend().get_texts()
    texts+=[t for t in items if t.get_text().strip() and t.get_visible()]
texts+=[t for t in fig.texts if t.get_text().strip()]
bad=[(t.get_text()[:24],t.get_fontsize()) for t in texts if t.get_fontsize()<11]
print('FONT   :','OK' if not bad else bad)
r=fig.canvas.get_renderer(); bb=[(t,t.get_window_extent(r)) for t in texts]
ov=[(a.get_text()[:18],b.get_text()[:18]) for (a,x),(b,y) in itertools.combinations(bb,2) if x.overlaps(y)]
print('OVERLAP:','OK' if not ov else ov)
w,h=fig.canvas.get_width_height()
print('CLIP   :','OK' if not [1 for t,x in bb if x.x0<-1 or x.y0<-1 or x.x1>w+1 or x.y1>h+1] else 'clipped')
