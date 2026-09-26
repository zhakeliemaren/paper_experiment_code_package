from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parent
rows=json.loads((ROOT/'results.json').read_text(encoding='utf-8'))['candidates']
best={}
for r in rows:
    if r.get('success') and (r['comparison_method'] not in best or r['reported_rho']<best[r['comparison_method']]['reported_rho']): best[r['comparison_method']]=r
ours=best['diffusion_direct_generator_sos']
def barrier(x):
    return sum(float(c)*np.prod(np.asarray(x)**np.asarray(e),axis=-1) for c,e in zip(ours['barrier_coefficients'],ours['barrier_exponents']))
def xy(v): return (int(450+float(v[0])*55), int(370-float(v[1])*55))
FONT=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22)
SMALL=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
def canvas(title):
    im=Image.new('RGB',(900,700),'white'); d=ImageDraw.Draw(im); d.text((48,20),title,fill='black',font=FONT); return im,d

im,d=canvas('D1 Duffing barrier certificate B(x)')
gx=np.linspace(-6,6,280); gy=np.linspace(-6,6,220)
X,Y=np.meshgrid(gx,gy); Q=barrier(np.stack([X,Y],axis=-1))
lo,hi=np.quantile(Q,[.08,.92])
def px(x): return 90+(x+6)*60
def py(y): return 610-(y+6)*43
for j in range(220):
    for i in range(280):
        t=max(0,min(1,(Q[j,i]-lo)/(hi-lo+1e-9)))
        col=(int(35+180*t),int(145-100*t),int(195-130*t))
        d.rectangle((px(gx[i]),py(gy[j])-2,px(gx[i])+3,py(gy[j])+2),fill=col)
for j in range(219):
    for i in range(279):
        cell=Q[j:j+2,i:i+2]
        if cell.min()<=1<=cell.max(): d.point((px(gx[i]),py(gy[j])),fill='white')
d.rectangle((px(-2.5),py(2),px(2.5),py(-2)),outline=(25,25,25),width=3)
d.rectangle((px(5),py(6),px(6),py(5)),outline=(210,30,40),width=4)
for tick in [-6,-3,0,3,6]:
    d.text((px(tick)-12,625),str(tick),fill='black',font=SMALL)
    d.text((52,py(tick)-10),str(tick),fill='black',font=SMALL)
d.text((427,655),'x1',fill='black',font=SMALL)
d.text((9,350),'x2',fill='black',font=SMALL)
d.text((370,60),'Black: initial',fill='black',font=SMALL)
d.text((525,60),'Red: unsafe',fill=(200,30,40),font=SMALL)
d.text((675,60),'White: B=1',fill='black',font=SMALL)
im.save(ROOT/'d1_barrier.png')

im,d=canvas('D1 closed-loop dynamics and sampled trajectories'); rng=np.random.default_rng(7)
def rhs(x): return np.array([x[1],-.6*x[1]-x[0]-x[0]**3+.39*x[0]-1.41*x[1]])
for x0 in rng.uniform([-2.5,-2],[2.5,2],size=(12,2)):
    p=[x0.copy()]; x=x0.copy()
    for _ in range(120):
        x=x+.02*rhs(x)
        if np.max(np.abs(x))>6: break
        p.append(x.copy())
    d.line([xy(z) for z in p],fill=(30,100,180),width=1)
d.rectangle((xy([5,0])[0],xy([0,6])[1],xy([6,0])[0],xy([0,5])[1]),outline='red',width=3); d.text((35,650),'Initial: [-2.5,2.5] x [-2,2]    Unsafe: [5,6]^2',fill='black',font=SMALL); im.save(ROOT/'d1_dynamics.png')

im,d=canvas('descent certificate degree sweep'); labels={'diffusion_direct_generator_sos':'Ours','gaussian_neural_rsm_bernstein':'Gauss.+Neural','gaussian_c_sbc_multi_candidate_sos':'DiffSBC'}; colors=[(20,90,180),(200,80,40),(30,140,80)]
def sweep_x(degree): return {2:250,4:450,6:650}[degree]
def sweep_y(bound): return int(590-430*bound)
for bound in [0,.25,.5,.75,1]:
    y=sweep_y(bound)
    d.line((145,y,725,y),fill=(220,220,220),width=1)
    d.text((90,y-10),f'{bound:.2f}',fill='black',font=SMALL)
d.line((145,160,145,590,725,590),fill='black',width=2)
for degree in [2,4,6]:
    x=sweep_x(degree)
    d.line((x,590,x,598),fill='black',width=2)
    d.text((x-10,605),str(degree),fill='black',font=SMALL)
for k,(m,label) in enumerate(labels.items()):
    color=colors[k]
    d.line((85+260*k,105,112+260*k,105),fill=color,width=4)
    d.text((120+260*k,94),label,fill=color,font=SMALL)
    rs=sorted((r for r in rows if r['comparison_method']==m),key=lambda r:r['barrier_degree'])
    pts=[]
    for r in rs:
        x=sweep_x(int(r['barrier_degree']))
        if not r.get('success'):
            d.text((x-7,565),'F',fill=color,font=SMALL)
            continue
        pts.append((x,sweep_y(float(r['safety_lower_bound']))))
    if len(pts)>1: d.line(pts,fill=color,width=4)
    for x,y in pts: d.ellipse((x-6,y-6,x+6,y+6),fill=color)
d.text((350,650),'Barrier degree',fill='black',font=SMALL)
axis_label=Image.new('RGBA',(180,25),(255,255,255,0))
ImageDraw.Draw(axis_label).text((0,0),'Safety lower bound',fill='black',font=SMALL)
im.paste(axis_label.rotate(90,expand=True),(15,285),axis_label.rotate(90,expand=True))
im.save(ROOT/'d1_degree_sweep.png')

im,d=canvas('D1 certificate diagnostics'); y=100
for x,title in [(40,'Method'),(320,'Degree'),(430,'rho'),(555,'Safety LB'),(740,'Status')]: d.text((x,y),title,fill='black',font=SMALL)
y+=65
for m,label in labels.items():
    r=best[m]
    for x,value in [(40,label),(320,str(r['barrier_degree'])),(430,f'{r["reported_rho"]:.6f}'),(555,f'{1-r["reported_rho"]:.6f}'),(740,'verified')]: d.text((x,y),value,fill='black',font=SMALL)
    y+=65
im.save(ROOT/'d1_diagnostics.png')
(ROOT/'figure_manifest.json').write_text(json.dumps({'source':'Zhu et al. PLDI 2019, Example 4.3','figures':['d1_barrier.png','d1_dynamics.png','d1_degree_sweep.png','d1_diagnostics.png']},indent=2),encoding='utf-8')
