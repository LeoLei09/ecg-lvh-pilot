from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize

def _directional_batch(a,b):
    from scipy.signal import resample_poly
    a=np.asarray(a,float); b=np.asarray(b,float)
    count=len(a)
    if not count: return np.empty(0)
    smalla=resample_poly(a,1,5,axis=1); smallb=resample_poly(b,1,5,axis=1)
    def standard(x): return (x-np.median(x,axis=1)[:,None])/np.maximum(np.percentile(x,75,axis=1)-np.percentile(x,25,axis=1),1e-8)[:,None]
    f=standard(smalla); g=standard(smallb)
    cost=np.full((count,101,101),np.inf); prev=np.full((count,100,100),-1,dtype=np.int8)
    cost[:,1,1]=(f[:,0]-g[:,0])**2
    for diagonal in range(1,199):
        i=np.arange(max(0,diagonal-99),min(99,diagonal)+1); j=diagonal-i
        mask=np.abs(i-j)<=20; i=i[mask]; j=j[mask]
        parents=np.stack((cost[:,i,j],cost[:,i,j+1],cost[:,i+1,j]),axis=2)
        code=np.argmin(parents,axis=2)
        cost[:,i+1,j+1]=np.take_along_axis(parents,code[:,:,None],axis=2)[:,:,0]+(f[:,i]-g[:,j])**2
        prev[:,i,j]=code
    sums=np.zeros((count,100)); nums=np.zeros((count,100)); ii=np.full(count,99); jj=ii.copy(); active=np.ones(count,bool)
    while np.any(active):
        k=np.flatnonzero(active); np.add.at(sums,(k,ii[k]),jj[k]); np.add.at(nums,(k,ii[k]),1)
        done=(ii==0)&(jj==0); active &= ~done
        k=np.flatnonzero(active); codes=prev[k,ii[k],jj[k]]
        ii[k]-=(codes!=2); jj[k]-=(codes!=1)
    maps=sums/nums; coords=np.arange(500,dtype=float); nodes=np.linspace(0,499,100)
    values=[]
    for k in range(count):
        mapping=np.interp(coords,nodes,maps[k]*499/99); shifts=np.clip(mapping-coords,-100,100)
        warped=np.interp(coords+shifts,coords,b[k]); ac=a[k]-np.mean(a[k]); wc=warped-np.mean(warped)
        energy=float(ac@ac)
        if energy<1e-12: values.append(np.nan); continue
        amp=np.clip(float(ac@wc)/(energy+1e-6*max(energy,1)),0,5)
        values.append(10*abs(amp-1)+(np.max(np.abs(shifts))+np.std(shifts,ddof=1))/500)
    return np.asarray(values)

def distance_batch(a,b,chunk_size=128):
    a=np.asarray(a,float); b=np.asarray(b,float)
    if len(a)==0: return np.empty(0)
    return np.concatenate([(_directional_batch(a[i:i+chunk_size],b[i:i+chunk_size])+_directional_batch(b[i:i+chunk_size],a[i:i+chunk_size]))/2 for i in range(0,len(a),chunk_size)])

def repair_candidates(peaks, composite):
    # Frozen developmental QRS/T-like discriminator; original 250-ms detector and RR rules unchanged.
    p=np.asarray(peaks,int); c=np.asarray(composite,float); amp=np.abs(c[p])
    slope=np.asarray([np.max(np.abs(np.diff(c[max(0,q-20):min(len(c),q+21)]))) for q in p])
    kept=[]; removed=[]
    for i,q in enumerate(p):
        if kept:
            j=kept[-1]; lag=(q-p[j])/500
            if 0.20<=lag<=0.40 and amp[i]<0.45*amp[j] and slope[i]<0.50*slope[j]:
                removed.append(int(q)); continue
        kept.append(i)
    return p[kept],tuple(removed)

def stat_vector(x,minimum=3):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    if len(x)<minimum: return np.full(4,np.nan)
    # Sorting gives exactly invariant floating point reductions, not merely approximate invariance.
    x=np.sort(x); med=np.median(x)
    return np.array([np.mean(x),np.std(x,ddof=1),np.median(np.abs(x-med)),np.percentile(x,75)-np.percentile(x,25)])

def sequence_features(vectors, pairwise, original_pair_mask, order=None):
    v=np.asarray(vectors,float); n=len(v); order=np.arange(n) if order is None else np.asarray(order)
    mask=np.asarray(original_pair_mask,bool)
    b=np.concatenate([stat_vector(v[:,j]) for j in range(v.shape[1])])
    q=v[order]; c=[]
    for j in range(v.shape[1]): c.extend(stat_vector(np.abs(np.diff(q[:,j]))[mask],minimum=2))
    for lead in range(3):
        d=pairwise[order[:-1],order[1:],lead] if n>1 else np.array([])
        c.extend(stat_vector(d[mask],minimum=2))
    return b,np.asarray(c),int(mask.sum())

def fit_predict(Xtr,ytr,Xte):
    Xtr=np.asarray(Xtr,float); Xte=np.asarray(Xte,float); med=np.nanmedian(Xtr,0); med[~np.isfinite(med)]=0; sd=np.nanstd(Xtr,0,ddof=1); sd[~np.isfinite(sd)|(sd<1e-8)]=1
    A=np.where(np.isfinite(Xtr),(Xtr-med)/sd,0); B=np.where(np.isfinite(Xte),(Xte-med)/sd,0); A=np.c_[np.ones(len(A)),A]; B=np.c_[np.ones(len(B)),B]
    def loss(w):
        z=np.clip(A@w,-40,40); return np.mean(np.logaddexp(0,z)-ytr*z)+.5*np.sum(w[1:]**2)
    def grad(w):
        z=np.clip(A@w,-40,40); p=1/(1+np.exp(-z)); g=A.T@(p-ytr)/len(A); g[1:]+=w[1:]; return g
    r=minimize(loss,np.zeros(A.shape[1]),jac=grad,method='L-BFGS-B',options={'maxiter':250}); return 1/(1+np.exp(-np.clip(B@r.x,-40,40))), bool(r.success), med, sd

def templates_for_fold(meta,reps,train_ids,fold,scheme,min_strict=20):
    refs={}; ids={}
    for lab in (0,1):
        chosen=[]
        for eid in sorted(train_ids):
            m=meta[eid]
            if m['label']!=lab or eid not in reps: continue
            if scheme=='strict' and not (m['role']=='prototype_pool' and m['vh_ok']): continue
            chosen.append(eid)
        ids[lab]=chosen; refs[lab]=np.mean(np.stack([reps[x] for x in chosen]),axis=0) if len(chosen)>=(min_strict if scheme=='strict' else 1) else None
    status='ok' if all(refs[x] is not None for x in refs) else 'insufficient'
    return refs,ids,status
