from __future__ import annotations
import sys, json, math, hashlib, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import warnings
warnings.filterwarnings('ignore', message='Glyph .* missing from current font')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parents[1]
BASE=OUT/'code'/'baseline'
ROOT=OUT.parent.parent
sys.path.insert(0,str(BASE))
from ecg5001.wfdb_io import load_wfdb_record
from ecg5001.beats import detect_r_peaks, extract_beats
from ecg5001.bsw_like import align_pair
from ecg5001.sqi import sqi_from_wfdb, _FLATLINE_THRESHOLD_MV

DATA=Path(r'D:\UserFiles\desktop\ECGdata\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3')
FINAL=Path(r'D:\UserFiles\desktop\dd\ecg5001-real-output-final5\run-20260918T013319Z-0edaca8ce0fe')
OUT=Path(__file__).resolve().parents[1]
FIG=OUT/'figures'; TABLE=OUT/'tables'; BEATS=OUT/'beats'; CASE=OUT/'casebook'
for p in [FIG,TABLE,BEATS,CASE]: p.mkdir(parents=True,exist_ok=True)
FS=500.0
LEADS=('I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6')
TARGET=('V1','V5','V6')
SEED=5001

# Utilities used by the analysis and tests.
def consecutive_adjacent_mean(values, source_indices):
    values=np.asarray(values,float); source=np.asarray(source_indices,int)
    if len(values)<2:return float('nan')
    d=np.abs(np.diff(values))[np.diff(source)==1]
    return float(np.mean(d)) if d.size else float('nan')

def patient_auc(scores, labels):
    scores=np.asarray(scores,float); labels=np.asarray(labels,int)
    pos=scores[labels==1]; neg=scores[labels==0]
    if len(pos)==0 or len(neg)==0:return float('nan')
    return float(np.mean((pos[:,None]>neg[None,:])+0.5*(pos[:,None]==neg[None,:])))

def robust_stats(x):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    if not len(x): return {'mean':np.nan,'sd':np.nan,'mad':np.nan,'iqr':np.nan}
    med=np.median(x)
    return {'mean':float(np.mean(x)),'sd':float(np.std(x,ddof=1)) if len(x)>1 else 0.0,'mad':float(np.median(np.abs(x-med))),'iqr':float(np.percentile(x,75)-np.percentile(x,25))}

def logistic_fit_predict(Xtr,ytr,Xte,l2=1.0):
    mu=np.nanmean(Xtr,axis=0); sd=np.nanstd(Xtr,axis=0,ddof=1); sd[~np.isfinite(sd)|(sd<1e-8)]=1.0
    med=np.nanmedian(Xtr,axis=0); med[~np.isfinite(med)]=0.0
    A=np.where(np.isfinite(Xtr),(Xtr-med)/sd,0.0); B=np.where(np.isfinite(Xte),(Xte-med)/sd,0.0)
    A=np.c_[np.ones(len(A)),A]; B=np.c_[np.ones(len(B)),B]
    def loss(w):
        z=np.clip(A@w,-40,40); p=1/(1+np.exp(-z)); return float(np.mean(np.logaddexp(0,z)-ytr*z)+0.5*l2*np.sum(w[1:]**2))
    def grad(w):
        z=np.clip(A@w,-40,40); p=1/(1+np.exp(-z)); g=A.T@(p-ytr)/len(A); g[1:]+=l2*w[1:]; return g
    res=minimize(loss,np.zeros(A.shape[1]),jac=grad,method='L-BFGS-B',options={'maxiter':300})
    p=1/(1+np.exp(-np.clip(B@res.x,-40,40)))
    return p,res.success

def auc(scores,labels): return patient_auc(scores,labels)
def balacc(scores,labels):
    s=np.asarray(scores)>=0.5; y=np.asarray(labels).astype(int)
    vals=[]
    for k in [0,1]:
        den=np.sum(y==k); vals.append(float(np.sum(s[y==k]==k)/den) if den else np.nan)
    return float(np.nanmean(vals))

def recompute_flat_flags(signal, thresholds):
    out=[]
    for th in thresholds:
        frac=np.mean(np.abs(np.diff(signal,axis=0))<th,axis=0)
        out.append({'threshold_mv':th,'record_flag':bool(np.any(frac>0.20)),'max_fraction':float(np.max(frac)),'median_fraction':float(np.median(frac)),**{f'flat_{LEADS[i]}':float(frac[i]) for i in range(12)}})
    return out

def bandpass(x):
    hp=sosfiltfilt(butter(4,0.5,btype='highpass',fs=FS,output='sos'),x,axis=0)
    bp=sosfiltfilt(butter(4,(5,20),btype='bandpass',fs=FS,output='sos'),hp,axis=0)
    pos={k:i for i,k in enumerate(LEADS)}
    comp=sum(bp[:,pos[k]] for k in ('I','II','V4','V5','V6'))-bp[:,pos['aVR']]
    return hp,bp,comp

def serialize(x):
    return json.dumps(x,ensure_ascii=False,separators=(',',':'))

manifest=pd.read_csv(FINAL/'pilot_manifest.csv')
metrics=pd.read_csv(FINAL/'record_metrics.csv')
manifest=manifest.merge(metrics[['ecg_id','read_ok','beat_ok','beat_failure_code','peak_failure_code','sqi_flags','valid_beat_count','rr_median_ms','rr_cv','rr_mad_ms','rr_iqr_ms']],on='ecg_id',how='left')
# Process records once and retain all beat-level payloads.
records={}; event_rows=[]; patient_rows=[]; representative={}; template_dist={}
for n,row in enumerate(manifest.to_dict('records'),1):
    eid=int(row['ecg_id']); rec=load_wfdb_record(Path(row['hea_path']),data_root=DATA)
    entry={'ecg_id':eid,'patient_id':int(row['patient_id']),'strict_label':row['strict_label'],'role':row['role'],'strat_fold':int(row['strat_fold']),'age_years':row.get('age_years',np.nan),'sex':row.get('sex',np.nan),'signal':None,'sqi':None,'peaks':np.array([],dtype=int),'filtered':None,'band':None,'composite':None,'beats':np.empty((0,12,500)),'beat_source_index':np.array([],dtype=int),'left':np.array([],float),'right':np.array([],float),'beat_ok':False,'failure':None}
    if not rec.ok or rec.signal is None:
        entry['failure']=getattr(rec,'failure_code','read_failure'); records[eid]=entry; continue
    signal=rec.signal; entry['signal']=signal; entry['sqi']=sqi_from_wfdb(rec)
    try:
        hp,bp,comp=bandpass(signal); entry['filtered']=hp; entry['band']=bp; entry['composite']=comp
        peak=detect_r_peaks(signal,rec.sampling_rate,rec.lead_names); entry['peaks']=peak.peaks
        if peak.ok:
            beat=extract_beats(signal,peak.peaks,rec.sampling_rate,rec.lead_names)
            entry['beats']=beat.beats; entry['beat_ok']=bool(beat.ok); entry['failure']=beat.failure_code
            if beat.ok:
                p=np.asarray(peak.peaks); valid=[]; left=[]; right=[]
                for i in range(1,len(p)-1):
                    ints=np.diff(p); ok=(300<=ints[i-1]/FS*1000<=2000) and (300<=ints[i]/FS*1000<=2000)
                    if ok:
                        valid.append(i-1); left.append((p[i-1]+p[i])/2); right.append((p[i]+p[i+1])/2)
                entry['beat_source_index']=np.asarray(valid,dtype=int); entry['left']=np.asarray(left); entry['right']=np.asarray(right)
                representative[eid]=beat.representative
                # Per beat metadata and physical R-point amplitude using high-pass (baseline=0).
                lead_idx={k:i for i,k in enumerate(LEADS)}
                for j,src_idx in enumerate(entry['beat_source_index']):
                    peak_i=int(src_idx+1); rp=int(p[peak_i]);
                    rrprev=float((p[peak_i]-p[peak_i-1])/FS*1000); rrnext=float((p[peak_i+1]-p[peak_i])/FS*1000)
                    amps={f'r_amp_{lead}_mv':float(entry['beats'][j,lead_idx[lead],250]) for lead in TARGET}
                    event_rows.append({'ecg_id':eid,'patient_id':entry['patient_id'],'strict_label':entry['strict_label'],'beat_order':j,'source_peak_index':int(src_idx),'r_peak_sample':rp,'r_peak_time_s':rp/FS,'left_boundary_sample':float(entry['left'][j]),'right_boundary_sample':float(entry['right'][j]),'rr_prev_ms':rrprev,'rr_next_ms':rrnext,'valid':True,**amps})
        else: entry['failure']=peak.failure_code
    except Exception as exc:
        entry['failure']=f'exception:{type(exc).__name__}:{exc}'
    records[eid]=entry

# Fixed reference/template distances are computed within each training fold for prediction; descriptive distances use full pilot.
def build_patient_features(ids, train_ids=None, shuffle=False, rng=None):
    ids=list(ids); train_ids=set(ids if train_ids is None else train_ids)
    refs={}
    for lab in ['LVH','NORM']:
        vals=[representative[e] for e in train_ids if e in representative and manifest.loc[manifest.ecg_id.eq(e),'strict_label'].iloc[0]==lab]
        if vals: refs[lab]=np.mean(np.stack(vals),axis=0)
    out=[]
    for eid in ids:
        e=records[eid]; m=manifest.loc[manifest.ecg_id.eq(eid)].iloc[0]; row={'ecg_id':eid,'patient_id':int(m.patient_id),'label':1 if m.strict_label=='LVH' else 0,'strict_label':m.strict_label,'fold':int(m.strat_fold)}
        rep=representative.get(eid); pos={k:i for i,k in enumerate(LEADS)}
        # A: compact morphology of representative V1/V5/V6 plus class-reference distances.
        for lead in TARGET:
            li=pos[lead]
            if rep is None: vals=[]
            else: vals=rep[li]
            for name,val in [('mean',np.mean(vals) if len(vals) else np.nan),('sd',np.std(vals) if len(vals) else np.nan),('min',np.min(vals) if len(vals) else np.nan),('max',np.max(vals) if len(vals) else np.nan),('q25',np.percentile(vals,25) if len(vals) else np.nan),('q75',np.percentile(vals,75) if len(vals) else np.nan)]: row[f'A_{lead}_{name}']=float(val)
            for lab in ['LVH','NORM']:
                if rep is None or lab not in refs: row[f'A_bsw_{lead}_{lab}']=np.nan
                else:
                    a=align_pair(rep[li],refs[lab][li]); b=align_pair(refs[lab][li],rep[li]); row[f'A_bsw_{lead}_{lab}']=float((a.distance+b.distance)/2) if a.ok and b.ok else np.nan
        ev=[x for x in event_rows if x['ecg_id']==eid]
        if shuffle and len(ev)>1:
            order=rng.permutation(len(ev)); ev=[ev[i] for i in order]
        # B: order-invariant beat statistics.
        for lead in TARGET:
            arr=np.array([x[f'r_amp_{lead}_mv'] for x in ev],float)
            st=robust_stats(arr)
            for k,v in st.items(): row[f'B_amp_{lead}_{k}']=v
            d=[]
            for i in range(len(ev)):
                if i>=len(ev)-1: continue
                d.append(abs(arr[i+1]-arr[i]))
            st2=robust_stats(d)
            for k,v in st2.items(): row[f'B_stepamp_{lead}_{k}']=v
        rr=np.array([x['rr_next_ms'] for x in ev],float)
        for k,v in robust_stats(rr).items(): row[f'B_rr_{k}']=v
        # Morphology distances to representative: per beat using BSW-like distance.
        for lead in TARGET:
            li=pos[lead]; ds=[]
            if rep is not None:
                for x in ev:
                    beat=records[eid]['beats'][x['beat_order'],li]
                    a=align_pair(beat,rep[li]); b=align_pair(rep[li],beat); ds.append(float((a.distance+b.distance)/2) if a.ok and b.ok else np.nan)
            for k,v in robust_stats(ds).items(): row[f'B_dist_{lead}_{k}']=v
            # C: only consecutive source peak indices, preserving original order.
            if rep is not None and not shuffle:
                arrd=np.asarray(ds,float); src=np.asarray([x['source_peak_index'] for x in ev],int)
                dif=np.abs(np.diff(arrd))[np.diff(src)==1] if len(arrd)>1 else np.array([])
            else:
                arrd=np.asarray(ds,float); dif=np.abs(np.diff(arrd)) if len(arrd)>1 else np.array([])
            row[f'C_adjdist_{lead}_mean']=float(np.nanmean(dif)) if len(dif) and np.isfinite(dif).any() else np.nan
            row[f'C_adjdist_{lead}_mad']=robust_stats(dif)['mad']
        # C adjacent R amplitude/shape pooled.
        for lead in TARGET:
            arr=np.array([x[f'r_amp_{lead}_mv'] for x in ev],float); src=np.array([x['source_peak_index'] for x in ev],int)
            dif=np.abs(np.diff(arr))[np.diff(src)==1] if len(arr)>1 and not shuffle else (np.abs(np.diff(arr)) if len(arr)>1 else np.array([]))
            row[f'C_adjramp_{lead}_mean']=float(np.nanmean(dif)) if len(dif) else np.nan
        out.append(row)
    return pd.DataFrame(out)

feat=build_patient_features(list(manifest.ecg_id.astype(int)),train_ids=None)
feat.to_csv(TABLE/'patient_features_full.csv',index=False)
pd.DataFrame(event_rows).to_csv(BEATS/'beat_events.csv',index=False)
# Ragged beat waveforms: flat + offsets, with order and provenance.
ids=[]; offsets=[0]; flat=[]
for eid,e in records.items():
    if len(e['beats']): ids.append(eid); flat.append(e['beats']); offsets.append(offsets[-1]+len(e['beats']))
    else: ids.append(eid); offsets.append(offsets[-1])
flat_arr=np.concatenate(flat,axis=0) if flat else np.empty((0,12,500))
np.savez_compressed(BEATS/'beat_waveforms.npz',beats_mv=flat_arr,record_ecg_ids=np.asarray(list(records),int),offsets=np.asarray(offsets,int),lead_names=np.asarray(LEADS),sampling_rate_hz=np.asarray(FS))
# Summary tables.
summary_rows=[]
for eid,e in records.items():
    m=manifest.loc[manifest.ecg_id.eq(eid)].iloc[0]; ev=[x for x in event_rows if x['ecg_id']==eid]
    rr=np.array([x['rr_next_ms'] for x in ev]);
    row={'ecg_id':eid,'patient_id':int(m.patient_id),'strict_label':m.strict_label,'role':m.role,'strat_fold':int(m.strat_fold),'age_years':m.age_years,'sex':m.sex,'read_ok':bool(m.read_ok),'beat_ok':bool(e['beat_ok']),'failure_reason':e['failure'],'peak_count':int(len(e['peaks'])),'valid_beat_count':int(len(ev)),'rr_median_ms':float(np.median(rr)) if len(rr) else np.nan,'rr_sd_ms':float(np.std(rr,ddof=1)) if len(rr)>1 else np.nan,'rr_mad_ms':robust_stats(rr)['mad'],'rr_iqr_ms':robust_stats(rr)['iqr'],'sqi_flags':serialize(list(e['sqi'].flags)) if e['sqi'] else '[]'}
    for lead in TARGET:
        arr=np.array([x[f'r_amp_{lead}_mv'] for x in ev]); st=robust_stats(arr)
        for k,v in st.items(): row[f'r_amp_{lead}_{k}_mv']=v
    summary_rows.append(row)
pd.DataFrame(summary_rows).to_csv(TABLE/'record_summary.csv',index=False)
# SQI threshold sensitivity for all 300; keep default SQI flags separate.
thr_rows=[]
for eid,e in records.items():
    if e['signal'] is None: continue
    for x in recompute_flat_flags(e['signal'],[0.0005,0.001,0.002,0.005]): thr_rows.append({'ecg_id':eid,'strict_label':e['strict_label'],**x})
pd.DataFrame(thr_rows).to_csv(TABLE/'sqi_flatline_threshold_sensitivity.csv',index=False)
# Casebook selection: 10/class plus all failures and probes, deterministic and visible.
sm=pd.DataFrame(summary_rows); selected=[]
for lab in ['LVH','NORM']:
    sub=sm[sm.strict_label.eq(lab)].copy(); sub['priority']=sub['beat_ok'].astype(int)*2 + (sub.sqi_flags!='[]').astype(int); selected.extend(sub.sort_values(['priority','valid_beat_count','ecg_id']).head(10).ecg_id.astype(int).tolist())
selected += sm[~sm.beat_ok].ecg_id.astype(int).tolist(); selected += [358,2858,5783]
selected=list(dict.fromkeys(selected))
case_rows=[]
for eid in selected:
    m=sm[sm.ecg_id.eq(eid)].iloc[0]; reason='beat_failure' if not m.beat_ok else ('probe_missing' if eid in [358,2858,5783] else 'class_quality_sample')
    case_rows.append({'ecg_id':eid,'strict_label':m.strict_label,'patient_id':m.patient_id,'role':m.role,'reason':reason,'failure_reason':m.failure_reason,'sqi_flags':m.sqi_flags,'valid_beat_count':m.valid_beat_count})
case_df=pd.DataFrame(case_rows); case_df.to_csv(CASE/'casebook_review_table.csv',index=False)
# Waveform figures with real signals.
for eid in selected:
    e=records[eid]; m=sm[sm.ecg_id.eq(eid)].iloc[0]
    if e['signal'] is None: continue
    t=np.arange(e['signal'].shape[0])/FS; fig=plt.figure(figsize=(16,13)); gs=fig.add_gridspec(4,1,height_ratios=[3,3,2,2])
    ax=fig.add_subplot(gs[0]); off=np.nanmedian(np.ptp(e['signal'],axis=0))*1.5
    for i,l in enumerate(LEADS): ax.plot(t,e['signal'][:,i]+i*off,lw=.45)
    ax.set_yticks(np.arange(12)*off); ax.set_yticklabels(LEADS); ax.set_title(f'原始12导联 | ecg_id={eid} | 类别={m.strict_label} | 单位=mV, 时间=s')
    ax=fig.add_subplot(gs[1]); ax.plot(t,e['composite'],color='0.25',lw=.6,label='5-20 Hz composite');
    if len(e['peaks']): ax.scatter(e['peaks']/FS,e['composite'][e['peaks']],s=12,c='r',label='R候选峰')
    ax.legend(loc='upper right'); ax.set_ylabel('滤波复合幅度'); ax.set_title(f'检测滤波波形与R峰 | peak_count={len(e["peaks"])} | failure={m.failure_reason}')
    ax=fig.add_subplot(gs[2]); ax.plot(t,e['signal'][:,LEADS.index('V5')],lw=.6,label='V5 raw')
    for j,(src,lft,rgt) in enumerate(zip(e['beat_source_index'],e['left'],e['right'])):
        ax.axvline((src+1)/FS,color='r',lw=.5); ax.axvspan(lft/FS,rgt/FS,color='C0',alpha=.06)
    ax.set_title(f'V5心拍切分边界 | 有效心拍数={len(e["beats"])} | 蓝色=边界区间')
    ax=fig.add_subplot(gs[3]);
    if len(e['beats']):
        for b in e['beats'][:,LEADS.index('V5')]: ax.plot(np.arange(500)/FS-0.5,b,lw=.45,alpha=.35)
        ax.plot(np.arange(500)/FS-0.5,np.mean(e['beats'][:,LEADS.index('V5')],axis=0),lw=2,c='k',label='代表性心拍均值'); ax.legend()
    ax.set_title('V5有效心拍叠加与代表性心拍 | beat window 500 samples | R点=0 s')
    fig.tight_layout(); fig.savefig(FIG/f'case_{eid}_{m.strict_label}.png',dpi=150); plt.close(fig)
# Patient-level distribution figures.
for metric in ['valid_beat_count','rr_median_ms','rr_sd_ms','rr_mad_ms','rr_iqr_ms']:
    fig,ax=plt.subplots(figsize=(7,4));
    for lab,c in [('NORM','C0'),('LVH','C3')]: ax.hist(sm.loc[sm.strict_label.eq(lab),metric].dropna(),bins=20,alpha=.55,label=lab,color=c)
    ax.set_title(metric+' 患者级分布'); ax.set_xlabel(metric); ax.set_ylabel('患者数'); ax.legend(); fig.tight_layout(); fig.savefig(FIG/f'distribution_{metric}.png',dpi=160); plt.close(fig)
# CV prediction: official folds 1-8, train-only imputation/scaling and templates.
base_cols=[c for c in feat.columns if c.startswith('A_')]
b_cols=[c for c in feat.columns if c.startswith('B_')]
c_cols=[c for c in feat.columns if c.startswith('C_')]
results=[]; fold_preds=[]
for fold in range(1,9):
    te=feat[feat.fold.eq(fold)]; tr=feat[feat.fold.ne(fold)]
    if len(te)==0 or len(tr)<10: continue
    for name,cols in [('A',base_cols),('B',base_cols+b_cols),('C',base_cols+b_cols+c_cols)]:
        p,ok=logistic_fit_predict(tr[cols].to_numpy(float),tr.label.to_numpy(int),te[cols].to_numpy(float),l2=1.0)
        results.append({'fold':fold,'model':name,'n_train_patients':len(tr),'n_validation_patients':len(te),'auroc':auc(p,te.label), 'balanced_accuracy':balacc(p,te.label),'optimizer_ok':ok})
        for eid,score,y in zip(te.ecg_id,p,te.label): fold_preds.append({'fold':fold,'model':name,'ecg_id':int(eid),'label':int(y),'score':float(score)})
resdf=pd.DataFrame(results); pred_df=pd.DataFrame(fold_preds); resdf.to_csv(TABLE/'abc_crossval_by_fold.csv',index=False); pred_df.to_csv(TABLE/'abc_crossval_predictions.csv',index=False)
# Aggregate patient-level out-of-fold metrics.
agg=[]
for name,g in pred_df.groupby('model'):
    agg.append({'model':name,'n_patients':len(g),'auroc':auc(g.score,g.label),'balanced_accuracy':balacc(g.score,g.label),'fold_auroc_median':float(resdf[resdf.model.eq(name)].auroc.median()),'fold_auroc_iqr':float(resdf[resdf.model.eq(name)].auroc.quantile(.75)-resdf[resdf.model.eq(name)].auroc.quantile(.25))})
pd.DataFrame(agg).to_csv(TABLE/'abc_crossval_summary.csv',index=False)
# Shuffle control: fixed seeds, same patients and fitted protocol, recompute B/C only.
shuf_rows=[]
for repn,seed in enumerate([7001,7002,7003,7004,7005],1):
    rng=np.random.default_rng(seed); sf=build_patient_features(list(manifest.ecg_id.astype(int)),train_ids=None,shuffle=True,rng=rng)
    for fold in range(1,9):
        te=sf[sf.fold.eq(fold)]; tr=sf[sf.fold.ne(fold)]
        if len(te)==0: continue
        for name,cols in [('B_shuffled',base_cols+b_cols),('C_shuffled',base_cols+b_cols+c_cols)]:
            p,ok=logistic_fit_predict(tr[cols].to_numpy(float),tr.label.to_numpy(int),te[cols].to_numpy(float),l2=1.0)
            shuf_rows.append({'replicate':repn,'seed':seed,'fold':fold,'model':name,'n_validation_patients':len(te),'auroc':auc(p,te.label),'balanced_accuracy':balacc(p,te.label),'optimizer_ok':ok})
pd.DataFrame(shuf_rows).to_csv(TABLE/'abc_shuffle_control_by_fold.csv',index=False)
# Small processing sensitivity: change peak distance and MAD multiplier by an independent diagnostic implementation.
sens=[]
for min_ms in [200,250,300]:
    for mad_mult in [2.0,3.0,4.0]:
        for eid,e in records.items():
            if e['signal'] is None: continue
            pos={k:i for i,k in enumerate(LEADS)}; comp=e['composite']; der=np.diff(comp,prepend=comp[0]); en=np.convolve(der*der,np.ones(max(1,int(round(.12*FS))))/max(1,int(round(.12*FS))),mode='same'); med=np.median(en); mad=np.median(np.abs(en-med)); th=med+mad_mult*mad if mad>0 else med+.1*(np.max(en)-med)
            from scipy.signal import find_peaks
            pk,_=find_peaks(en,height=th,distance=int(round(min_ms/1000*FS))); rr=np.diff(pk)/FS*1000; ok=np.mean((rr>=300)&(rr<=2000))>=.75 if len(rr) else False
            sens.append({'min_peak_distance_ms':min_ms,'mad_multiplier':mad_mult,'ecg_id':eid,'strict_label':e['strict_label'],'peak_count':len(pk),'rr_in_range_fraction':float(np.mean((rr>=300)&(rr<=2000))) if len(rr) else np.nan,'beat_gate_pass':bool(ok)})
sensdf=pd.DataFrame(sens); sensdf.to_csv(TABLE/'peak_filter_sensitivity.csv',index=False)
# Data dictionary and run metadata.
dictionary={'record_summary.csv':'one row per selected patient ECG; mV/ms; patient-level denominator','beat_events.csv':'one row per valid internal beat; sample indices are 0-based, time seconds, amplitude is 0.5-Hz high-pass mV at R-anchored index 250','beat_waveforms.npz':'beats_mv flattened (n_beats,12,500), offsets per record, no interpolation-created extra beats','patient_features_full.csv':'patient-level A/B/C features; adjacent features only when source peak indices are consecutive','sqi_flatline_threshold_sensitivity.csv':'descriptive flag rates; no threshold used as exclusion','abc_crossval_by_fold.csv':'official folds 1-8, patient-level logistic regression; fold 9/10 not read','abc_shuffle_control_by_fold.csv':'within-patient order shuffle control with seeds'}
(OUT/'data_dictionary.json').write_text(json.dumps(dictionary,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'run_config.json').write_text(json.dumps({'seed':SEED,'data_root':str(DATA),'source_final5':str(FINAL),'folds_read':[1,2,3,4,5,6,7,8],'fold9_10_waveforms_read':False,'target_leads':TARGET,'flatline_thresholds_mv':[.0005,.001,.002,.005],'shuffle_seeds':[7001,7002,7003,7004,7005]},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'records':len(records),'events':len(event_rows),'casebook_records':len(selected),'beat_ok':int(sm.beat_ok.sum()),'failures':int((~sm.beat_ok).sum()),'figures':len(list(FIG.glob('*.png'))),'abc_rows':len(resdf)},ensure_ascii=False))



