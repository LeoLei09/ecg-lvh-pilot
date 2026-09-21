from pathlib import Path
import pandas as pd, numpy as np, sys, json
from scipy.optimize import minimize
OUT=Path('outputs/ecg-beat-exploration'); T=OUT/'tables'; B=OUT/'beats'
feat=pd.read_csv(T/'patient_features_full.csv'); ev=pd.read_csv(B/'beat_events.csv')
base=[c for c in feat.columns if c.startswith('A_')]; bcols=[c for c in feat.columns if c.startswith('B_')]; ccols=[c for c in feat.columns if c.startswith('C_')]
def fit(Xtr,ytr,Xte):
 med=np.nanmedian(Xtr,0); med[~np.isfinite(med)]=0; sd=np.nanstd(Xtr,0,ddof=1); sd[~np.isfinite(sd)|(sd<1e-8)]=1; A=np.where(np.isfinite(Xtr),(Xtr-med)/sd,0); B=np.where(np.isfinite(Xte),(Xte-med)/sd,0); A=np.c_[np.ones(len(A)),A]; B=np.c_[np.ones(len(B)),B]
 def loss(w):
  z=np.clip(A@w,-40,40); return np.mean(np.logaddexp(0,z)-ytr*z)+.5*np.sum(w[1:]**2)
 def grad(w):
  z=np.clip(A@w,-40,40); p=1/(1+np.exp(-z)); g=A.T@(p-ytr)/len(A); g[1:]+=w[1:]; return g
 r=minimize(loss,np.zeros(A.shape[1]),jac=grad,method='L-BFGS-B',options={'maxiter':200}); return 1/(1+np.exp(-np.clip(B@r.x,-40,40)))
def auc(s,y):
 s=np.asarray(s); y=np.asarray(y); p=s[y==1]; n=s[y==0]; return np.nan if len(p)==0 or len(n)==0 else float(np.mean((p[:,None]>n)+(p[:,None]==n)*.5))
def ba(s,y):
 z=np.asarray(s)>=.5; y=np.asarray(y); return float(np.mean([np.mean(z[y==k]==k) for k in [0,1] if np.any(y==k)]))
def make_shuf(seed):
 rng=np.random.default_rng(seed); x=feat.copy()
 for eid,g in ev.groupby('ecg_id'):
  idx=rng.permutation(len(g)); gg=g.iloc[idx].reset_index(drop=True)
  for lead in ['V1','V5','V6']:
   a=gg[f'r_amp_{lead}_mv'].to_numpy(float); d=np.abs(np.diff(a));
   for k,v in [('mean',np.mean(a)),('sd',np.std(a,ddof=1) if len(a)>1 else 0),('mad',np.median(np.abs(a-np.median(a)))),('iqr',np.percentile(a,75)-np.percentile(a,25))]: x.loc[x.ecg_id.eq(eid),f'B_amp_{lead}_{k}']=v
   for k,v in [('mean',np.mean(d) if len(d) else np.nan),('sd',np.std(d,ddof=1) if len(d)>1 else 0),('mad',np.median(np.abs(d-np.median(d))) if len(d) else np.nan),('iqr',np.percentile(d,75)-np.percentile(d,25) if len(d) else np.nan)]: x.loc[x.ecg_id.eq(eid),f'B_stepamp_{lead}_{k}']=v
   x.loc[x.ecg_id.eq(eid),f'C_adjramp_{lead}_mean']=np.mean(d) if len(d) else np.nan
  rr=gg.rr_next_ms.to_numpy(float); x.loc[x.ecg_id.eq(eid),'B_rr_mean']=np.nanmean(rr); x.loc[x.ecg_id.eq(eid),'B_rr_sd']=np.nanstd(rr,ddof=1) if len(rr)>1 else 0; x.loc[x.ecg_id.eq(eid),'B_rr_mad']=np.nanmedian(np.abs(rr-np.nanmedian(rr))); x.loc[x.ecg_id.eq(eid),'B_rr_iqr']=np.nanpercentile(rr,75)-np.nanpercentile(rr,25)
 return x
rows=[]
for rep,seed in enumerate([7001,7002,7003,7004,7005],1):
 x=make_shuf(seed)
 for fold in range(1,9):
  tr=x[x.fold.ne(fold)]; te=x[x.fold.eq(fold)]
  for name,cols in [('B_shuffled',base+bcols),('C_shuffled',base+bcols+ccols)]:
   p=fit(tr[cols].to_numpy(float),tr.label.to_numpy(int),te[cols].to_numpy(float)); rows.append({'replicate':rep,'seed':seed,'fold':fold,'model':name,'n_validation_patients':len(te),'auroc':auc(p,te.label),'balanced_accuracy':ba(p,te.label)})
pd.DataFrame(rows).to_csv(T/'abc_shuffle_control_by_fold.csv',index=False)
# aggregate simple processing sensitivity from existing record metrics and alternate thresholds recorded by a direct rerun is omitted if no table; report default only
agg=pd.DataFrame(rows).groupby('model').agg(n=('auroc','count'),auroc_mean=('auroc','mean'),auroc_sd=('auroc','std'),balacc_mean=('balanced_accuracy','mean'),balacc_sd=('balanced_accuracy','std')).reset_index(); agg.to_csv(T/'abc_shuffle_control_summary.csv',index=False)
# threshold aggregate
s=pd.read_csv(T/'sqi_flatline_threshold_sensitivity.csv'); a=s.groupby('threshold_mv').agg(records=('ecg_id','nunique'),flag_rate=('record_flag','mean'),median_max_fraction=('max_fraction','median'),q25_max_fraction=('max_fraction',lambda x:x.quantile(.25)),q75_max_fraction=('max_fraction',lambda x:x.quantile(.75))).reset_index(); a.to_csv(T/'sqi_flatline_threshold_summary.csv',index=False)
# peak sensitivity aggregate\nps=pd.read_csv(T/'peak_filter_sensitivity.csv') if (T/'peak_filter_sensitivity.csv').exists() else pd.DataFrame()\nif not ps.empty:\n    pa=ps.groupby(['min_peak_distance_ms','mad_multiplier']).agg(records=('ecg_id','nunique'),pass_rate=('beat_gate_pass','mean'),median_peak_count=('peak_count','median'),median_rr_fraction=('rr_in_range_fraction','median')).reset_index(); pa.to_csv(T/'peak_filter_sensitivity_summary.csv',index=False)\n# class summaries / effect sizes
rs=pd.read_csv(T/'record_summary.csv'); out=[]
for metric in ['valid_beat_count','rr_median_ms','rr_sd_ms','rr_mad_ms','rr_iqr_ms','r_amp_V1_sd_mv','r_amp_V5_sd_mv','r_amp_V6_sd_mv']:
 g=[]
 for lab in ['NORM','LVH']:
  z=rs.loc[rs.strict_label.eq(lab),metric].dropna(); g.append((lab,len(z),z.mean(),z.median(),z.std(ddof=1)))
 out.append({'metric':metric,'norm_n':g[0][1],'norm_mean':g[0][2],'norm_median':g[0][3],'norm_sd':g[0][4],'lvh_n':g[1][1],'lvh_mean':g[1][2],'lvh_median':g[1][3],'lvh_sd':g[1][4]})
pd.DataFrame(out).to_csv(T/'patient_group_descriptives.csv',index=False)
print(agg.to_string(index=False)); print(a.to_string(index=False))


