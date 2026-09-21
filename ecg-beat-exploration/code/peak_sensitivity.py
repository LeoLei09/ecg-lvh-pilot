import sys, pathlib, pandas as pd, numpy as np
from scipy.signal import butter,sosfiltfilt,find_peaks
base=pathlib.Path('outputs/ecg-beat-exploration'); sys.path.insert(0,str(base/'code'/'baseline'))
from ecg5001.wfdb_io import load_wfdb_record
m=pd.read_csv(r'D:\UserFiles\desktop\dd\ecg5001-real-output-final5\run-20260918T013319Z-0edaca8ce0fe\pilot_manifest.csv'); root=pathlib.Path(r'D:\UserFiles\desktop\ECGdata\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3')
rows=[]
for _,r in m.iterrows():
 rec=load_wfdb_record(pathlib.Path(r.hea_path),data_root=root)
 if not rec.ok: continue
 x=rec.signal; hp=sosfiltfilt(butter(4,.5,btype='highpass',fs=500,output='sos'),x,axis=0); bp=sosfiltfilt(butter(4,(5,20),btype='bandpass',fs=500,output='sos'),hp,axis=0); pos={k:i for i,k in enumerate(('I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6'))}; comp=sum(bp[:,pos[k]] for k in ('I','II','V4','V5','V6'))-bp[:,pos['aVR']]; der=np.diff(comp,prepend=comp[0]); ker=60; en=np.convolve(der*der,np.ones(ker)/ker,mode='same'); med=np.median(en); mad=np.median(np.abs(en-med))
 for dist in [200,250,300]:
  for mult in [2.,3.,4.]:
   th=med+mult*mad if mad>0 else med+.1*(en.max()-med); pk,_=find_peaks(en,height=th,distance=int(dist/1000*500)); rr=np.diff(pk)/500*1000; frac=np.mean((rr>=300)&(rr<=2000)) if len(rr) else np.nan; rows.append({'ecg_id':int(r.ecg_id),'strict_label':r.strict_label,'min_peak_distance_ms':dist,'mad_multiplier':mult,'peak_count':len(pk),'rr_in_range_fraction':frac,'beat_gate_pass':bool(np.isfinite(frac) and frac>=.75)})
df=pd.DataFrame(rows); df.to_csv(base/'tables/peak_filter_sensitivity.csv',index=False); print(df.groupby(['min_peak_distance_ms','mad_multiplier']).agg(records=('ecg_id','nunique'),pass_rate=('beat_gate_pass','mean'),median_peak_count=('peak_count','median'),median_rr_fraction=('rr_in_range_fraction','median')).reset_index().to_string(index=False)); df.groupby(['min_peak_distance_ms','mad_multiplier']).agg(records=('ecg_id','nunique'),pass_rate=('beat_gate_pass','mean'),median_peak_count=('peak_count','median'),median_rr_fraction=('rr_in_range_fraction','median')).reset_index().to_csv(base/'tables/peak_filter_sensitivity_summary.csv',index=False)
