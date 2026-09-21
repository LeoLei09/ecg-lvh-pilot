from pathlib import Path
import sys,pandas as pd,numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.signal import butter,sosfiltfilt
sys.path.insert(0,'outputs/ecg-beat-exploration/code/baseline'); from ecg5001.wfdb_io import load_wfdb_record; from ecg5001.beats import detect_r_peaks,extract_beats
DATA=Path(r'D:\UserFiles\desktop\ECGdata\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'); p=Path('outputs/ecg-beat-exploration'); sm=pd.read_csv(p/'tables/record_summary.csv'); case=pd.read_csv(p/'casebook/casebook_review_table.csv'); FS=500.; leads=('I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6'); pos={k:i for i,k in enumerate(leads)}; figdir=p/'figures'
for eid in case.ecg_id:
 r=sm[sm.ecg_id.eq(eid)].iloc[0]; hea=pd.read_csv(r'D:\UserFiles\desktop\dd\ecg5001-real-output-final5\run-20260918T013319Z-0edaca8ce0fe\pilot_manifest.csv').query('ecg_id==@eid').iloc[0].hea_path; rec=load_wfdb_record(Path(hea),data_root=DATA)
 if not rec.ok: continue
 x=rec.signal; hp=sosfiltfilt(butter(4,.5,btype='highpass',fs=FS,output='sos'),x,axis=0); bp=sosfiltfilt(butter(4,(5,20),btype='bandpass',fs=FS,output='sos'),hp,axis=0); comp=sum(bp[:,pos[k]] for k in ('I','II','V4','V5','V6'))-bp[:,pos['aVR']]; pk=detect_r_peaks(x,FS,rec.lead_names).peaks; b=extract_beats(x,pk,FS,rec.lead_names); t=np.arange(5000)/FS; fig=plt.figure(figsize=(16,13)); gs=fig.add_gridspec(4,1,height_ratios=[3,3,2,2]); ax=fig.add_subplot(gs[0]); off=max(np.median(np.ptp(x,axis=0))*1.5,.1)
 for i,l in enumerate(leads): ax.plot(t,x[:,i]+i*off,lw=.45)
 ax.set_yticks(np.arange(12)*off); ax.set_yticklabels(leads); ax.set_title(f'Raw 12-lead ECG | ecg_id={eid} | label={r.strict_label} | amplitude=mV | time=s')
 ax=fig.add_subplot(gs[1]); ax.plot(t,comp,color='0.25',lw=.6); ax.scatter(pk/FS,comp[pk],s=12,c='r'); ax.set_title(f'Detection bandpass composite and R candidates | peaks={len(pk)} | failure={r.failure_reason}')
 ax=fig.add_subplot(gs[2]); ax.plot(t,x[:,pos['V5']],lw=.6); ax.set_title(f'V5 segmentation boundaries | valid_beats={int(r.valid_beat_count)} | units=mV,s')
 for i in range(1,len(pk)-1): ax.axvline(pk[i]/FS,color='r',lw=.4); ax.axvspan((pk[i-1]+pk[i])/(2*FS),(pk[i]+pk[i+1])/(2*FS),color='C0',alpha=.05)
 ax=fig.add_subplot(gs[3]);
 if b.ok:
  for z in b.beats[:,pos['V5']]: ax.plot(np.arange(500)/FS-.5,z,lw=.4,alpha=.3)
  ax.plot(np.arange(500)/FS-.5,b.representative[pos['V5']],lw=2,c='k')
 ax.set_title('V5 valid-beat overlay and representative beat | 500 samples | R=0 s'); fig.tight_layout(); fig.savefig(figdir/f'case_{eid}_{r.strict_label}.png',dpi=150); plt.close(fig)
print('done')
