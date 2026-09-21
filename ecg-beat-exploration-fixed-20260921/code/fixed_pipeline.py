"""Reproducible two-stage correction; run with --output pointing to a NEW directory."""
import argparse,sys,json,hashlib,platform,warnings
from pathlib import Path
import numpy as np,pandas as pd
from scipy.signal import butter,sosfiltfilt,find_peaks
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'baseline'))
from ecg5001.wfdb_io import load_wfdb_record,validate_ptbxl_500hz_record
from ecg5001.beats import detect_r_peaks,extract_beats,ventricular_heterogeneity
from analysis_core import distance_batch,repair_candidates,sequence_features,templates_for_fold,fit_predict
LEADS=('I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6'); TARGET=[6,10,11]
BASE_DEFAULT=Path(r'D:\UserFiles\desktop\dd\ecg5001-real-output-final5\run-20260918T013319Z-0edaca8ce0fe')
DATA_DEFAULT=Path(r'D:\UserFiles\desktop\ECGdata\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3')
OLD=HERE.parents[1]/'ecg-beat-exploration'
SEEDS=[0,7001,7002,7003,7004,7005]

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def manifest_hashes(paths): return {str(p):{'sha256':digest(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in sorted(set(paths))}
def write_json(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda y:y.item() if isinstance(y,np.generic) else str(y)),encoding='utf-8')
def save(p,rows): pd.DataFrame(rows).to_csv(p,index=False,encoding='utf-8-sig')

def process(x,stage,eid):
 old=detect_r_peaks(x,500,LEADS); comp=old.composite
 d=np.diff(comp,prepend=comp[0]); energy=np.convolve(d*d,np.ones(60)/60,mode='same')
 med=np.median(energy); mad=np.median(np.abs(energy-med)); threshold=med+3*mad if mad>0 else med+.1*(energy.max()-med)
 energy_candidates=find_peaks(energy,height=threshold,distance=125)[0]
 peaks,removed=(old.peaks,()) if stage=='stage1' else repair_candidates(old.peaks,comp)
 rr=np.diff(peaks)*2.; gate=len(peaks)>=5 and np.mean((rr>=300)&(rr<=2000))>=.75
 result=extract_beats(x,peaks,500,LEADS) if gate else None
 ok=bool(result and result.ok); failure='' if ok else (result.failure_code if result else ('insufficient_peaks' if len(peaks)<5 else 'rr_out_of_range'))
 events=[]; beats=[]; sources=[]; rrs=[]
 for i,p in enumerate(peaks):
  left=(peaks[i-1]+p)/2 if i>0 else np.nan; right=(p+peaks[i+1])/2 if i+1<len(peaks) else np.nan
  reason=''; h=None
  if i==0 or i==len(peaks)-1: reason='edge_no_two_neighbors'
  elif not gate: reason=failure
  elif not(300<=rr[i-1]<=2000 and 300<=rr[i]<=2000): reason='neighbor_rr_out_of_range'
  else:
   coords=np.r_[np.linspace(left,p,250,endpoint=False),np.linspace(p,right,250,endpoint=False)]
   h=np.vstack([np.interp(coords,np.arange(len(x)),old.filtered[:,j]) for j in range(12)])
   if np.max(np.abs(h))>20: reason='gross_artifact_gt20mV'
   elif not ok: reason=failure
  valid=not reason
  bi=len(beats) if valid else -1
  if valid: beats.append(h); sources.append(i); rrs.append(rr[i])
  events.append({'stage':stage,'ecg_id':eid,'source_peak_index':i,'r_peak_sample':int(p),'r_peak_time_s':p/500,'left_sample':left,'right_sample':right,'rr_prev_ms':rr[i-1] if i>0 else np.nan,'rr_next_ms':rr[i] if i<len(rr) else np.nan,'valid':valid,'failure_reason':reason,'beat_index':bi})
 b=np.asarray(beats).reshape(-1,12,500)
 if ok: np.testing.assert_allclose(b,result.beats,rtol=0,atol=1e-12)
 vh=ventricular_heterogeneity(b).per_lead_vh if len(b)>=3 else np.full(12,np.nan)
 return dict(ecg_id=eid,stage=stage,old_peaks=old.peaks,peaks=peaks,removed=np.asarray(removed,int),signal=x,filtered=old.filtered,composite=comp,energy=energy,energy_candidates=energy_candidates,energy_threshold=threshold,events=events,beats=b,source=np.asarray(sources,int),rr=np.asarray(rrs),ok=ok,failure=failure,vh=vh)

def make_sequence_cache(r):
 n=len(r['beats']); pair=np.full((n,n,3),np.nan); rep=r['beats'].mean(0) if n else None
 v=np.full((n,7),np.nan)
 if n:
  v[:,:3]=r['beats'][:,TARGET,250]; v[:,3]=r['rr']
  a=r['beats'][:,TARGET,:].reshape(-1,500); b=np.broadcast_to(rep[TARGET],(n,3,500)).reshape(-1,500)
  v[:,4:]=distance_batch(a,b).reshape(n,3)
  ii,jj=np.triu_indices(n,1)
  if len(ii):
   distances=distance_batch(r['beats'][ii][:,TARGET].reshape(-1,500),r['beats'][jj][:,TARGET].reshape(-1,500)).reshape(-1,3)
   pair[ii,jj]=distances; pair[jj,ii]=distances
  for i in range(n):pair[i,i]=0
 mask=np.diff(r['source'])==1
 cache={}; orders={}
 for seed in SEEDS:
  order=np.arange(n) if seed==0 else np.random.default_rng(np.random.SeedSequence([seed,r['ecg_id']])).permutation(n)
  cache[seed]=sequence_features(v,pair,mask,order); orders[seed]=order
  np.testing.assert_array_equal(cache[seed][0],cache[0][0]); assert cache[seed][2]==cache[0][2]
 return dict(vectors=v,pairwise=pair,mask=mask,features=cache,orders=orders)

def morphology_features(ids,reps,refs):
 rows=[]
 for eid in ids:
  if eid not in reps:rows.append(np.full(24,np.nan));continue
  a=[]
  for lead in TARGET:
   x=reps[eid][lead]; a.extend([x.mean(),x.std(),x.min(),x.max(),np.percentile(x,25),np.percentile(x,75),np.nan,np.nan])
  rows.append(a)
 rows=np.asarray(rows,float)
 good=[i for i,e in enumerate(ids) if e in reps]
 for lab in [0,1]:
  if refs[lab] is None: continue
  a=np.asarray([reps[ids[i]][TARGET] for i in good]).reshape(-1,500)
  b=np.broadcast_to(refs[lab][TARGET],(len(good),3,500)).reshape(-1,500)
  ds=distance_batch(a,b).reshape(-1,3)
  for j in range(3): rows[good,j*8+6+(1-lab)]=ds[:,j]
 return rows

def feature_names():
 a=[]
 for lead in ['V1','V5','V6']: a += [f'A_{lead}_{k}' for k in ['mean','sd','min','max','q25','q75','bsw_LVH','bsw_NORM']]
 vector_names=['amp_V1','amp_V5','amp_V6','rr_next','self_reference_distance_V1','self_reference_distance_V5','self_reference_distance_V6']
 b=[f'B_{v}_{s}' for v in vector_names for s in ['mean','sd','mad','iqr']]
 c=[f'C_abs_adjacent_difference_{v}_{s}' for v in vector_names for s in ['mean','sd','mad','iqr']]+[f'C_direct_adjacent_waveform_distance_{l}_{s}' for l in ['V1','V5','V6'] for s in ['mean','sd','mad','iqr']]
 return a,b,c

def evaluate(stage,records,cache,meta,ids,out,cohort):
 pred=[]; contributors=[]; template_status=[]; checks=[]; feature_missing=[]; names=sum(feature_names(),[])
 reps={e:r['beats'].mean(0) for e,r in records.items() if r['ok']}
 for scheme in ['exploratory','strict']:
  for fold in range(1,9):
   tr=[e for e in ids if meta[e]['fold']!=fold]; te=[e for e in ids if meta[e]['fold']==fold]
   refs,cont,status=templates_for_fold(meta,reps,set(tr),fold,scheme)
   for lab in [0,1]:
    template_status.append(dict(stage=stage,cohort=cohort,scheme=scheme,fold=fold,label=lab,count=len(cont[lab]),minimum=20 if scheme=='strict' else 1,status='ok' if refs[lab] is not None else 'insufficient'))
    for e in cont[lab]:
     assert e in tr and e not in te
     contributors.append(dict(stage=stage,cohort=cohort,scheme=scheme,fold=fold,label=lab,ecg_id=e,patient_id=meta[e]['patient_id']))
   if status!='ok':
    for seed in SEEDS:
     for model in ['A','B','C']:
      for e in te:pred.append(dict(stage=stage,cohort=cohort,scheme=scheme,fold=fold,seed=seed,model=model,ecg_id=e,patient_id=meta[e]['patient_id'],label=meta[e]['label'],score=np.nan,status='insufficient_train_prototypes'))
    continue
   # Save actual training-only templates; no old class templates are loaded.
   np.savez_compressed(out/'templates'/f'{stage}_{cohort}_{scheme}_{fold}.npz',NORM=refs[0],LVH=refs[1],NORM_ecg_ids=cont[0],LVH_ecg_ids=cont[1])
   allids=tr+te; A=morphology_features(allids,reps,refs); B=np.asarray([cache[e]['features'][0][0] for e in allids])
   ytr=np.asarray([meta[e]['label'] for e in tr]); ntr=len(tr)
   changed={e:dict(v) for e,v in meta.items()}
   for e in te:changed[e]['label']=1-changed[e]['label']
   refs2,cont2,status2=templates_for_fold(changed,reps,set(tr),fold,scheme)
   for lab in [0,1]:np.testing.assert_array_equal(refs[lab],refs2[lab])
   A2=morphology_features(allids,reps,refs2); np.testing.assert_array_equal(A,A2)
   baseline_pred={}
   for seed in SEEDS:
    C=np.asarray([cache[e]['features'][seed][1] for e in allids]); X=np.c_[A,B,C]
    assert X.shape[1]==len(names)
    for model,end in [('A',len(A[0])),('B',A.shape[1]+B.shape[1]),('C',X.shape[1])]:
     for i,e in enumerate(te):feature_missing.append(dict(stage=stage,cohort=cohort,scheme=scheme,fold=fold,seed=seed,model=model,ecg_id=e,n_features=end,n_missing=int(np.isnan(X[ntr+i,:end]).sum())))
     probs,ok,med,scale=fit_predict(X[:ntr,:end],ytr,X[ntr:,:end]); assert ok
     if seed==0:
      baseline_pred[model]=probs
      p2=fit_predict(np.c_[A2,B,C][:ntr,:end],ytr,np.c_[A2,B,C][ntr:,:end])[0]
      np.testing.assert_array_equal(probs,p2)
     elif model in ['A','B']:np.testing.assert_array_equal(probs,baseline_pred[model])
     for e,pr in zip(te,probs):pred.append(dict(stage=stage,cohort=cohort,scheme=scheme,fold=fold,seed=seed,model=model,ecg_id=e,patient_id=meta[e]['patient_id'],label=meta[e]['label'],score=pr,status='ok'))
    if seed==0:
     df=pd.DataFrame(X,columns=names); df.insert(0,'ecg_id',allids); df.insert(1,'role_in_fold',['train']*ntr+['validation']*len(te)); df.to_csv(out/'features'/f'{stage}_{cohort}_{scheme}_{fold}.csv',index=False)
   checks.append(dict(stage=stage,cohort=cohort,scheme=scheme,fold=fold,disjoint=True,validation_label_invariance=True,AB_feature_and_prediction_shuffle_invariance=True))
   print(stage,cohort,scheme,'fold',fold,'done',flush=True)
 save(out/'predictions'/f'{stage}_{cohort}.csv',pred)
 save(out/'templates'/f'{stage}_{cohort}_contributors.csv',contributors); save(out/'templates'/f'{stage}_{cohort}_status.csv',template_status)
 save(out/'checks'/f'{stage}_{cohort}.csv',checks); save(out/'features'/f'{stage}_{cohort}_missing.csv',feature_missing)
 return pd.DataFrame(pred)

def source_audit(out,final):
 paths=list(OLD.rglob('*')); paths=[p for p in paths if p.is_file() and '__pycache__' not in str(p)]
 audit=[dict(path=str(p),sha256=digest(p),bytes=p.stat().st_size) for p in paths]
 save(out/'source_provenance.csv',audit)
 (out/'SOURCE_AUDIT.md').write_text('# 上轮来源核查\n`run_exploration.py` 在分折前建立全样本标签模板，并含错误红线 `(src+1)/FS`。\n现有英文 PNG 与 `regenerate_ascii_casebook.py` 一致；后者用 `pk[i]/FS`，`make_casebook_pdf.py` 将这些 PNG 嵌入 PDF。\nABC 表最可能由主脚本生成；shuffle 表由 `postprocess.py` 重写，该脚本未重算全部 C，且汇总的是5×8折。\n报告、v_h表、数据字典曾由会话内一次性代码生成，缺少完整独立入口。当前哈希仅证明审计时文件状态，不冒充历史运行代码哈希；来源关联由代码、内容和会话记录支持，不是字节级历史重现证明。\n旧报告关于训练折模板、纯顺序无关 B、正确 shuffle 和 SQI“明显敏感”的结论撤回。旧性能数字仅列作有缺陷的历史结果。\n本次所有正文/表/图由所交代码生成；初期未完成草稿已隔离于 work，不作结果。\n',encoding='utf-8')
 oldevents=pd.read_csv(OLD/'beats/beat_events.csv'); gaps=[]
 for e,g in oldevents.groupby('ecg_id'):
  for i in range(1,len(g)):
   if g.iloc[i].source_peak_index-g.iloc[i-1].source_peak_index!=1:gaps.append(dict(ecg_id=e,previous_source_peak_index=int(g.iloc[i-1].source_peak_index),next_source_peak_index=int(g.iloc[i].source_peak_index)))
 save(out/'old_B_stepamp_crossed_gaps.csv',gaps)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--data-root',type=Path,default=DATA_DEFAULT);parser.add_argument('--final5',type=Path,default=BASE_DEFAULT);args=parser.parse_args();out=args.output.resolve()
 if out.exists() and any(out.iterdir()):raise ValueError('Output must be NEW/empty; prior outputs never overwritten')
 if out==args.data_root.resolve() or args.data_root.resolve() in out.parents:raise ValueError('Output inside raw data prohibited')
 for sub in ['beats','templates','features','predictions','checks','figures','casebook','tables']: (out/sub).mkdir(parents=True,exist_ok=True)
 source_audit(out,args.final5)
 m=pd.read_csv(args.final5/'pilot_manifest.csv').sort_values('ecg_id'); assert len(m)==300 and m.patient_id.nunique()==300 and m.strat_fold.between(1,8).all()
 originals=list(args.final5.rglob('*'))+list(OLD.rglob('*')); originals=[p for p in originals if p.is_file()]
 raw=[args.data_root/'ptbxl_database.csv',args.data_root/'scp_statements.csv']+[Path(p) for col in ['hea_path','dat_path'] for p in m[col]]
 before=manifest_hashes(raw+originals);write_json(out/'input_before.json',before)
 config=dict(shuffle_seeds=SEEDS,folds=list(range(1,9)),model='L2 logistic mean loss + 0.5 sum(w^2), intercept unpenalized, threshold=0.5',strict_minimum=20,vh_threshold=.3,minimum_beats=3,minimum_pairs=2,detector='original 250ms; stage2 rejects post-candidate 200-400ms when amplitude<0.45 and slope<0.50 of previous retained candidate; no RR relaxation',distance='exact symmetric BSW-like baseline, batched equivalent',templates='exploratory all available training representatives; strict training prototype_pool and all12 vh<0.3',data_root=str(args.data_root),final5=str(args.final5),output=str(out))
 write_json(out/'run_config.json',config)
 stage_records={'stage1':{},'stage2':{}}; sqi=[]; reads=[]; oldnpz=np.load(args.final5/'representative_beats.npz'); oldreps={int(e):b for e,b in zip(oldnpz['ecg_ids'],oldnpz['beats_mv'])}
 for row in m.itertuples():
  assert row.strat_fold in range(1,9)
  rec=load_wfdb_record(row.hea_path,data_root=args.data_root); assert validate_ptbxl_500hz_record(rec).ok
  reads.append(dict(ecg_id=row.ecg_id,fold=row.strat_fold,path=row.hea_path,access='read_only'))
  x=rec.signal
  for stage in stage_records:
   r=process(x,stage,int(row.ecg_id));stage_records[stage][int(row.ecg_id)]=r
   if stage=='stage1':
    assert r['ok']==(int(row.ecg_id) in oldreps)
    if r['ok']:np.testing.assert_allclose(r['beats'].mean(0),oldreps[int(row.ecg_id)],rtol=0,atol=1e-12)
  from deliverables import sqi_rows
  sqi.extend(sqi_rows(x,row.ecg_id))
 save(out/'waveform_read_audit.csv',reads);save(out/'tables/sqi_lead_detail.csv',sqi)
 common=[e for e in m.ecg_id if all(stage_records[s][e]['ok'] for s in stage_records)]
 summaries=[];allpred=[]
 for stage,records in stage_records.items():
  events=[]; cache={};meta={};orders=[]
  for row in m.itertuples():
   eid=int(row.ecg_id);r=records[eid]; events.extend(r['events']);cache[eid]=make_sequence_cache(r)
   meta[eid]=dict(patient_id=row.patient_id,fold=row.strat_fold,label=int(row.strict_label=='LVH'),role=row.role,vh_ok=bool(np.all(r['vh']<.3)))
   summaries.append(dict(stage=stage,ecg_id=eid,patient_id=row.patient_id,label=row.strict_label,fold=row.strat_fold,role=row.role,age=row.age_years,sex=row.sex,ok=r['ok'],failure=r['failure'],peaks=len(r['peaks']),valid_beats=len(r['beats']),adjacent_pairs=int(cache[eid]['mask'].sum()),removed_candidates=len(r['removed']),vh_max=float(np.max(r['vh']))))
   np.savez_compressed(out/'beats'/f'{stage}_{eid}.npz',beats_mv=r['beats'],source_peak_indices=r['source'],peaks_samples=r['peaks'],old_peaks_samples=r['old_peaks'],removed_samples=r['removed'],vectors=cache[eid]['vectors'],pairwise_BSW=cache[eid]['pairwise'],edge_mask=cache[eid]['mask'],lead_names=LEADS)
   for seed in SEEDS:
    for pos,origin in enumerate(cache[eid]['orders'][seed]):orders.append(dict(stage=stage,seed=seed,ecg_id=eid,valid_position=pos,vector_origin=int(origin)))
  save(out/'tables'/f'{stage}_events.csv',events);save(out/'tables'/f'{stage}_shuffle_orders.csv',orders)
  print(stage,'extraction and distance cache ready',flush=True)
  for cohort,ids in [('all300',list(m.ecg_id)),('common_valid',common)]:allpred.append(evaluate(stage,records,cache,meta,ids,out,cohort))
 save(out/'tables/record_summary.csv',summaries)
 predictions=pd.concat(allpred,ignore_index=True);predictions.to_csv(out/'predictions/all_oof.csv',index=False)
 from deliverables import create_figures_and_review,metrics_and_report,verify_artifacts
 create_figures_and_review(stage_records,m,out)
 metrics_and_report(predictions,out,feature_names())
 after=manifest_hashes(raw+originals); assert before==after;write_json(out/'input_after.json',after)
 write_json(out/'dependency_snapshot.json',dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=__import__('scipy').__version__,matplotlib=__import__('matplotlib').__version__))
 files=sorted(HERE.rglob('*.py'));write_json(out/'code_hashes.json',{str(p.relative_to(HERE)):digest(p) for p in files})
 verify_artifacts(out)
 print('COMPLETE',out,flush=True)
if __name__=='__main__': main()
