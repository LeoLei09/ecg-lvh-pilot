from __future__ import annotations
from pathlib import Path
import json, hashlib
import numpy as np,pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import periodogram

def sqi_rows(x,eid):
    rows=[]
    for j in range(x.shape[1]):
        z=np.asarray(x[:,j],float); diff=np.abs(np.diff(z)); q=np.percentile(z,[1,99]); flat=diff<.001; qv=np.percentile(diff,[50,75]); rows.append({'ecg_id':int(eid),'lead_index':j,'lead':('I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6')[j],'amplitude_range_p99_p1_mv':float(q[1]-q[0]),'flatline_fraction_diff_lt_0.001mv':float(np.mean(flat)),'longest_flatline_samples':int(max(np.diff(np.r_[[-1],np.flatnonzero(~flat),[len(flat)]])-1)) if len(flat) else 0,'quantized_value_fraction':float(pd.Series(z).value_counts(normalize=True).iloc[0]),'median_abs_diff_mv':float(np.median(diff)),'p95_abs_diff_mv':float(np.percentile(diff,95)),'flag_flatline_gt_0.20':bool(np.mean(flat)>.2)})
    return rows

def write_json(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda y:y.item() if isinstance(y,np.generic) else str(y)),encoding='utf-8')

def _auc(y,s):
 from scipy.stats import rankdata
 y=np.asarray(y,int);s=np.asarray(s,float);n1=y.sum();n0=len(y)-n1
 return float((rankdata(s)[y==1].sum()-n1*(n1+1)/2)/(n1*n0)) if n1 and n0 else np.nan

def _ba(y,s):
 y=np.asarray(y,int);p=np.asarray(s)>=.5
 return float(np.mean([np.mean(p[y==k]==k) for k in [0,1]])) if len(np.unique(y))==2 else np.nan

def _paired(y,a,b,seed=5001):
 rng=np.random.default_rng(seed);idx0=np.flatnonzero(y==0);idx1=np.flatnonzero(y==1);ds=[]
 for _ in range(1000):
  ix=np.r_[rng.choice(idx0,len(idx0),replace=True),rng.choice(idx1,len(idx1),replace=True)]
  ds.append([_auc(y[ix],b[ix])-_auc(y[ix],a[ix]),_ba(y[ix],b[ix])-_ba(y[ix],a[ix])])
 ci=np.percentile(ds,[2.5,97.5],axis=0)
 return dict(delta_auroc=_auc(y,b)-_auc(y,a),auroc_low=ci[0,0],auroc_high=ci[1,0],delta_balanced_accuracy=_ba(y,b)-_ba(y,a),ba_low=ci[0,1],ba_high=ci[1,1])

def metrics_and_report(pred,out,names):
 rows=[];folds=[];deltas=[];wide=[]
 group=['stage','cohort','scheme','seed','model']
 for key,g in pred.groupby(group):
  h=g.dropna(subset=['score']); row=dict(zip(group,key));row.update(n_expected=len(g),n_evaluated=len(h),n_missing=len(g)-len(h),auroc=_auc(h.label,h.score) if len(h) else np.nan,balanced_accuracy=_ba(h.label,h.score) if len(h) else np.nan,coverage='complete' if len(h)==len(g) else 'partial_not_full_CV')
  rows.append(row)
  for fold,z in g.groupby('fold'):
   h=z.dropna(subset=['score']);folds.append(dict(**dict(zip(group,key)),fold=fold,n=len(h),auroc=_auc(h.label,h.score) if len(h) else np.nan,balanced_accuracy=_ba(h.label,h.score) if len(h) else np.nan))
 metric=pd.DataFrame(rows);metric.to_csv(out/'tables/metrics_pooled_oof.csv',index=False);pd.DataFrame(folds).to_csv(out/'tables/metrics_by_fold.csv',index=False)
 for key,g in pred.groupby(['stage','cohort','scheme','seed']):
  q=g.pivot(index=['ecg_id','patient_id','fold','label'],columns='model',values='score').reset_index();base=dict(zip(['stage','cohort','scheme','seed'],key))
  for k,v in base.items():q[k]=v
  wide.append(q); h=q.dropna(subset=['A','B','C']); y=h.label.to_numpy(int)
  if len(np.unique(y))<2:continue
  for a,b in [('A','B'),('B','C')]:
   d=_paired(y,h[a].to_numpy(),h[b].to_numpy());deltas.append(dict(**base,comparison=b+'-'+a,n=len(h),**d))
  if key[3]!=0:
   orig=pred[(pred.stage==key[0])&(pred.cohort==key[1])&(pred.scheme==key[2])&(pred.seed==0)&(pred.model=='C')][['ecg_id','score']]
   h=h.merge(orig,on='ecg_id').dropna(subset=['score']);d=_paired(h.label.to_numpy(int),h.C.to_numpy(),h.score.to_numpy());deltas.append(dict(**base,comparison='original_C-minus-shuffled_C',n=len(h),**d))
 pd.concat(wide).to_csv(out/'predictions/paired_oof_wide.csv',index=False);pd.DataFrame(deltas).to_csv(out/'tables/paired_bootstrap.csv',index=False)
 sh=metric[(metric.seed!=0)&(metric.model=='C')].groupby(['stage','cohort','scheme']).agg(repetitions=('seed','count'),auroc_mean=('auroc','mean'),auroc_sd_over_5_repetitions=('auroc','std'),auroc_min=('auroc','min'),auroc_max=('auroc','max'),ba_mean=('balanced_accuracy','mean'),ba_sd_over_5_repetitions=('balanced_accuracy','std'),n_evaluated=('n_evaluated','min')).reset_index();sh.to_csv(out/'tables/shuffle_summary.csv',index=False)
 defs=[]
 for group,cols in zip(['A','B_addition','C_addition'],names):
  for c in cols:
   if group=='A':meaning='代表性心拍500点上的指定统计量（SD ddof=0，mV）；bsw为代表性心拍到相应类别训练模板的对称BSW-like距离'
   elif group=='B_addition':meaning='逐拍向量的顺序无关统计；mean/SD(ddof=1)/MAD(未乘1.4826)/IQR，至少3个有限心拍；self_reference为本患者固定均值心拍，非类别模板'
   else:meaning='固定原始连续有效边上统计，至少2对；abs_adjacent_difference=|v[j+1]-v[j]|；direct_adjacent_waveform_distance=D(h[j],h[j+1])，不能等同于|D(h[j],ref)-D(h[j+1],ref)|'
   defs.append(dict(feature=c,group=group,definition=meaning))
 pd.DataFrame(defs).to_csv(out/'feature_dictionary.csv',index=False,encoding='utf-8-sig')
 (out/'FEATURES.md').write_text('''# 完整特征定义
A=24维；B=A+28维；C=B+40维，完整逐列清单见 feature_dictionary.csv。
A保留原探索脚本的每导联6个代表性统计和两个类别模板距离。B_stepamp旧4种统计已移入C的amp相邻差4种统计。
逐拍向量为[V1/V5/V6有符号R锚点电压(mV), RR_next(ms), V1/V5/V6到患者固定代表性模板的BSW-like距离]。
幅度基线=0.5Hz四阶零相位高通的零值；锚点不是各导联独立局部R最大值，V1可能为负。
D(a,b)=(d(a→b)+d(b→a))/2，沿用基线100点DTW路径、500点重建、振幅项与时间变形项。BSW-like为独立基线，不称作者实现。
每拍与本记录固定均值心拍距离用于B/C；A的两类模板只由训练患者构成。患者自身均值是该患者记录的无标签表示，非跨患者学习模板。
所有有效向量一起置换，波形按同一索引置换；RR是原事件附着值，不重算时间差；未置换心拍内采样点。
固定位置对应的有效连续边mask来自原始保留峰序号差=1；置换前后mask不变。至少3个有限向量才给B统计，C至少2对；不足为NaN。
失败记录全部A/B/C缺失，训练折每列中位数填补后用训练折有限值SD缩放；全缺失列填0、缩放1。未加失败指示器，未调参/筛特征。
逻辑回归目标=平均交叉熵+0.5*sum(w^2)，截距不惩罚，阈值0.5。与旧代码同一固定正则强度。
500点网格R索引250，每半窗250点，是按相邻RR缩放的相位轴，非真实1秒。真实R时间和中点边界另存。
严格方案少于20个任一类训练原型时整折三模型拒绝评估；保留全部患者行score=NaN，不用填补来伪造该方案。
''',encoding='utf-8')
 # figures
 fig,ax=plt.subplots(figsize=(8,4))
 for stage,color in [('stage1','C0'),('stage2','C3')]:
  z=metric[(metric.stage==stage)&(metric.cohort=='all300')&(metric.scheme=='exploratory')&(metric.seed==0)].set_index('model');ax.plot(['A','B','C'],[z.loc[k,'auroc'] for k in ['A','B','C']],'-o',label=stage,color=color)
 ax.set_ylabel('Pooled patient OOF AUROC');ax.legend();ax.set_title('Corrected pipeline: same 300 patients');fig.tight_layout();fig.savefig(out/'figures/ABC_corrected.png',dpi=160);plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,4))
 for i,stage in enumerate(['stage1','stage2']):
  z=metric[(metric.stage==stage)&(metric.cohort=='all300')&(metric.scheme=='exploratory')&(metric.model=='C')];ax.scatter(np.full(5,i),z[z.seed!=0].auroc,label=f'{stage} shuffled');ax.scatter(i,z[z.seed==0].auroc,marker='*',s=180,c='k')
 ax.set_xticks([0,1],['stage1','stage2']);ax.set_ylabel('Pooled OOF AUROC');ax.set_title('C: original (star) and 5 whole-vector permutations');fig.tight_layout();fig.savefig(out/'figures/shuffle_C.png',dpi=160);plt.close(fig)
 summary=pd.read_csv(out/'tables/record_summary.csv');old=pd.read_csv(Path(__file__).resolve().parents[2]/'ecg-beat-exploration/tables/abc_crossval_summary.csv')
 comparison=metric[(metric.seed==0)&(metric.cohort=='all300')].copy();oldrows=[]
 for r in old.itertuples():oldrows.append(dict(stage='old_invalid',cohort='all300',scheme='leaky_old',seed=0,model=r.model,n_expected=300,n_evaluated=300,n_missing=0,auroc=r.auroc,balanced_accuracy=r.balanced_accuracy,coverage='withdrawn_not_valid_CV'))
 pd.concat([pd.DataFrame(oldrows),comparison]).to_csv(out/'tables/old_vs_corrected.csv',index=False)
 def table(frame,cols):
  f=frame[cols];return '| '+' | '.join(cols)+' |\n|'+'|'.join(['---']*len(cols))+'|\n'+'\n'.join('| '+' | '.join(f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |' for row in f.itertuples(index=False,name=None))
 text=['# ECG-5001 逐搏探索：方法纠错与两阶段重跑','', '本轮纠错不以提高AUROC为目标。旧输出保留；原始数据只读，fold9/10波形未打开，未做GARCH/EGARCH。','', '## 撤回与来源','旧报告的“训练折模板”“B为纯无序统计”“shuffle支持顺序无收益”三项论证撤回。旧A/B/C含标签模板泄漏，旧B含相邻差，shuffle未重算全部C且将5×8折SD误当置换不确定性，故旧性能只作有缺陷的历史数字。','PDF来自 regenerate_ascii_casebook.py 的英文PNG，再由 make_casebook_pdf.py 合成，红线使用pk[i]/500；run_exploration.py的(src+1)/500不是该PDF最终红线来源。报告/v_h表无独立生成入口，当前文件哈希不能冒充历史运行哈希。详见 SOURCE_AUDIT.md、source_provenance.csv。','', '## 两阶段与分析单位','阶段一精确重放旧检测与切拍，并逐条比对final5代表性心拍。阶段二在原候选峰上加入预先记录的低幅低斜率候选筛查：前一保留峰后200–400ms，幅度<45%、±40ms局部差分峰值<50%时删除。250ms候选峰距和RR门槛均未抬高；强弱条件不满足时保留短RR。仍可能删掉真实低幅/宽QRS，也可能漏掉错误峰。该规则利用当前样本反馈，属于开发探索，未独立确认准确率。','每位患者一条记录。主对照300患者，失败记录特征在训练折内填补；共同有效患者敏感性重新在两阶段共同患者上训练/验证。严格方案只允许训练折prototype_pool且12导联v_h<0.3，每类至少20，缺少时不补验证数据，不报全体CV。','',table(summary.groupby(['stage','label']).agg(n=('ecg_id','size'),success=('ok','sum'),beats=('valid_beats','sum')).reset_index(),['stage','label','n','success','beats']),'', '## 合并折外性能（原始顺序）', table(metric[metric.seed==0],['stage','cohort','scheme','model','n_evaluated','n_missing','auroc','balanced_accuracy','coverage']),'','严格方案若只覆盖部分折，上表partial值只是可评估子集，不能与全300探索方案直接排名。','', '## 配对差与顺序置换', table(pd.DataFrame(deltas).query("seed==0 and scheme=='exploratory'"),['stage','cohort','comparison','n','delta_auroc','auroc_low','auroc_high','delta_balanced_accuracy']),'',table(sh[sh.scheme=='exploratory'],['stage','cohort','n_evaluated','auroc_mean','auroc_sd_over_5_repetitions','auroc_min','auroc_max']),'每次置换先合并8折患者预测计算AUROC/BA，再对5次指标汇总；与原始C比较也使用合并OOF。A/B特征及预测逐折逐次保持不变。paired_bootstrap.csv给B−A、C−B及original_C−shuffle_C的配对区间，1000次患者级分层重采样。它固定折外预测，不包含重新训练造成的全部不确定性；5次置换也不足构造精细置换检验。','', '## 波形与质量','旧B_stepamp的4个跨缺失连接（4481、11404、14351、17652）已留表并排除。casebook覆盖全部15条旧失败、全部旧成功例和新增确定性NORM/LVH成功对照；局部图包含真实原始信号、带通composite、检测实际阈值信号（平方导数120ms平滑能量）、能量候选与精化峰。原casebook只显示composite，未显示实际阈值能量，不能把两者混称。','图像初审与算法输出不是人工金标准；manual_review_form.csv的人审核字段空白、状态未审核。失败时候选边界和最终接受心拍边界分开。归一化心拍用采样索引，不映射成固定生理秒。','SQI在0.0005–0.005mV、比例阈值0.20下记录标志率接近饱和，撤回“标志率明显敏感”。逐导联表进一步列连续低差分时长、最长严格重复值时长、众数比例、非QRS低斜率比例，不能据此认定全是假阳性或全不合格。SQI仍未用于排除。','', '## 支持与不支持的结论','新结果只支持本开发样本和当前特征/模型下的增量大小，不能证明时序必然有用或不存在。需要结合C−B配对区间、原始C与五次置换分布，而不是单一最高AUROC。两阶段共同有效患者结果有助区分检测修改与样本组成，但检测器已受本样本视觉反馈，仍不是独立测试。','下一步优先独立标注全部疑似额外峰及成功对照，冻结检测规则后在未读fold9/10另行预注册验证；年龄、性别、噪声、心率仍可能解释组间差异，本轮不作因果或临床归因。','', '## 复现','从 code/fixed_pipeline.py 用 --output 指向全新空目录；README.md给出命令。所有模板、逐拍缓存、特征缺失表、完整OOF及汇总验证均保存。代码哈希、依赖版本和原始输入/旧输出前后哈希随结果交付。']
 (out/'ECG-5001-fixed-report-zh.md').write_text('\n\n'.join(text),encoding='utf-8')

def create_figures_and_review(stages,m,out):
 leads=['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
 failed=[e for e,r in stages['stage1'].items() if not r['ok']]
 oldcases=[2762,1385,18147,19635,3316,7503,358,2858,5783]
 controls=[]
 for label in ['NORM','LVH']:
  eligible=m[m.strict_label.eq(label)&m.ecg_id.map(lambda e:stages['stage1'][e]['ok'])].sort_values('sample_hash')
  controls.extend(eligible.ecg_id.head(5).tolist())
 ids=sorted(set(failed+oldcases+controls)); reviews=[];marks=[];peaktable=[]
 events={s:pd.read_csv(out/'tables'/f'{s}_events.csv') for s in stages}
 for stage in stages:
  for eid,r in stages[stage].items():
   for i,p in enumerate(r['peaks']):
    peaktable.append(dict(stage=stage,ecg_id=eid,index=i,sample=int(p),time_s=p/500,rr_prev_ms=(p-r['peaks'][i-1])*2 if i else np.nan,rr_next_ms=(r['peaks'][i+1]-p)*2 if i+1<len(r['peaks']) else np.nan,status='algorithm_only_not_gold_standard'))
 pd.DataFrame(peaktable).to_csv(out/'casebook/peak_table.csv',index=False)
 with PdfPages(out/'casebook/waveform_casebook.pdf') as pdf:
  for eid in ids:
   a=stages['stage1'][eid];b=stages['stage2'][eid];row=m[m.ecg_id.eq(eid)].iloc[0];t=np.arange(5000)/500
   reviews.append(dict(ecg_id=eid,patient_id=row.patient_id,label=row.strict_label,inclusion='old_failure' if eid in failed else ('requested_old_success' if eid in oldcases else 'hash_selected_success_control'),old_ok=a['ok'],new_ok=b['ok'],old_n_peaks=len(a['peaks']),new_n_peaks=len(b['peaks']),removed_samples=';'.join(map(str,b['removed'])),ai_observation='低幅低斜率后随候选被开发规则删除，需核查是否真实搏动' if len(b['removed']) else '本规则未删峰，不等于检测已确认正确',ai_review_status='AI/算法初审',human_review_status='未审核',human_reviewer='',human_peak_accuracy='',human_notes='',confirmation='待独立确认',questions='核对R误检/漏检、真实早搏或宽QRS、切窗、平线/量化；不可用通过率当准确率'))
   fig,axes=plt.subplots(6,1,figsize=(14,17),gridspec_kw={'height_ratios':[3,1.5,1.5,1.3,1.3,1.8]})
   for j,l in enumerate(leads):axes[0].plot(t,a['signal'][:,j]+j*4,lw=.45)
   axes[0].set_yticks(np.arange(12)*4,leads);axes[0].set_title(f'ecg_id={eid} | {row.strict_label} | raw 12 leads | offsets 4 mV | AI review only');axes[0].set_ylabel('mV + offsets')
   axes[1].plot(t,a['composite'],lw=.6,color='0.3');axes[1].scatter(a['peaks']/500,a['composite'][a['peaks']],s=18,c='r',label='stage1 refined anchors');axes[1].scatter(b['peaks']/500,b['composite'][b['peaks']],s=40,facecolors='none',edgecolors='b',label='stage2 kept');axes[1].legend(fontsize=8);axes[1].set_ylabel('composite mV')
   axes[2].plot(t,a['energy'],lw=.7);axes[2].axhline(a['energy_threshold'],c='r',ls='--',label='median + 3 MAD');axes[2].scatter(a['energy_candidates']/500,a['energy'][a['energy_candidates']],s=12,c='k');axes[2].set_ylabel('120-ms energy\n(mV/sample)^2');axes[2].legend(fontsize=8)
   for ax,stage,r in [(axes[3],'stage1',a),(axes[4],'stage2',b)]:
    ax.plot(t,r['filtered'][:,10],lw=.6);ev=events[stage].query('ecg_id==@eid')
    for e in ev.itertuples():
     if np.isfinite(e.left_sample) and np.isfinite(e.right_sample):
      ax.axvline(e.left_sample/500,color='0.6',ls=':',lw=.4);ax.axvline(e.right_sample/500,color='0.6',ls=':',lw=.4)
      if e.valid:ax.axvspan(e.left_sample/500,e.right_sample/500,alpha=.12,color='green')
    ax.scatter(r['peaks']/500,r['filtered'][r['peaks'],10],s=9,c='b');ax.set_title(f'{stage}: candidate boundaries dotted; ACCEPTED windows green | n={len(r["beats"])} | failure={r["failure"]}',fontsize=9);ax.set_ylabel('V5 mV')
   for j,l in [(6,'V1'),(10,'V5'),(11,'V6')]:
    if len(b['beats']):
     for h in b['beats'][:,j]:axes[5].plot(np.arange(500),h,alpha=.08,lw=.5)
     axes[5].plot(np.arange(500),b['beats'][:,j].mean(0),label=l)
   axes[5].axvline(250,ls='--',color='gray');axes[5].set_xlabel('Normalized beat sample index (0..499); R anchor=250; NOT seconds');axes[5].set_ylabel('mV');
   if len(b['beats']): axes[5].legend(fontsize=8)
   for ax in axes[:5]:ax.set_xlim(0,10);ax.set_xlabel('Original time (s)')
   fig.tight_layout();pdf.savefig(fig);fig.savefig(out/'figures'/f'case_{eid}.png',dpi=100);plt.close(fig)
   # All requested records get zooms; 2762 has explicit windows around both questionable marks.
   windows=[(6.1,7.1),(8.3,9.2)] if eid==2762 else [(max(0,(a['peaks'][0]/500)-.15),min(10,(a['peaks'][0]/500)+2.4))]
   for wi,(lo,hi) in enumerate(windows):
    fig,ax=plt.subplots(5,1,figsize=(14,11),sharex=True)
    for j,l in [(1,'II'),(6,'V1'),(10,'V5'),(11,'V6')]:ax[0].plot(t,a['signal'][:,j]+(j in [10,11])*2,label=l,lw=.8)
    ax[0].legend(ncol=4);ax[0].set_title(f'ecg_id={eid} {row.strict_label}: local inspection {lo:.2f}-{hi:.2f}s; no independent peak annotation');ax[0].set_ylabel('raw mV (+2 V5/V6)')
    ax[1].plot(t,a['composite'],lw=.9);ax[1].set_ylabel('composite mV')
    for stage,r,color in [('stage1',a,'r'),('stage2',b,'b')]:
     ps=r['peaks'][(r['peaks']/500>=lo)&(r['peaks']/500<=hi)]
     ax[1].scatter(ps/500,r['composite'][ps],s=45 if stage=='stage2' else 15,facecolors='none' if stage=='stage2' else color,edgecolors=color,label=stage)
     for k,p in enumerate(ps):
      text=f'{int(p)} @ {p/500:.3f}s';ax[1].annotate(text,(p/500,r['composite'][p]),xytext=(0,10+12*(k%2)),textcoords='offset points',fontsize=6,rotation=20)
      marks.append(dict(ecg_id=eid,figure=f'zoom_{eid}_{wi}.png',stage=stage,sample=int(p),plotted_time_s=p/500))
    ax[1].legend();ax[2].plot(t,a['energy']);ax[2].axhline(a['energy_threshold'],ls='--',c='r');ax[2].scatter(a['energy_candidates']/500,a['energy'][a['energy_candidates']],s=15,c='k');ax[2].set_ylabel('threshold energy')
    for aa,stage,r in [(ax[3],'stage1',a),(ax[4],'stage2',b)]:
     aa.plot(t,r['filtered'][:,10]);ev=events[stage].query('ecg_id==@eid')
     for e in ev.itertuples():
      if np.isfinite(e.left_sample):aa.axvline(e.left_sample/500,color='gray',ls=':',lw=.5)
      if np.isfinite(e.right_sample):aa.axvline(e.right_sample/500,color='gray',ls=':',lw=.5)
      if e.valid:aa.axvspan(e.left_sample/500,e.right_sample/500,color='green',alpha=.15)
     aa.set_ylabel(stage+' V5 mV');aa.set_title('Candidate boundaries dotted; accepted windows green',fontsize=8)
    ax[-1].set_xlim(lo,hi);ax[-1].set_xlabel('Original time (s); sample / 500 Hz');fig.tight_layout();pdf.savefig(fig);fig.savefig(out/'figures'/f'zoom_{eid}_{wi}.png',dpi=130);plt.close(fig)
 pd.DataFrame(reviews).to_csv(out/'casebook/manual_review_form.csv',index=False,encoding='utf-8-sig');pd.DataFrame(marks).to_csv(out/'casebook/plot_peak_coordinates.csv',index=False)
 # SQI: threshold saturation plus continuous runs, exact repeats and non-QRS low slopes.
 sq=[]
 def longest(mask):
  points=np.r_[-1,np.flatnonzero(~mask),len(mask)];return int(np.max(np.diff(points)-1)) if len(mask) else 0
 for eid,r in stages['stage1'].items():
  x=r['signal'];nonqrs=np.ones(4999,bool)
  for p in r['old_peaks']:nonqrs[max(0,p-50):min(4999,p+51)]=False
  for j,l in enumerate(leads):
   d=np.abs(np.diff(x[:,j]));exact=(d==0)
   for th in [.0005,.001,.002,.005]:
    low=d<th;sq.append(dict(ecg_id=eid,lead=l,threshold_mv=th,fraction=float(low.mean()),flag=bool(low.mean()>.2),longest_lowdiff_s=longest(low)/500,longest_exact_repeat_s=longest(exact)/500,exact_repeat_fraction=exact.mean(),nonqrs_lowdiff_fraction=low[nonqrs].mean() if nonqrs.any() else np.nan))
 sq=pd.DataFrame(sq);sq.to_csv(out/'tables/sqi_threshold_lead_runs.csv',index=False)
 rec=sq.groupby(['ecg_id','threshold_mv']).flag.any().reset_index();summary=rec.groupby('threshold_mv').agg(records=('ecg_id','size'),flag_fraction=('flag','mean')).reset_index();summary.to_csv(out/'tables/sqi_record_saturation.csv',index=False)
 sq.groupby(['lead','threshold_mv']).agg(flag_fraction=('flag','mean'),median_fraction=('fraction','median'),median_longest_lowdiff_s=('longest_lowdiff_s','median'),max_longest_lowdiff_s=('longest_lowdiff_s','max'),median_exact_repeat_fraction=('exact_repeat_fraction','median')).reset_index().to_csv(out/'tables/sqi_lead_summary.csv',index=False)
 fig,ax=plt.subplots(figsize=(9,4))
 for th,g in sq.groupby('threshold_mv'):ax.plot(leads,[g[g.lead==l].flag.mean() for l in leads],'-o',label=str(th)+' mV')
 ax.set_ylabel('Flag fraction across 300 records');ax.legend();ax.set_title('SQI: near-saturated record rule; lead-level pattern');fig.tight_layout();fig.savefig(out/'figures/SQI_leads.png',dpi=160);plt.close(fig)


def verify_artifacts(out):
 pred=pd.read_csv(out/'predictions/all_oof.csv');assert pred.groupby(['stage','cohort','scheme','seed','model','patient_id']).size().eq(1).all()
 assert set(pd.read_csv(out/'waveform_read_audit.csv').fold)<=set(range(1,9))
 checked=0
 for f in (out/'templates').glob('*_contributors.csv'):
  df=pd.read_csv(f);meta=pd.read_csv(out/'tables/record_summary.csv').drop_duplicates('ecg_id').set_index('ecg_id')
  for r in df.itertuples():assert meta.loc[r.ecg_id,'fold']!=r.fold;checked+=1
 for key,g in pred[pred.model.isin(['A','B'])].groupby(['stage','cohort','scheme','model','ecg_id']):
  s=g.score.dropna().to_numpy();assert not len(s) or np.all(s==s[0])
 marks=pd.read_csv(out/'casebook/plot_peak_coordinates.csv');peaks=pd.read_csv(out/'casebook/peak_table.csv')
 for r in marks.itertuples():
  assert r.plotted_time_s==r.sample/500
  assert ((peaks.ecg_id==r.ecg_id)&(peaks.stage==r.stage)&(peaks['sample']==r.sample)).any()
 met=pd.read_csv(out/'tables/metrics_pooled_oof.csv')
 for r in met.itertuples():
  g=pred[(pred.stage==r.stage)&(pred.cohort==r.cohort)&(pred.scheme==r.scheme)&(pred.seed==r.seed)&(pred.model==r.model)].dropna(subset=['score'])
  assert len(g)==r.n_evaluated
  if len(g):assert np.isclose(_auc(g.label,g.score),r.auroc) and np.isclose(_ba(g.label,g.score),r.balanced_accuracy)
 for key,g in pred.groupby(['stage','cohort','scheme','seed']):assert g.groupby('model').patient_id.apply(set).apply(lambda s:s==set(g.patient_id)).all()
 checks=dict(unique_patient_oof=True,templates_disjoint=True,template_contributor_rows_checked=checked,AB_predictions_shuffle_invariant=True,plot_times_match_peak_table=True,all_metrics_recomputed=True,fold9_10_not_read=True,input_hashes_unchanged=json.loads((out/'input_before.json').read_text())==json.loads((out/'input_after.json').read_text()))
 assert checks['input_hashes_unchanged'];write_json(out/'verification.json',checks)

