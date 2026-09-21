import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analysis_core import distance_batch, repair_candidates, sequence_features, fit_predict, templates_for_fold
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"baseline"))
from ecg5001.bsw_like import align_pair

def test_bsw_batch_matches_baseline():
 rng=np.random.default_rng(32); a=rng.normal(size=(4,500)); b=np.roll(a,8,axis=1)*1.2
 got=distance_batch(a,b)
 want=[(align_pair(x,y).distance+align_pair(y,x).distance)/2 for x,y in zip(a,b)]
 np.testing.assert_allclose(got,want,rtol=1e-10,atol=1e-10)

def test_equal_strength_fast_beats_not_removed():
 comp=np.zeros(1500); peaks=np.array([200,340,700,1100]);
 for p in peaks: comp[p]=2
 kept,why=repair_candidates(peaks,comp)
 np.testing.assert_array_equal(kept,peaks)

def test_sequence_mask_and_invariance():
 vectors=np.array([[1,10,100,1000,2,3,4],[2,11,101,1001,3,4,5],[100,50,200,500,4,5,6]],float)
 mask=np.array([True,False]); pair=np.ones((3,3,3)); order=np.array([2,0,1])
 b0,c0,n0=sequence_features(vectors,pair,mask,None); b1,c1,n1=sequence_features(vectors,pair,mask,order)
 np.testing.assert_array_equal(b0,b1); assert n0==n1==1
 assert np.isnan(c0[0])  # one edge is below the declared two-pair minimum

def test_templates_strict_and_validation_labels():
 reps={i:np.full((12,500),float(i)) for i in range(1,8)}
 meta={i:{"label":i%2,"fold":2 if i==7 else 1,"role":"prototype_pool" if i<=4 else "exploratory_probe","vh_ok":True,"patient_id":i} for i in reps}
 refs,ids,status=templates_for_fold(meta,reps,set(range(1,7)),2,"strict",min_strict=2)
 assert ids[0]==[2,4] and ids[1]==[1,3]
 meta[7]["label"]=0
 r2,i2,s2=templates_for_fold(meta,reps,set(range(1,7)),2,"strict",min_strict=2)
 for k in refs: np.testing.assert_array_equal(refs[k],r2[k])
 assert 7 not in sum(ids.values(),[])

def test_imputation_training_only():
 tr=np.array([[1,np.nan],[2,2],[4,4],[5,5]],float); y=np.array([0,0,1,1])
 a=fit_predict(tr,y,np.array([[3,np.nan]]))[0]
 b=fit_predict(tr,y,np.array([[3,np.nan],[1000,1000]]))[0][0:1]
 np.testing.assert_allclose(a,b)

def test_repair_keeps_weak_but_steep_candidate_and_removes_broad_low_one():
 t=np.arange(1200); c=4*np.exp(-((t-200)/5)**2)+.7*np.exp(-((t-345)/20)**2)+4*np.exp(-((t-700)/5)**2)
 p=np.array([200,345,700]); kept,removed=repair_candidates(p,c)
 np.testing.assert_array_equal(kept,[200,700]); assert removed==(345,)

def test_strict_minimum_reports_shortage():
 reps={1:np.zeros((12,500)),2:np.ones((12,500))}; meta={i:{'label':i-1,'role':'prototype_pool','vh_ok':True,'fold':1,'patient_id':i} for i in reps}
 refs,ids,status=templates_for_fold(meta,reps,{1,2},2,'strict')
 assert status=='insufficient' and refs[0] is None and refs[1] is None
