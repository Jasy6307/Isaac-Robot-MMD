"""Temporal diagnostics only: no filtering is applied to animation or video."""
from pathlib import Path
import argparse,json
import numpy as np
from scipy.signal import savgol_filter,welch
from pxr import Usd,UsdSkel
p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--frames',type=int,default=7011);a=p.parse_args()
s=Usd.Stage.Open(str(a.cache.resolve()));sk=UsdSkel.Skeleton.Get(s,'/Character/Skeleton');names=list(sk.GetJointNamesAttr().Get());ids=[i for i,n in enumerate(names) if n.startswith('qun_') and int(n.split('_')[1])>0];hip=names.index('下半身');cache=UsdSkel.Cache();cache.Populate(UsdSkel.Root.Get(s,'/Character'),Usd.PrimDefaultPredicate);q=cache.GetSkelQuery(sk)
x=[]
for f in range(min(a.frames,int(s.GetEndTimeCode())+1)):
 mats=q.ComputeJointSkelTransforms(f);points=np.array([mats[i].ExtractTranslation() for i in ids]);inv=np.linalg.inv(np.array(mats[hip]));x.append(np.c_[points,np.ones(len(points))]@inv)
x=np.asarray(x)[:,:,:3];smooth=savgol_filter(x,9,2,axis=0);hp=x-smooth;acc=np.diff(x,n=2,axis=0)*900.;slice_=slice(30,-4)
h=np.linalg.norm(hp,axis=-1)[slice_];am=np.linalg.norm(acc,axis=-1)[slice_];perframe=np.sqrt(np.mean(h*h,axis=1));freq,psd=welch(x[30:],fs=30,axis=0,nperseg=min(256,len(x)-30));band=(freq>=6)&(freq<=14)
r=dict(frames=len(x),diagnostic='Pelvis-local node high-pass residual: Savitzky-Golay 9 frames, order 2; no output filtering',metrics=dict(highpass_rms_mm=float(np.sqrt(np.mean(h*h))*1000),highpass_p99_mm=float(np.percentile(h,99)*1000),accel_rms_m_s2=float(np.sqrt(np.mean(am*am))),accel_p99_m_s2=float(np.percentile(am,99)),accel_p999_m_s2=float(np.percentile(am,99.9)),accel_max_m_s2=float(am.max()),highpass_max_mm=float(h.max()*1000),band_6_14_hz_power=float(np.mean(psd[band]))),worst_frames=(np.argsort(perframe)[-12:]+30).tolist())
a.output.parent.mkdir(exist_ok=True,parents=True);a.output.write_text(json.dumps(r,indent=2),encoding='utf-8');print('SKIRT_TEMPORAL_AUDIT',r)
