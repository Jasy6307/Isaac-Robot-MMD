"""Recompute common reward terms from fixed inference trajectories."""
import json
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'output/anon_locomotion/low_speed_diagnosis_20261117'
sys.path[:0]=[str(ROOT/'source/pmx_prototype'),str(ROOT/'source/pmx_prototype/training')]
from amp_motion_prior import features


def main():
    torch.set_num_threads(2)
    meta=json.loads((OUT/'matched/metadata.json').read_text());cfg=meta['config']
    source=np.load(cfg['full_motion_path']);ref_links=list(source['link_names'])
    names=meta['dof_names'];links=meta['link_names'];legs=meta['leg_ids'];upper=meta['upper_ids'];dt=meta['dt']
    keys=['waist_pitch','head','left_knee_pitch','right_knee_pitch','left_ankle_roll','right_ankle_roll','left_hand','right_hand']
    actual_ids=[links.index(n) for n in keys];reference_ids=[ref_links.index(n) for n in keys]
    pelvis=links.index('pelvis');ref_pelvis=ref_links.index('pelvis')
    jw=np.ones(len(names));jw[upper]=cfg.get('amp_upper_pose_weight',.2)
    kw=np.ones(len(keys));kw[[0,1,6,7]]=cfg.get('amp_upper_keypoint_weight',.25)
    rows={}
    for model in ['old','new','nominal']:
        path=OUT/'matched'/f'{model}_030.npz'
        if not path.exists():continue
        z=np.load(path);a,b=round(8/dt),round(14/dt)
        valid_env=~z['done'][:b].any(0)
        if not valid_env.any():
            rows[model]=dict(valid_for_steady_comparison=False,falls=int(z['done'].sum()),
                reason='Repeated reference-state resets invalidate steady speed/reward comparisons.')
            continue
        e=((z['root_velocity'][:,:,:3]-z['reference_velocity'][:,:,:3])**2).sum(-1)
        delta=z['root_velocity'][:,:,:3]-z['reference_velocity'][:,:,:3]
        heading=z['headings'];sine=np.sin(heading);cosine=np.cos(heading)
        heading_error=np.stack([sine*delta[:,:,0]-cosine*delta[:,:,1],
            cosine*delta[:,:,0]+sine*delta[:,:,1],delta[:,:,2]],-1)
        speed=np.linalg.norm(z['ref_commands'][:,:,:2],axis=-1)
        scale=cfg['continuous_tracking_error_base']+cfg['continuous_tracking_error_speed_scale']*speed**2
        pose=(((z['q']-z['reference_q'])**2)*jw).sum(-1)/jw.sum()
        qd=(((z['qd']-z['reference_qd'])**2)*jw).sum(-1)/jw.sum()
        actual=z['poses'][:,:,actual_ids,:3]-z['poses'][:,:,pelvis,None,:3]
        reference=z['reference_positions'][:,:,reference_ids]-z['reference_positions'][:,:,ref_pelvis,None]
        key=(((actual-reference)**2).sum(-1)*kw).sum(-1)/kw.sum()
        accel=np.diff(z['q'][:,:,legs],n=2,axis=0)/dt**2
        smooth=(np.sqrt(1+(accel/30)**2)-1).mean(-1)
        terms=dict(continuous_velocity=cfg['continuous_tracking_weight']*np.exp(-e/scale),
            imitation_velocity=cfg['imitation_velocity_weight']*np.exp(-e/cfg['imitation_velocity_error_scale']),
            joint_pose=3*np.exp(-pose/.065),joint_velocity=.7*np.exp(-qd/12),
            keypoints=cfg['imitation_keypoint_weight']*np.exp(-key/.008),
            action_cost=-.005*(z['actions']**2).mean(-1),
            action_change_cost=-.003*(np.diff(z['actions'],axis=0)**2).sum(-1))
        mean_terms={name:float(value[a:b,valid_env].mean()) for name,value in terms.items()}
        mean_terms['leg_acceleration_cost']=-cfg['leg_acceleration_weight']*float(smooth[a-2:b-2,valid_env].mean())
        joints={}
        for name in ['left_hip_pitch_joint','right_hip_pitch_joint','left_knee_pitch_joint','right_knee_pitch_joint']:
            k=names.index(name)
            joints[name]=dict(reference_mean=float(z['reference_q'][a:b,:,k].mean()),actual_mean=float(z['q'][a:b,:,k].mean()),
                residual_mean_rad=float((z['target_q'][a:b,:,k]-z['reference_q'][a:b,:,k]).mean()),
                reference_rms_variation_rad=float(z['reference_q'][a:b,:,k].std(0).mean()),
                target_rms_variation_rad=float(z['target_q'][a:b,:,k].std(0).mean()),
                actual_rms_variation_rad=float(z['q'][a:b,:,k].std(0).mean()))
        rows[model]=dict(valid_for_steady_comparison=True,valid_envs=int(valid_env.sum()),
            walk_falls=int(z['done'][round(2/dt):b].sum()),stop_falls=int(z['done'][b:].sum()),
            mean_velocity_reward_target_error_rms=float(np.sqrt(e[a:b,valid_env].mean())),
            heading_frame_velocity_error_mean=heading_error[a:b,valid_env].mean((0,1)).tolist(),
            heading_frame_velocity_error_mse=(heading_error[a:b,valid_env]**2).mean((0,1)).tolist(),
            mean_reward_terms_per_second=mean_terms,leg_reference_rmse_rad=float(np.sqrt(((z['q'][a:b,:,legs]-z['reference_q'][a:b,:,legs])**2).mean())),
            leg_pd_target_rmse_rad=float(np.sqrt(((z['q'][a:b,:,legs]-z['target_q'][a:b,:,legs])**2).mean())),joints=joints)
    curve=[]
    for error in [0,.03,.06,.066,.098,.1,.15]:
        imitation=cfg['imitation_velocity_weight']*np.exp(-error**2/cfg['imitation_velocity_error_scale'])
        tracking=cfg['continuous_tracking_weight']*np.exp(-error**2/(cfg['continuous_tracking_error_base']+cfg['continuous_tracking_error_speed_scale']*.3**2))
        curve.append(dict(horizontal_error_m_s=error,imitation_velocity=float(imitation),continuous_tracking=float(tracking),
            combined=float(imitation+tracking),fraction_of_zero_error=float((imitation+tracking)/(cfg['imitation_velocity_weight']+cfg['continuous_tracking_weight']))))
    windows={};models=json.loads((OUT/'models.json').read_text())
    for name in ['old','new']:
        z=np.load(OUT/'matched'/f'{name}_030.npz');t,n=z['q'].shape[:2]
        def tensor(x):return torch.tensor(x).flatten(0,1)
        f=features(tensor(z['q']),tensor(z['qd']),tensor(z['poses'][:,:,pelvis,:3]),
            tensor(z['poses'][:,:,pelvis,3:]),tensor(z['root_velocity']),tensor(z['poses'][:,:,actual_ids,:3])).reshape(t,n,-1)
        windows[name]=torch.stack([f[i:t-3+i] for i in range(4)],2)[a-3:b-3]
    cross={}
    with torch.inference_mode():
        for discriminator_name,path in models.items():
            state=torch.load(path,map_location='cpu',weights_only=False)['amp_state'];size=len(state['mean'])
            discriminator=torch.nn.Sequential(torch.nn.Linear(size*4,256),torch.nn.LeakyReLU(.2),
                torch.nn.Linear(256,128),torch.nn.LeakyReLU(.2),torch.nn.Linear(128,1))
            discriminator.load_state_dict(state['discriminator']);discriminator.eval()
            cross[discriminator_name]={}
            for trajectory,window in windows.items():
                normalized=(((window.reshape(-1,4,size)-state['mean'])/state['scale'])*state['feature_weights']).clamp(-8,8).flatten(1)
                score=discriminator(normalized).squeeze(-1);style=(1-.25*(score-1)**2).clamp(0,1)
                cross[discriminator_name][trajectory]=dict(score=float(score.mean()),style_reward=float(style.mean()))
    result=dict(training=False,reward_targets='Instantaneous3D root COM velocity from the smoothed whole-body reference, not raw command or mean horizontal error.',
        core_terms=rows,velocity_reward_curve_at_command_030=curve,
        amp_cross_frozen_discriminators=cross,
        lower_pose_bonus_applies_at_030=False,lower_pose_bonus_speed_range=[.02,.16],
        limitations='Reported terms are a common subset, not total reward. AMP scores under different learned discriminators are not directly comparable. No reward settings changed.')
    (OUT/'reward_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
