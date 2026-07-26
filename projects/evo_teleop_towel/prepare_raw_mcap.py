#!/usr/bin/env python3
from __future__ import annotations
import argparse, io, json, shutil
from pathlib import Path
import h5py, imageio_ffmpeg, numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
import pinocchio as pin

FPS=25; TASK='fold the large blue towel twice'; GRIP=.901637; ACTION_DIM=29
TOPICS=['/joint_states','/left_arm/joint_trajectory','/right_arm/joint_trajectory','/left_arm/gripper/commanded_position','/right_arm/gripper/commanded_position','/elevator/commanded_position','/left_arm/fk_pose','/right_arm/fk_pose','/robot_description','/base/image_raw/compressed','/left_arm/image_raw/compressed','/right_arm/image_raw/compressed','/left_gripper_controller/commands','/right_gripper_controller/commands','/elevator/cmd_vel']
CAMS={'cam_high':'/base/image_raw/compressed','cam_left_wrist':'/left_arm/image_raw/compressed','cam_right_wrist':'/right_arm/image_raw/compressed'}

def nearest(t,x):
 j=np.searchsorted(t,x).clip(0,len(t)-1); i=np.maximum(j-1,0); return np.where(abs(t[i]-x)<=abs(t[j]-x),i,j)
def p7(x):
 p=getattr(x,'pose',x); p=getattr(p,'pose',p)
 return np.array([p.position.x,p.position.y,p.position.z,p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w],float)
def read(path):
 out={x:[] for x in TOPICS}
 with path.open('rb') as f:
  for _,c,m,d in make_reader(f,decoder_factories=[DecoderFactory()]).iter_decoded_messages():
   if c.topic in out: out[c.topic].append((m.log_time*1e-9,d))
 return out
def stream(rows,fn):
 t=np.array([x[0] for x in rows],float); v=[fn(x[1]) for x in rows]; o=np.argsort(t); return t[o],np.asarray(v,object if fn.__name__=='blob' else None)[o]
def blob(x): return bytes(x.data)
def scalar(x): return float(np.asarray(x.data).reshape(-1)[0])
def pose_to_6(p): return np.r_[p[:3],Rotation.from_quat(p[3:7]).as_rotvec()]
def joint_dict(x): return dict(zip(x.name,map(float,x.position)))
def traj(rows):
 ts=[]; vals=[]
 for t,x in rows:
  if x.points: ts.append(t); vals.append(dict(zip(x.joint_names,map(float,x.points[0].positions))))
 o=np.argsort(ts); return np.asarray(ts)[o],np.asarray(vals,object)[o]
def make_pin(urdf):
 model=pin.buildModelFromXML(urdf); data=model.createData()
 for frame in ['left_palm','right_palm','carriage_link']:
  if not model.existFrame(frame): raise RuntimeError(f'URDF missing {frame}')
 return model,data
def fk(model,data,qdict):
 q=pin.neutral(model)
 for name,val in qdict.items():
  if model.existJointName(name):
   j=model.getJointId(name); q[model.joints[j].idx_q]=val
 pin.forwardKinematics(model,data,q); pin.updateFramePlacements(model,data)
 c=data.oMf[model.getFrameId('carriage_link')]
 ans=[]
 for side in ['left','right']:
  m=c.inverse()*data.oMf[model.getFrameId(side+'_palm')]
  ans.append((m.translation.copy(),Rotation.from_matrix(m.rotation)))
 return ans
def write_video(path,rows,grid,size):
 valid=[]; bad=0
 for t,msg in rows:
  b=blob(msg)
  try:
   with Image.open(io.BytesIO(b)) as im: im.verify()
   valid.append((t,b))
  except Exception: bad+=1
 if not rows:
  metrics={'messages':0,'valid':0,'corrupt':0,'max_valid_gap_s':0.,
           'leading_fill_s':0.,'trailing_fill_s':0.,'synthetic_black':True}
  print(f'WARNING {path.name}: source stream absent; writing explicit black placeholder',flush=True)
  w=imageio_ffmpeg.write_frames(str(path),(size,size),fps=FPS,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',output_params=['-crf','23','-movflags','+faststart'],macro_block_size=1); w.send(None)
  try:
   black=np.zeros((size,size,3),np.uint8)
   for _ in grid: w.send(black)
  finally: w.close()
  return metrics
 if not valid: raise RuntimeError(f'camera has messages but no decodable frames: {path.name}')
 t=np.array([x[0] for x in valid]); vals=[x[1] for x in valid]; ix=nearest(t,grid)
 gaps=np.diff(t)
 metrics={'messages':len(rows),'valid':len(valid),'corrupt':bad,
          'max_valid_gap_s':float(gaps.max()) if len(gaps) else 0.,
          'leading_fill_s':float(max(0.,t[0]-grid[0])),
          'trailing_fill_s':float(max(0.,grid[-1]-t[-1]))}
 if bad or metrics['max_valid_gap_s']>.5 or metrics['leading_fill_s']>.5 or metrics['trailing_fill_s']>.5:
  print(f'WARNING {path.name}: nearest-valid resampling metrics={metrics}',flush=True)
 w=imageio_ffmpeg.write_frames(str(path),(size,size),fps=FPS,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',output_params=['-crf','23','-movflags','+faststart'],macro_block_size=1); w.send(None)
 try:
  for i in ix:
   with Image.open(io.BytesIO(vals[int(i)])) as im:
    a=np.asarray(im.convert('RGB').resize((size,size),Image.BICUBIC),np.uint8); w.send(np.ascontiguousarray(a))
 finally: w.close()
 return metrics
def convert(rawpath,out,ep,size):
 r=read(rawpath); required=['/joint_states','/left_arm/joint_trajectory','/right_arm/joint_trajectory','/left_arm/fk_pose','/right_arm/fk_pose','/robot_description',CAMS['cam_high']]
 missing=[x for x in required if not r[x]]
 if missing: raise RuntimeError(f'episode {ep} missing {missing}')
 for high,low in [('/left_arm/gripper/commanded_position','/left_gripper_controller/commands'),('/right_arm/gripper/commanded_position','/right_gripper_controller/commands')]:
  if not r[high] and not r[low]: raise RuntimeError(f'episode {ep} missing both {high} and {low}')
 # Synchronize the complete sample to the ego camera. Wrist streams may start
 # late, end early, or drop frames; nearest-valid resampling preserves the
 # trajectory instead of silently truncating every modality.
 ego=r[CAMS['cam_high']]
 cam_start=ego[0][0]; cam_end=ego[-1][0]
 grid=np.arange(cam_start,cam_end,1/FPS);
 if len(grid)<51: raise RuntimeError(f'episode {ep} too short')
 jt,jv=stream(r['/joint_states'],joint_dict); lt,lv=traj(r['/left_arm/joint_trajectory']); rt,rv=traj(r['/right_arm/joint_trajectory']); lrows=r['/left_arm/gripper/commanded_position'] or r['/left_gripper_controller/commands']; rrows=r['/right_arm/gripper/commanded_position'] or r['/right_gripper_controller/commands']; lgt,lg=stream(lrows,scalar); rgt,rg=stream(rrows,scalar); elevator_fallback=not r['/elevator/commanded_position']; lft,lf=stream(r['/left_arm/fk_pose'],p7); rft,rf=stream(r['/right_arm/fk_pose'],p7)
 if elevator_fallback:
  vt,vv=stream(r['/elevator/cmd_vel'],scalar)
  carriage=np.asarray([x['carriage_joint'] for x in jv])
  if not len(vv) or np.max(np.abs(vv))>1e-8 or np.ptp(carriage)>5e-4: raise RuntimeError(f'episode {ep} cannot infer stationary elevator: vmax={np.max(np.abs(vv)) if len(vv) else None}, range={np.ptp(carriage):.6f}')
  et,ev=jt,carriage
 else: et,ev=stream(r['/elevator/commanded_position'],scalar)
 ji=nearest(jt,grid); li=nearest(lt,grid); ri=nearest(rt,grid); lgi=nearest(lgt,grid); rgi=nearest(rgt,grid); ei=nearest(et,grid); lfi=nearest(lft,grid); rfi=nearest(rft,grid)
 urdf=r['/robot_description'][-1][1].data; model,pdata=make_pin(urdf)
 proprio=np.zeros((len(grid),14),np.float32); action=np.zeros((len(grid),ACTION_DIM),np.float32)
 fk_check=[]
 for n in range(len(grid)):
  cur=jv[ji[n]]; lpose=np.asarray(lf[lfi[n]],float); rpose=np.asarray(rf[rfi[n]],float)
  proprio[n,:6]=pose_to_6(lpose); proprio[n,6]=np.clip(-cur['left_finger_joint1']/GRIP,0,1); proprio[n,7:13]=pose_to_6(rpose); proprio[n,13]=np.clip(-cur['right_finger_joint1']/GRIP,0,1)
  for k in range(1,8): action[n,k-1]=lv[li[n]][f'left_joint{k}']-cur[f'left_joint{k}']; action[n,7+k]=rv[ri[n]][f'right_joint{k}']-cur[f'right_joint{k}']
  action[n,7]=np.clip(-lg[lgi[n]]/GRIP,0,1)-proprio[n,6]; action[n,15]=np.clip(-rg[rgi[n]]/GRIP,0,1)-proprio[n,13]; action[n,16]=ev[ei[n]]-cur['carriage_joint']
  target=dict(cur); target.update(lv[li[n]]); target.update(rv[ri[n]]); target['carriage_joint']=float(ev[ei[n]])
  target_fk=fk(model,pdata,target)
  for sl,obs,tgt in [(slice(17,23),lpose,target_fk[0]),(slice(23,29),rpose,target_fk[1])]:
   action[n,sl.start:sl.start+3]=tgt[0]-obs[:3]; action[n,sl.start+3:sl.stop]=(Rotation.from_quat(obs[3:7]).inv()*tgt[1]).as_rotvec()
  if n%100==0:
   measured_fk=fk(model,pdata,cur)
   fk_check.extend([np.linalg.norm(measured_fk[0][0]-lpose[:3]),np.linalg.norm(measured_fk[1][0]-rpose[:3])])
 if not np.isfinite(action).all() or not np.isfinite(proprio).all(): raise RuntimeError(f'episode {ep} nonfinite')
 med=float(np.median(fk_check)); p95=float(np.percentile(fk_check,95))
 if med>.005 or p95>.015: raise RuntimeError(f'episode {ep} FK check median={med:.4f} p95={p95:.4f}')
 stem=f'episode_{ep:03d}'; vids={}; camera_metrics={}
 for name,topic in CAMS.items():
  vids[name]=f'{stem}_{name}.mp4'
  camera_metrics[name]=write_video(out/vids[name],r[topic],grid,size)
 with h5py.File(out/f'{stem}.hdf5','w') as h:
  h.attrs.update(sim=False,success=True,task_description=TASK,fps=FPS,source_mcap=rawpath.name,action_layout='left_joint_delta7,left_gripper_delta,right_joint_delta7,right_gripper_delta,elevator_delta,left_ee_delta_xyz_rotvec6,right_ee_delta_xyz_rotvec6')
  h.attrs['camera_resampling_metrics']=json.dumps(camera_metrics)
  o=h.create_group('observations'); o.create_dataset('qpos',data=proprio); o.create_dataset('qvel',data=np.gradient(proprio,1/FPS,axis=0)); o.create_dataset('effort',data=np.zeros_like(proprio)); vp=o.create_group('video_paths')
  for k,v in vids.items(): vp.create_dataset(k,data=v.encode())
  h.create_dataset('action',data=action); h.create_dataset('relative_action',data=action); h.create_dataset('action_dim_mask',data=np.ones(ACTION_DIM,np.float32))
 return {'episode':ep,'frames':len(grid),'duration_s':float(grid[-1]-grid[0]),'fk_median_m':med,'fk_p95_m':p95,'camera_resampling':camera_metrics,'action_abs_p99':np.percentile(abs(action),99,axis=0).tolist()}
def main():
 a=argparse.ArgumentParser(); a.add_argument('--raw-root',type=Path,required=True); a.add_argument('--mapping',type=Path,required=True); a.add_argument('--out',type=Path,required=True); a.add_argument('--image-size',type=int,default=256); a.add_argument('--overwrite',action='store_true'); a.add_argument('--resume',action='store_true'); z=a.parse_args()
 if z.out.exists():
  if z.overwrite: shutil.rmtree(z.out)
  elif not z.resume: raise FileExistsError(z.out)
 train=z.out/'train'; train.mkdir(parents=True,exist_ok=True); result=[]
 for m in json.loads(z.mapping.read_text()):
  ep=int(m['episode']); existing=train/f'episode_{ep:03d}.hdf5'
  videos=[train/f'episode_{ep:03d}_{c}.mp4' for c in CAMS]
  if z.resume and existing.is_file() and all(x.is_file() for x in videos):
   with h5py.File(existing,'r') as h: x=h['action'][:]; frames=len(x)
   result.append({'episode':ep,'frames':frames,'duration_s':frames/FPS,'fk_median_m':None,'fk_p95_m':None,'action_abs_p99':np.percentile(abs(x),99,axis=0).tolist(),'resumed_prevalidated':True}); print(f'[{ep+1}/50] retained validated episode',flush=True); continue
  raw=z.raw_root/f'episode-{ep:03d}'/f"{m['name']}_0.mcap"; print(f'[{ep+1}/50] {raw.name}',flush=True); result.append(convert(raw,train,ep,z.image_size))
 summary={'task':TASK,'fps':FPS,'action_dim':ACTION_DIM,'episodes':result,'total_frames':sum(x['frames'] for x in result)}; (z.out/'conversion_manifest.json').write_text(json.dumps(summary,indent=2)); print(json.dumps({'episodes':len(result),'total_frames':summary['total_frames']},indent=2))
if __name__=='__main__': main()
