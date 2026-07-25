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
TOPICS=['/joint_states','/left_arm/joint_trajectory','/right_arm/joint_trajectory','/left_arm/gripper/commanded_position','/right_arm/gripper/commanded_position','/elevator/commanded_position','/left_arm/fk_pose','/right_arm/fk_pose','/robot_description','/base/image_raw/compressed','/left_arm/image_raw/compressed','/right_arm/image_raw/compressed']
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
def scalar(x): return float(x.data)
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
 t=np.array([x[0] for x in rows]); vals=[blob(x[1]) for x in rows]; ix=nearest(t,grid)
 w=imageio_ffmpeg.write_frames(str(path),(size,size),fps=FPS,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',output_params=['-crf','23','-movflags','+faststart'],macro_block_size=1); w.send(None)
 try:
  for i in ix:
   with Image.open(io.BytesIO(vals[int(i)])) as im:
    a=np.asarray(im.convert('RGB').resize((size,size),Image.BICUBIC),np.uint8); w.send(np.ascontiguousarray(a))
 finally: w.close()
def convert(rawpath,out,ep,size):
 r=read(rawpath); missing=[x for x in TOPICS if not r[x]]
 if missing: raise RuntimeError(f'episode {ep} missing {missing}')
 cam_start=max(r[x][0][0] for x in CAMS.values()); cam_end=min(r[x][-1][0] for x in CAMS.values())
 grid=np.arange(cam_start,cam_end,1/FPS);
 if len(grid)<51: raise RuntimeError(f'episode {ep} too short')
 jt,jv=stream(r['/joint_states'],joint_dict); lt,lv=traj(r['/left_arm/joint_trajectory']); rt,rv=traj(r['/right_arm/joint_trajectory']); lgt,lg=stream(r['/left_arm/gripper/commanded_position'],scalar); rgt,rg=stream(r['/right_arm/gripper/commanded_position'],scalar); et,ev=stream(r['/elevator/commanded_position'],scalar); lft,lf=stream(r['/left_arm/fk_pose'],p7); rft,rf=stream(r['/right_arm/fk_pose'],p7)
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
 stem=f'episode_{ep:03d}'; vids={}
 for name,topic in CAMS.items(): vids[name]=f'{stem}_{name}.mp4'; write_video(out/vids[name],r[topic],grid,size)
 with h5py.File(out/f'{stem}.hdf5','w') as h:
  h.attrs.update(sim=False,success=True,task_description=TASK,fps=FPS,source_mcap=rawpath.name,action_layout='left_joint_delta7,left_gripper_delta,right_joint_delta7,right_gripper_delta,elevator_delta,left_ee_delta_xyz_rotvec6,right_ee_delta_xyz_rotvec6')
  o=h.create_group('observations'); o.create_dataset('qpos',data=proprio); o.create_dataset('qvel',data=np.gradient(proprio,1/FPS,axis=0)); o.create_dataset('effort',data=np.zeros_like(proprio)); vp=o.create_group('video_paths')
  for k,v in vids.items(): vp.create_dataset(k,data=v.encode())
  h.create_dataset('action',data=action); h.create_dataset('relative_action',data=action); h.create_dataset('action_dim_mask',data=np.ones(ACTION_DIM,np.float32))
 return {'episode':ep,'frames':len(grid),'duration_s':float(grid[-1]-grid[0]),'fk_median_m':med,'fk_p95_m':p95,'action_abs_p99':np.percentile(abs(action),99,axis=0).tolist()}
def main():
 a=argparse.ArgumentParser(); a.add_argument('--raw-root',type=Path,required=True); a.add_argument('--mapping',type=Path,required=True); a.add_argument('--out',type=Path,required=True); a.add_argument('--image-size',type=int,default=256); a.add_argument('--overwrite',action='store_true'); z=a.parse_args()
 if z.out.exists():
  if not z.overwrite: raise FileExistsError(z.out)
  shutil.rmtree(z.out)
 train=z.out/'train'; train.mkdir(parents=True); result=[]
 for m in json.loads(z.mapping.read_text()):
  ep=int(m['episode']); raw=z.raw_root/f'episode-{ep:03d}'/f"{m['name']}_0.mcap"; print(f'[{ep+1}/50] {raw.name}',flush=True); result.append(convert(raw,train,ep,z.image_size))
 summary={'task':TASK,'fps':FPS,'action_dim':ACTION_DIM,'episodes':result,'total_frames':sum(x['frames'] for x in result)}; (z.out/'conversion_manifest.json').write_text(json.dumps(summary,indent=2)); print(json.dumps({'episodes':len(result),'total_frames':summary['total_frames']},indent=2))
if __name__=='__main__': main()
