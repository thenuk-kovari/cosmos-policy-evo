#!/usr/bin/env python3
import argparse,json,pickle,shutil
from pathlib import Path
import h5py,numpy as np
TASK='fold the large blue towel twice'
def summarize(x): return {'min':x.min(0),'max':x.max(0),'mean':x.mean(0),'std':x.std(0),'median':np.median(x,0)}
def main():
 a=argparse.ArgumentParser(); a.add_argument('root',type=Path); a.add_argument('--embedding',type=Path,required=True); z=a.parse_args(); files=sorted((z.root/'train').glob('episode_*.hdf5'))
 if len(files)!=50: raise RuntimeError(f'expected 50 episodes, found {len(files)}')
 aa=[]; pp=[]; frames=0
 for f in files:
  with h5py.File(f,'r') as h:
   x=h['action'][:]; p=h['observations/qpos'][:]; m=h['action_dim_mask'][:]
   if x.ndim!=2 or x.shape[1]!=29 or p.shape!=(len(x),14) or m.shape!=(29,) or not np.all(m==1): raise RuntimeError(f'bad tensors {f}: {x.shape}/{p.shape}/{m.shape}')
   if not np.isfinite(x).all() or not np.isfinite(p).all(): raise RuntimeError(f'nonfinite {f}')
   if h.attrs['task_description']!=TASK: raise RuntimeError(f'bad task {f}')
   aa.append(x); pp.append(p); frames+=len(x)
 A=np.concatenate(aa); P=np.concatenate(pp); stats={}
 for name,x in [('actions',A),('proprio',P)]:
  s=summarize(x)
  for k,v in s.items(): stats[f'{name}_{k}']=v.tolist()
 (z.root/'dataset_statistics.json').write_text(json.dumps(stats,indent=2))
 post={}
 for name,x in [('actions',A),('proprio',P)]:
  lo=np.asarray(stats[f'{name}_min']); hi=np.asarray(stats[f'{name}_max']); y=2*(x-lo)/np.maximum(hi-lo,1e-8)-1; s=summarize(y)
  for k,v in s.items(): post[f'{name}_{k}']=v.tolist()
 (z.root/'dataset_statistics_post_norm.json').write_text(json.dumps(post,indent=2)); shutil.copy2(z.embedding,z.root/'t5_embeddings.pkl')
 with (z.root/'t5_embeddings.pkl').open('rb') as f: e=pickle.load(f)
 if TASK not in e or tuple(e[TASK].shape)!=(1,512,1024): raise RuntimeError('bad task embedding')
 manifest=json.loads((z.root/'conversion_manifest.json').read_text())
 if len(manifest['episodes'])!=50 or manifest['total_frames']!=frames: raise RuntimeError('manifest mismatch')
 fk_values=[x['fk_p95_m'] for x in manifest['episodes'] if x['fk_p95_m'] is not None]; max_fk=max(fk_values) if fk_values else None; print(json.dumps({'episodes':50,'frames':frames,'hours':frames/25/3600,'max_episode_fk_p95_m':max_fk},indent=2))
if __name__=='__main__': main()
