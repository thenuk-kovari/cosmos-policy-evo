#!/usr/bin/env python3
"""Add base_link-to-palm FK, relabelled to UMI axes, to Evo episodes.

The URDF is the sole source of the FK pose and the measured carriage_joint is
part of that chain.  After FK, position is left untouched while orientation is
right-multiplied by the explicitly specified UMI-axis relabelling quaternion.
"""
import argparse, json, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
from cosmos_policy.datasets.ee_q0_actions import (
    SHARED_ACTION_DIM, LEFT_EE_TRANSLATION, LEFT_EE_ROTATION_6D,
    RIGHT_EE_TRANSLATION, RIGHT_EE_ROTATION_6D, LEFT_JOINT_DELTA,
    LEFT_GRIPPER, RIGHT_JOINT_DELTA, RIGHT_GRIPPER, ELEVATOR_DELTA,
    CONTRACT_NAME, EVO_STORAGE_CONTRACT, NORMALIZATION_CONTRACT,
    EVO_TO_UMI_ORIENTATION_OFFSET_WXYZ, canonical_shared_statistics,
    matrix_to_rotation_6d, relabel_evo_orientation_to_umi,
)
from projects.evo_q0_towel.newoffice73_split import (
    VALIDATION_OUTPUT_INDICES,
    load_original_split,
    split_summary,
)

OPEN_RAD = -0.90163709

def tf(rotation=np.eye(3), translation=np.zeros(3)):
    out=np.eye(4); out[:3,:3]=rotation; out[:3,3]=translation; return out

class FK:
    def __init__(self, path):
        self.joints={}
        for node in ET.parse(path).getroot().findall("joint"):
            origin=node.find("origin")
            xyz=np.fromstring(origin.get("xyz","0 0 0"),sep=" ") if origin is not None else np.zeros(3)
            rpy=np.fromstring(origin.get("rpy","0 0 0"),sep=" ") if origin is not None else np.zeros(3)
            axis=node.find("axis")
            axis=np.fromstring(axis.get("xyz","1 0 0"),sep=" ") if axis is not None else np.array([1.,0.,0.])
            child=node.find("child").get("link")
            self.joints[child]=(node.get("name"),node.find("parent").get("link"),
                node.get("type"),tf(Rotation.from_euler("xyz",rpy).as_matrix(),xyz),axis)

    def pose(self, root, tip, q):
        chain=[]; link=tip
        while link != root:
            if link not in self.joints: raise ValueError(f"{tip} not descended from {root}")
            joint=self.joints[link]; chain.append(joint); link=joint[1]
        out=np.eye(4)
        for name,parent,kind,origin,axis in chain[::-1]:
            out=out@origin
            if kind in ("revolute","continuous"):
                out=out@tf(Rotation.from_rotvec(axis*q[name]).as_matrix())
            elif kind=="prismatic": out=out@tf(translation=axis*q[name])
            elif kind!="fixed": raise ValueError(f"unsupported joint {kind}")
        return out

def convert_states(states, fk):
    states=np.asarray(states,dtype=np.float64)
    if states.ndim!=2 or states.shape[1]!=17 or not np.isfinite(states).all():
        raise ValueError(f"bad measured states {states.shape}")
    source=np.zeros((len(states),SHARED_ACTION_DIM),np.float32)
    proprio=states.astype(np.float32)
    proprio[:,7]=np.clip(states[:,7]/OPEN_RAD,0,1)
    proprio[:,15]=np.clip(states[:,15]/OPEN_RAD,0,1)
    for i,state in enumerate(states):
        q={**{f"left_joint{j+1}":state[j] for j in range(7)},
           **{f"right_joint{j+1}":state[8+j] for j in range(7)},
           "carriage_joint":state[16]}
        for side,ps,rs in (("left",LEFT_EE_TRANSLATION,LEFT_EE_ROTATION_6D),
                           ("right",RIGHT_EE_TRANSLATION,RIGHT_EE_ROTATION_6D)):
            # Vanilla base-frame URDF FK, followed by the requested child-axis
            # relabelling. Position is deliberately not transformed.
            pose=fk.pose("base_link",f"{side}_palm",q)
            source[i,ps]=pose[:3,3]
            relabelled_rotation=relabel_evo_orientation_to_umi(pose[:3,:3])
            source[i,rs]=matrix_to_rotation_6d(relabelled_rotation)
    source[:,LEFT_JOINT_DELTA]=states[:,:7]
    source[:,LEFT_GRIPPER]=proprio[:,7]
    source[:,RIGHT_JOINT_DELTA]=states[:,8:15]
    source[:,RIGHT_GRIPPER]=proprio[:,15]
    source[:,ELEVATOR_DELTA]=states[:,16]
    return source,proprio

def selected(selection, roots):
    for output,row in enumerate(json.loads(Path(selection).read_text())):
        name=row["dataset"]; index=int(row["file_index"])
        yield output,name,index,Path(roots[name])/f"data/chunk-000/file-{index:03d}.parquet"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",action="append",required=True,help="NAME=/root")
    ap.add_argument("--selection",required=True)
    ap.add_argument("--urdf",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--val-count",type=int,default=6)
    ap.add_argument("--expected-episodes",type=int,default=73)
    args=ap.parse_args()
    roots=dict(x.split("=",1) for x in args.dataset)
    split_rows=load_original_split(args.selection)
    if args.val_count!=6 or args.expected_episodes!=73:
        raise ValueError(
            "the original Evo-only ablation split requires --val-count 6 and --expected-episodes 73"
        )
    rows=list(selected(args.selection,roots))
    identities=[(name,index) for _,name,index,_ in rows]
    expected=[(str(row["dataset"]),int(row["file_index"])) for row in split_rows]
    if identities!=expected:
        raise RuntimeError("resolved episode ordering differs from the original Evo-only split contract")
    out=Path(args.out)
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    val=set(VALIDATION_OUTPUT_INDICES)
    fk=FK(args.urdf); manifest=[]
    for n,(output,name,index,path) in enumerate(rows,1):
        print(f"[{n}/{len(rows)}] {name}/file-{index:03d}",flush=True)
        table=pq.read_table(path)
        states=np.asarray(table["observation.state"].to_pylist(),np.float64)
        if len(states)<51: raise ValueError(f"{path}: only {len(states)} frames")
        source,proprio=convert_states(states,fk)
        action=pa.FixedSizeListArray.from_arrays(pa.array(source.ravel()),SHARED_ACTION_DIM)
        observation=pa.FixedSizeListArray.from_arrays(pa.array(proprio.ravel()),17)
        ai=table.schema.get_field_index("action")
        oi=table.schema.get_field_index("observation.state")
        table=table.set_column(ai,pa.field("action",action.type),action)
        table=table.set_column(oi,pa.field("observation.state",observation.type),observation)
        metadata=dict(table.schema.metadata or {})
        metadata.update({
            b"action_contract":CONTRACT_NAME.encode(),
            b"action_storage":EVO_STORAGE_CONTRACT.encode(),
            b"normalization_contract":NORMALIZATION_CONTRACT.encode(),
            b"fk_root":b"base_link",b"fk_tips":b"left_palm,right_palm",
            b"source_pose_link":b"base_link",
            b"position_post_transform":b"none",
            b"orientation_post_transform":b"right_multiply_q_offset_wxyz",
            b"orientation_offset_wxyz":b"0,0.7071067811865476,0,0.7071067811865476",
            b"recorded_action_column_used":b"false",
            b"task":b"fold the blue towel twice",
        })
        table=table.replace_schema_metadata(metadata)
        split="val" if output in val else "train"
        destination=out/split/f"file-{output:03d}.parquet"
        destination.parent.mkdir(exist_ok=True)
        pq.write_table(table,destination,compression="zstd")
        manifest.append({"output_index":output,"split":split,"source_dataset":name,
                         "source_file_index":index,"frames":len(states)})
    statistics={k:v.tolist() for k,v in canonical_shared_statistics().items()}
    (out/"dataset_statistics.json").write_text(json.dumps(statistics,indent=2)+"\n")
    payload={"action_contract":CONTRACT_NAME,"storage_contract":EVO_STORAGE_CONTRACT,
        "normalization_contract":NORMALIZATION_CONTRACT,"action_dim":35,"proprio_dim":17,
        "chunk_size":50,"fps":30,"task":"fold the blue towel twice",
        "fk":{"root":"base_link","tips":["left_palm","right_palm"],
              "position_post_transform":"none",
              "orientation_post_transform":"right_multiply_q_offset_wxyz",
              "orientation_offset_wxyz":EVO_TO_UMI_ORIENTATION_OFFSET_WXYZ.tolist(),
              "includes_joint":"carriage_joint"},
        "recorded_action_column_used":False,
        "counts":{"total":len(rows),"train":len(rows)-len(val),"val":len(val)},
        "validation_output_indices":[int(value) for value in sorted(val)],
        "ablation_split":split_summary(split_rows),"episodes":manifest}
    (out/"conversion_manifest.json").write_text(json.dumps(payload,indent=2)+"\n")

if __name__=="__main__": main()
