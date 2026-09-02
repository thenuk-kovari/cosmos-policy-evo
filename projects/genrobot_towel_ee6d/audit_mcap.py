#!/usr/bin/env python3
"""Audit GenRobot dual-hand MCAPs before conversion.

This reads only metadata and numeric streams.  It intentionally does not
decode H.264 video, so hundreds of files can be checked cheaply after they are
downloaded.  Exit status is non-zero when any episode fails a hard gate.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

from cosmos_policy.datasets.ee_q0_actions import quaternion_xyzw_to_matrix


REQUIRED_TOPICS = tuple(
    f"/robot{robot}/{suffix}"
    for robot in (0, 1)
    for suffix in (
        "sensor/camera0/compressed",
        "sensor/camera0/camera_info",
        "sensor/magnetic_encoder",
        "vio/eef_pose",
    )
)


# CameraInfo is a latched, one-message calibration stream. It must exist but
# must not participate in trajectory-overlap calculations.
TIMELINE_TOPICS = tuple(topic for topic in REQUIRED_TOPICS if not topic.endswith("/camera_info"))

def _timestamp_ns(decoded, fallback: int) -> int:
    header = getattr(decoded, "header", None)
    value = int(getattr(header, "timestamp", 0)) if header is not None else 0
    return value or int(fallback)


def _frequency(timestamps_ns: np.ndarray) -> dict:
    if len(timestamps_ns) < 2:
        return {"count": int(len(timestamps_ns)), "hz": 0.0, "max_gap_s": None, "non_monotonic": 0}
    delta = np.diff(timestamps_ns).astype(np.float64) / 1e9
    positive = delta[delta > 0]
    hz = 1.0 / np.median(positive) if len(positive) else 0.0
    return {
        "count": int(len(timestamps_ns)),
        "hz": float(hz),
        "max_gap_s": float(delta.max()),
        "non_monotonic": int(np.count_nonzero(delta <= 0)),
        "large_gaps": int(np.count_nonzero(delta > max(0.2, 3.0 / hz))) if hz else 0,
    }


def inspect(path: Path) -> dict:
    timestamps: dict[str, list[int]] = defaultdict(list)
    poses: dict[int, list[list[float]]] = defaultdict(list)
    grippers: dict[int, list[float]] = defaultdict(list)
    camera: dict[int, dict] = {}
    formats: dict[int, set[str]] = defaultdict(set)
    schemas: dict[str, str] = {}
    pose_frames: dict[int, set[str]] = defaultdict(set)

    with path.open("rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        for schema, channel, message, decoded in reader.iter_decoded_messages():
            topic = channel.topic
            schemas[topic] = schema.name
            timestamps[topic].append(_timestamp_ns(decoded, message.log_time))
            for robot in (0, 1):
                if topic == f"/robot{robot}/vio/eef_pose":
                    p, q = decoded.pose.position, decoded.pose.orientation
                    poses[robot].append([p.x, p.y, p.z, q.x, q.y, q.z, q.w])
                    pose_frames[robot].add(str(decoded.frame_id))
                elif topic == f"/robot{robot}/sensor/magnetic_encoder":
                    grippers[robot].append(float(decoded.value))
                elif topic == f"/robot{robot}/sensor/camera0/compressed":
                    formats[robot].add(decoded.format)
                elif topic == f"/robot{robot}/sensor/camera0/camera_info":
                    camera[robot] = {
                        "width": int(decoded.width),
                        "height": int(decoded.height),
                        "distortion_model": decoded.distortion_model,
                        "frame_id": decoded.frame_id,
                        "T_b_c": list(decoded.T_b_c),
                    }

    result = {
        "path": str(path),
        "topics": {topic: {"schema": schemas[topic], **_frequency(np.asarray(ts))} for topic, ts in timestamps.items()},
        "missing_required_topics": sorted(set(REQUIRED_TOPICS) - set(timestamps)),
        "camera": camera,
        "checks": [],
    }
    timeline_present = [topic for topic in TIMELINE_TOPICS if timestamps.get(topic)]
    if len(timeline_present) == len(TIMELINE_TOPICS):
        overlap_start = max(min(timestamps[x]) for x in timeline_present)
        overlap_end = min(max(timestamps[x]) for x in timeline_present)
        result["common_overlap_s"] = max(0.0, (overlap_end - overlap_start) / 1e9)
    else:
        result["common_overlap_s"] = 0.0

    for robot in (0, 1):
        pose = np.asarray(poses[robot], dtype=np.float64)
        pose_ts = np.asarray(timestamps.get(f"/robot{robot}/vio/eef_pose", []), dtype=np.int64)
        grip = np.asarray(grippers[robot], dtype=np.float64)
        if len(pose):
            qnorm = np.linalg.norm(pose[:, 3:7], axis=1)
            rotation = quaternion_xyzw_to_matrix(pose[:, 3:7])
            relative = np.einsum("tji,tjk->tik", rotation[:-1], rotation[1:])
            cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
            angular_step = np.arccos(cosine)
            translation_step = np.linalg.norm(np.diff(pose[:, :3], axis=0), axis=1)
            dt = np.diff(pose_ts).astype(np.float64) / 1e9
            valid_dt = np.maximum(dt, 1e-9)
            result[f"robot{robot}_pose"] = {
                "frame_ids": sorted(pose_frames[robot]),
                "position_min": pose[:, :3].min(axis=0).tolist(),
                "position_max": pose[:, :3].max(axis=0).tolist(),
                "quaternion_norm_min": float(qnorm.min()),
                "quaternion_norm_max": float(qnorm.max()),
                "max_translation_step_m": float(translation_step.max(initial=0)),
                "max_translation_speed_m_s": float((translation_step / valid_dt).max(initial=0)),
                "max_rotation_step_deg": float(np.degrees(angular_step.max(initial=0))),
                "max_rotation_speed_deg_s": float(np.degrees((angular_step / valid_dt).max(initial=0))),
                "translation_steps_over_10cm": int(np.count_nonzero(translation_step > 0.10)),
                "rotation_steps_over_45deg": int(np.count_nonzero(angular_step > np.pi / 4)),
            }
        if len(grip):
            result[f"robot{robot}_gripper"] = {
                "minimum_m": float(grip.min()),
                "maximum_m": float(grip.max()),
                "outside_documented_range": int(np.count_nonzero((grip < -0.002) | (grip > 0.105))),
                "steps_over_3cm": int(np.count_nonzero(np.abs(np.diff(grip)) > 0.03)),
            }
        result[f"robot{robot}_camera_formats"] = sorted(formats[robot])

    if result["missing_required_topics"]:
        result["checks"].append("missing_required_topic")
    if result["common_overlap_s"] < 2.0:
        result["checks"].append("insufficient_common_overlap")
    for topic in REQUIRED_TOPICS:
        if topic in result["topics"] and result["topics"][topic]["non_monotonic"]:
            result["checks"].append(f"non_monotonic:{topic}")
    for robot in (0, 1):
        pose_result = result.get(f"robot{robot}_pose", {})
        gripper_result = result.get(f"robot{robot}_gripper", {})
        if pose_result.get("translation_steps_over_10cm", 0):
            result["checks"].append(f"pose_translation_jump:robot{robot}")
        if pose_result.get("rotation_steps_over_45deg", 0):
            result["checks"].append(f"pose_rotation_jump:robot{robot}")
        if gripper_result.get("outside_documented_range", 0):
            result["checks"].append(f"gripper_out_of_range:robot{robot}")
        if formats[robot] != {"h264"}:
            result["checks"].append(f"camera_format:robot{robot}")
    result["accepted"] = not result["checks"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mcaps", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reports = [inspect(path) for path in args.mcaps]
    payload = {
        "files": len(reports),
        "accepted": sum(x["accepted"] for x in reports),
        "rejected": sum(not x["accepted"] for x in reports),
        "reports": reports,
    }
    text = json.dumps(payload, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    raise SystemExit(0 if payload["rejected"] == 0 else 2)


if __name__ == "__main__":
    main()
