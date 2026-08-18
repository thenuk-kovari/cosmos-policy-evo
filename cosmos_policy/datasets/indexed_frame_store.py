"""Random-access JPEG sidecars for ALOHA-style MP4 datasets.

Each HDF5 episode keeps its existing MP4 references.  A sidecar consists of a
single concatenated byte file and a small NumPy index containing one ``T+1``
offset array per camera.  Reading a frame is one ``pread`` plus one independent
JPEG decode; it never seeks through an inter-frame video GOP.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np


FORMAT_VERSION = 1
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
DATA_SUFFIX = ".frames.bin"
INDEX_SUFFIX = ".frames.npz"


def sidecar_paths(hdf5_path: str | Path) -> tuple[Path, Path]:
    path = Path(hdf5_path)
    return path.with_suffix(DATA_SUFFIX), path.with_suffix(INDEX_SUFFIX)


def _scalar(array: np.ndarray) -> object:
    return np.asarray(array).reshape(()).item()


def load_index(hdf5_path: str | Path, expected_frames: int | None = None) -> dict | None:
    """Load and validate an episode sidecar, returning ``None`` if absent."""
    default_data_path, index_path = sidecar_paths(hdf5_path)
    if not index_path.is_file():
        return None
    with np.load(index_path, allow_pickle=False) as payload:
        version = int(_scalar(payload["version"]))
        if version != FORMAT_VERSION:
            raise ValueError(f"{index_path}: unsupported frame-store version {version}")
        codec = str(_scalar(payload["codec"]))
        if codec != "jpeg":
            raise ValueError(f"{index_path}: unsupported codec {codec!r}")
        data_name = str(_scalar(payload["data_file"]))
        data_path = index_path.parent / data_name if data_name else default_data_path
        offsets = {camera: np.asarray(payload[f"offsets__{camera}"], dtype=np.uint64) for camera in CAMERAS}
        digest = str(_scalar(payload["sha256"]))
        quality = int(_scalar(payload["jpeg_quality"]))

    if not data_path.is_file():
        raise FileNotFoundError(f"{index_path}: missing data file {data_path}")
    data_size = data_path.stat().st_size
    frame_counts = set()
    for camera, camera_offsets in offsets.items():
        if camera_offsets.ndim != 1 or len(camera_offsets) < 2:
            raise ValueError(f"{index_path}: invalid offsets for {camera}")
        if camera_offsets[0] > data_size or camera_offsets[-1] > data_size:
            raise ValueError(f"{index_path}: {camera} offset outside {data_path}")
        if np.any(np.diff(camera_offsets) <= 0):
            raise ValueError(f"{index_path}: {camera} records are empty or non-monotonic")
        frame_counts.add(len(camera_offsets) - 1)
    if len(frame_counts) != 1:
        raise ValueError(f"{index_path}: camera frame counts differ: {sorted(frame_counts)}")
    frame_count = frame_counts.pop()
    if expected_frames is not None and frame_count != expected_frames:
        raise ValueError(f"{index_path}: {frame_count} frames, expected {expected_frames}")
    return {
        "data_path": str(data_path),
        "index_path": str(index_path),
        "offsets": offsets,
        "num_frames": frame_count,
        "codec": codec,
        "jpeg_quality": quality,
        "sha256": digest,
    }


class _FileDescriptorCache:
    """Small process-local LRU of read-only descriptors used with ``pread``."""

    def __init__(self) -> None:
        self.pid = os.getpid()
        self.handles: OrderedDict[str, int] = OrderedDict()

    def _reset_after_fork(self) -> None:
        if self.pid == os.getpid():
            return
        self.close()
        self.pid = os.getpid()

    def get(self, path: str) -> int:
        self._reset_after_fork()
        if path in self.handles:
            fd = self.handles.pop(path)
            self.handles[path] = fd
            return fd
        fd = os.open(path, os.O_RDONLY)
        self.handles[path] = fd
        limit = max(1, int(os.environ.get("FRAME_STORE_OPEN_FILES_PER_WORKER", "16")))
        while len(self.handles) > limit:
            _, evicted = self.handles.popitem(last=False)
            os.close(evicted)
        return fd

    def close(self) -> None:
        for fd in self.handles.values():
            try:
                os.close(fd)
            except OSError:
                pass
        self.handles.clear()


_FD_CACHE = _FileDescriptorCache()


def load_frames(store: Mapping, camera: str, frame_indices: Sequence[int]) -> np.ndarray:
    """Read selected JPEG records and return RGB ``uint8`` frames in order."""
    if camera not in CAMERAS:
        raise KeyError(f"unknown camera {camera!r}")
    offsets = np.asarray(store["offsets"][camera], dtype=np.uint64)
    indices = [int(index) for index in frame_indices]
    if any(index < 0 or index + 1 >= len(offsets) for index in indices):
        raise IndexError(f"frame indices {indices} outside {camera} length {len(offsets) - 1}")
    fd = _FD_CACHE.get(str(store["data_path"]))
    decoded = []
    for index in indices:
        start, end = int(offsets[index]), int(offsets[index + 1])
        encoded = os.pread(fd, end - start, start)
        if len(encoded) != end - start:
            raise IOError(f"short read for {camera} frame {index}: {len(encoded)} != {end - start}")
        bgr = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"JPEG decode failed for {camera} frame {index}")
        decoded.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return np.stack(decoded, axis=0)


def build_from_videos(
    hdf5_path: str | Path,
    video_paths: Mapping[str, str | Path],
    expected_frames: int,
    *,
    jpeg_quality: int = 95,
    overwrite: bool = False,
) -> dict:
    """Sequentially transcode three episode MP4s into one indexed sidecar."""
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be in [1,100]")
    missing = sorted(set(CAMERAS) - set(video_paths))
    if missing:
        raise ValueError(f"missing video paths for {missing}")
    data_path, index_path = sidecar_paths(hdf5_path)
    if not overwrite:
        existing = load_index(hdf5_path, expected_frames=expected_frames)
        if existing is not None:
            return existing

    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_tmp = tempfile.NamedTemporaryFile(prefix=data_path.name + ".", suffix=".tmp", dir=data_path.parent, delete=False)
    data_tmp_path = Path(data_tmp.name)
    index_tmp_path: Path | None = None
    digest = hashlib.sha256()
    all_offsets: dict[str, np.ndarray] = {}
    try:
        with data_tmp:
            for camera in CAMERAS:
                source = str(video_paths[camera])
                capture = cv2.VideoCapture(source)
                if not capture.isOpened():
                    raise ValueError(f"could not open {camera} video {source}")
                offsets = [data_tmp.tell()]
                frame_count = 0
                try:
                    while True:
                        ok, bgr = capture.read()
                        if not ok:
                            break
                        encoded_ok, encoded = cv2.imencode(
                            ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
                        )
                        if not encoded_ok:
                            raise ValueError(f"JPEG encode failed for {camera} frame {frame_count}")
                        record = encoded.tobytes()
                        data_tmp.write(record)
                        digest.update(record)
                        offsets.append(data_tmp.tell())
                        frame_count += 1
                finally:
                    capture.release()
                if frame_count != expected_frames:
                    raise ValueError(f"{source}: decoded {frame_count} frames, expected {expected_frames}")
                all_offsets[camera] = np.asarray(offsets, dtype=np.uint64)
            data_tmp.flush()
            os.fsync(data_tmp.fileno())

        with tempfile.NamedTemporaryFile(
            prefix=index_path.name + ".", suffix=".tmp", dir=index_path.parent, delete=False
        ) as index_tmp:
            index_tmp_path = Path(index_tmp.name)
            np.savez(
                index_tmp,
                version=np.asarray(FORMAT_VERSION, dtype=np.int64),
                codec=np.asarray("jpeg"),
                jpeg_quality=np.asarray(jpeg_quality, dtype=np.int64),
                data_file=np.asarray(data_path.name),
                sha256=np.asarray(digest.hexdigest()),
                **{f"offsets__{camera}": offsets for camera, offsets in all_offsets.items()},
            )
            index_tmp.flush()
            os.fsync(index_tmp.fileno())
        os.replace(data_tmp_path, data_path)
        os.replace(index_tmp_path, index_path)
    except Exception:
        data_tmp_path.unlink(missing_ok=True)
        if index_tmp_path is not None:
            index_tmp_path.unlink(missing_ok=True)
        raise
    return load_index(hdf5_path, expected_frames=expected_frames)
