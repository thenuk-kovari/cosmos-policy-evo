from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from cosmos_policy.datasets.indexed_frame_store import (
    CAMERAS,
    build_from_videos,
    load_frames,
    load_index,
    sidecar_paths,
)


def write_video(path: Path, frames_rgb: np.ndarray) -> None:
    height, width = frames_rgb.shape[1:3]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height))
    if not writer.isOpened():
        pytest.skip("OpenCV MP4 writer unavailable")
    try:
        for frame in frames_rgb:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def test_indexed_frame_store_round_trip(tmp_path: Path) -> None:
    frame_count = 11
    source_frames = {}
    videos = {}
    yy, xx = np.mgrid[:48, :64]
    for camera_index, camera in enumerate(CAMERAS):
        frames = []
        for frame_index in range(frame_count):
            frames.append(
                np.stack(
                    (
                        (xx + frame_index * 7 + camera_index * 11) % 256,
                        (yy * 3 + frame_index * 5) % 256,
                        ((xx + yy) * 2 + camera_index * 17) % 256,
                    ),
                    axis=-1,
                ).astype(np.uint8)
            )
        source_frames[camera] = np.stack(frames)
        videos[camera] = tmp_path / f"{camera}.mp4"
        write_video(videos[camera], source_frames[camera])

    episode = tmp_path / "episode_000.hdf5"
    episode.touch()
    store = build_from_videos(episode, videos, frame_count, jpeg_quality=95)
    assert store is not None
    assert store["num_frames"] == frame_count
    assert load_index(episode, expected_frames=frame_count)["sha256"] == store["sha256"]
    with pytest.raises(ValueError, match="expected 12"):
        load_index(episode, expected_frames=frame_count + 1)

    indices = [0, 5, 10, 5]
    for camera in CAMERAS:
        decoded = load_frames(store, camera, indices)
        assert decoded.shape == (len(indices), 48, 64, 3)
        assert decoded.dtype == np.uint8
        assert np.array_equal(decoded[1], decoded[3])
        # The source MP4 and sidecar JPEG are both lossy. This guards against
        # camera/order corruption without requiring pixel identity.
        assert np.mean(np.abs(decoded[0].astype(np.int16) - source_frames[camera][0].astype(np.int16))) < 12

    data_path, index_path = sidecar_paths(episode)
    assert data_path.stat().st_size > 0
    assert index_path.stat().st_size > 0


def test_absent_index(tmp_path: Path) -> None:
    episode = tmp_path / "episode_001.hdf5"
    episode.touch()
    assert load_index(episode) is None
