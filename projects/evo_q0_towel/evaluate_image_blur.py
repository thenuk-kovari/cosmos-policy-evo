#!/usr/bin/env python3
"""Compare clean and Gaussian-blurred Evo-q0 predictions for one training MCAP."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
import tritonclient.grpc as grpcclient
from PIL import Image, ImageFilter

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from cosmos_policy.datasets.evo_q0_actions import CHUNK_SIZE, build_q0_anchored_chunk
from projects.evo_q0_towel.prepare_mcap import (
    CAMERA_TOPICS,
    decode_image,
    interpolate_states,
    measured_state_stream,
    nearest_indices,
    read_mcap,
    select_camera_stream,
)


def jpeg(image: np.ndarray) -> np.ndarray:
    stream = io.BytesIO()
    Image.fromarray(image, "RGB").save(stream, format="JPEG", quality=95)
    return np.frombuffer(stream.getvalue(), dtype=np.uint8)


def gaussian_blur(image: np.ndarray, radius: float) -> np.ndarray:
    """Apply the paired image intervention in RGB space before JPEG transport."""
    return np.asarray(Image.fromarray(image, "RGB").filter(ImageFilter.GaussianBlur(radius)))


def infer(client: grpcclient.InferenceServerClient, state: np.ndarray, images: dict[str, np.ndarray]) -> np.ndarray:
    arrays = {
        "observation__state": np.asarray(state, dtype=np.float32),
        "observation__images__base": jpeg(images["cam_high"]),
        "observation__images__left_gripper": jpeg(images["cam_left_wrist"]),
        "observation__images__right_gripper": jpeg(images["cam_right_wrist"]),
    }
    inputs = []
    for name, value in arrays.items():
        datatype = "FP32" if value.dtype == np.float32 else "UINT8"
        tensor = grpcclient.InferInput(name, list(value.shape), datatype)
        tensor.set_data_from_numpy(value)
        inputs.append(tensor)
    result = client.infer("slot_b", inputs, client_timeout=300)
    action = np.asarray(result.as_numpy("action"), dtype=np.float32)
    if action.shape != (CHUNK_SIZE, 17) or not np.isfinite(action).all():
        raise RuntimeError(f"invalid action output {action.shape}")
    return action


def svg_plot(
    path: Path,
    frames: np.ndarray,
    clean: np.ndarray,
    blurred: np.ndarray,
    blur_radius: float,
) -> None:
    width, height = 1280, 640
    left, right, top, bottom = 85, 30, 45, 75
    plot_w, plot_h = width - left - right, height - top - bottom
    x0, x1 = float(frames.min()), float(frames.max())
    y1 = max(float(clean.max()), float(blurred.max()), 1e-8)

    def point(x: float, y: float) -> tuple[float, float]:
        px = left + (x - x0) / max(x1 - x0, 1.0) * plot_w
        py = top + (1.0 - y / y1) * plot_h
        return px, py

    def polyline(values: np.ndarray, color: str) -> str:
        points = " ".join(f"{x:.2f},{y:.2f}" for x, y in (point(a, b) for a, b in zip(frames, values)))
        return f'<polyline fill="none" stroke="{color}" stroke-width="1.5" points="{points}"/>'

    grid = []
    labels = []
    for tick in range(6):
        value = y1 * tick / 5
        _, y = point(x0, value)
        grid.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#ddd"/>')
        labels.append(f'<text x="{left - 10}" y="{y + 5:.2f}" text-anchor="end">{value:.4f}</text>')
    body = f"""<!doctype html><meta charset="utf-8"><title>Evo q0 image-blur sensitivity</title>
<style>body{{font-family:sans-serif;margin:24px}} text{{font-size:13px}} .legend{{font-size:15px}}</style>
<h2>Evo q0: clean versus Gaussian-blurred prediction error</h2>
<p>Gaussian blur radius: {blur_radius:.1f}px. Error is mean(|prediction − ground truth|) over 17 physical action channels.</p>
<svg viewBox="0 0 {width} {height}" width="100%" xmlns="http://www.w3.org/2000/svg">
{"".join(grid)}{"".join(labels)}
<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="black"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="black"/>
{polyline(clean, "#1769aa")}{polyline(blurred, "#d32f2f")}
<text x="{width / 2}" y="{height - 18}" text-anchor="middle">MCAP timestep (30 Hz)</text>
<text transform="translate(20 {height / 2}) rotate(-90)" text-anchor="middle">mean absolute action error</text>
<line x1="{left + 15}" y1="20" x2="{left + 55}" y2="20" stroke="#1769aa" stroke-width="3"/>
<text class="legend" x="{left + 62}" y="25">clean images</text>
<line x1="{left + 180}" y1="20" x2="{left + 220}" y2="20" stroke="#d32f2f" stroke-width="3"/>
<text class="legend" x="{left + 227}" y="25">Gaussian-blurred images</text>
</svg>"""
    path.write_text(body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mcap", type=Path)
    parser.add_argument("--triton-url", default="192.168.90.2:8201")
    parser.add_argument("--out", type=Path, default=Path("/tmp/evoq0_image_blur"))
    parser.add_argument("--blur-radius", type=float, default=25.0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    streams = read_mcap(args.mcap)
    state_t, raw_states = measured_state_stream(streams["/joint_states"])
    cameras = {name: select_camera_stream(streams, name)[1] for name in CAMERA_TOPICS}
    # Match the training converter's 30 Hz timeline exactly: base camera is
    # authoritative, and wrist cameras are nearest-neighbor sampled onto it.
    start = max(state_t[0], cameras["cam_high"][0][0])
    stop = min(state_t[-1], cameras["cam_high"][-1][0])
    grid = np.arange(start, stop, 1.0 / 30.0, dtype=np.float64)
    states = interpolate_states(state_t, raw_states, grid, max_gap_s=0.2)
    camera_indices = {
        name: nearest_indices(np.asarray([row[0] for row in rows]), grid) for name, rows in cameras.items()
    }
    anchors = np.arange(0, len(grid) - 1, CHUNK_SIZE, dtype=np.int64)
    client = grpcclient.InferenceServerClient(url=args.triton_url)
    if not client.is_model_ready("slot_b"):
        raise RuntimeError(f"slot_b is not ready at {args.triton_url}")

    clean_predictions, blurred_predictions, ground_truth, valid_rows = [], [], [], []
    for number, anchor in enumerate(anchors, start=1):
        images = {name: decode_image(cameras[name][int(camera_indices[name][anchor])][1]) for name in cameras}
        blurred_images = {name: gaussian_blur(image, args.blur_radius) for name, image in images.items()}
        clean_predictions.append(infer(client, states[anchor], images))
        blurred_predictions.append(infer(client, states[anchor], blurred_images))
        ground_truth.append(build_q0_anchored_chunk(states, int(anchor)))
        valid_rows.append(min(CHUNK_SIZE, len(states) - int(anchor) - 1))
        print(f"[{number}/{len(anchors)}] anchor={anchor} valid_rows={valid_rows[-1]}", flush=True)

    clean_array = np.stack(clean_predictions)
    blurred_array = np.stack(blurred_predictions)
    truth_array = np.stack(ground_truth)
    valid_array = np.asarray(valid_rows, dtype=np.int64)
    np.savez_compressed(
        args.out / "chunks.npz",
        anchors=anchors,
        valid_rows=valid_array,
        clean_prediction=clean_array,
        blurred_prediction=blurred_array,
        ground_truth=truth_array,
    )

    frame_rows, clean_errors, blurred_errors = [], [], []
    with (args.out / "errors.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame", "seconds", "anchor", "chunk_row", "clean_mae", "blurred_mae"])
        for chunk_index, anchor in enumerate(anchors):
            for row in range(int(valid_array[chunk_index])):
                frame = int(anchor + row + 1)
                clean_error = float(np.abs(clean_array[chunk_index, row] - truth_array[chunk_index, row]).mean())
                blurred_error = float(np.abs(blurred_array[chunk_index, row] - truth_array[chunk_index, row]).mean())
                writer.writerow([frame, frame / 30.0, int(anchor), row, clean_error, blurred_error])
                frame_rows.append(frame)
                clean_errors.append(clean_error)
                blurred_errors.append(blurred_error)

    frame_array = np.asarray(frame_rows)
    clean_error_array = np.asarray(clean_errors)
    blurred_error_array = np.asarray(blurred_errors)
    svg_plot(args.out / "plot.html", frame_array, clean_error_array, blurred_error_array, args.blur_radius)
    summary = {
        "mcap": str(args.mcap),
        "frames": len(grid),
        "chunks": len(anchors),
        "gaussian_blur_radius_px": args.blur_radius,
        "clean_mae": float(clean_error_array.mean()),
        "blurred_mae": float(blurred_error_array.mean()),
        "blurred_to_clean_ratio": float(blurred_error_array.mean() / max(clean_error_array.mean(), 1e-12)),
        "plot": str(args.out / "plot.html"),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
