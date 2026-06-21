# VRPS-RunPod — offline VR/3D + passthrough on RunPod

Headless Docker packaging of the **offline** half of
[VR-Video-Passthrough-Server](https://github.com/zerochocobo/VR-Video-Passthrough-Server)
(AGPL-3.0), for batch GPU jobs on RunPod — same operational shape as the Jasna
service.

It keeps only the offline features:

- **Offline passthrough video generation** (matting: RVM / MatAnyone2, green-screen or alpha) — `offline/convert.py`
- **Offline 2D→3D / VR** for flat 2D video, via DA3 ONNX depth + GPU stereo render — `offline/two_dvr.py`
- **Depth stabilization** — built-in temporal (`--temporal-depth`) and NVDS ONNX (`--depth-stabilizer nvds`, 16:9 only)

The realtime DLNA/UPnP server, PySide6 UI, subtitles, and streaming pipeline are
dropped.

## Why CUDA 12.8 (not 12.6)

Your GPU is a **Blackwell** card (RTX PRO 4500, sm_120). CuPy must emit native
sm_120 cubins through NVRTC ≥ 12.8; on 12.6 it falls back to a slow PTX→driver
JIT or fails with `CUDA_ERROR_NO_BINARY_FOR_GPU`. The base image is therefore
`nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04`. **RunPod host driver ≥ 570**;
filter pods for **CUDA 12.8+** (different from Jasna's CUDA 13 filter).

## Model strategy (hybrid)

| When | Models | Where they live |
|---|---|---|
| **Baked into image** (build time, ~1 GB) | DA3 `base` + `base_hd`, NVDS (both tiers), RVM | image → migrated to `/workspace/models` on first boot |
| **Fetched on demand** (runtime) | MatAnyone2 (512/1024), SAM3, DA3 `large_hd` | `/workspace/models` on the pod disk |
| **Never baked** | TensorRT engines | built on the pod (GPU-specific), cached in `/workspace/runtime_cache` |

`config.ROOT` is hardcoded to the source dir, so `start.sh` symlinks
`/app/models` and `/app/runtime_cache` onto the pod's writable `/workspace`
disk. TRT engines are intentionally not baked: an engine compiled on another GPU
is unusable.

To prefetch the heavy set at boot: set `VRPS_FETCH_HEAVY=1`. Otherwise the job
runner pulls them the first time a job asks for a heavy engine/model.

## Build

CI (recommended — GitHub Actions is the reliable path here):

1. Repo secrets: `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`.
2. Push to `main` → image at `<user>/vrps-runpod:latest`.

Local:

```bash
docker build -t <user>/vrps-runpod:latest .
docker push <user>/vrps-runpod:latest
```

## Run on RunPod

Create a pod (CUDA 12.8+ filter, the RTX PRO 4500 or similar), image
`<user>/vrps-runpod:latest`, container disk large enough for `/workspace`
(models + I/O — budget ~20 GB if you use MatAnyone2 + SAM3). The default command
is `serve` (idle; pod stays up for job submission).

### Submit a job

The job runner takes a single JSON spec and runs the right offline CLI with a
**list argv** (no shell), so option values can't inject commands.

2D→3D / VR:

```bash
runpodctl exec python -- /opt/venv/bin/python /app/job_runner.py '{
  "task": "2d3d",
  "input": "/workspace/inputs/clip.mp4",
  "out_dir": "/workspace/outputs",
  "options": {
    "model": "base",
    "projection": "flat3d",
    "strength": 1.2,
    "depth-stabilizer": "nvds",
    "nvds-res": "512x288",
    "temporal-depth": true
  }
}'
```

Passthrough matting (alpha, MatAnyone2 — heavy model auto-fetched):

```bash
... /app/job_runner.py '{
  "task": "matting",
  "input": "/workspace/inputs/clip.mp4",
  "out_dir": "/workspace/outputs",
  "options": { "mode": "alpha", "engine": "matanyone2", "matanyone2-size": 1024 }
}'
```

Batch a directory:

```bash
... /app/job_runner.py '{ "task": "2d3d", "input": "/workspace/inputs", "batch": true,
                          "options": { "model": "base_hd", "projection": "hequirect" } }'
```

Or bypass the runner and call the CLI directly:

```bash
/app/start.sh two_dvr single /workspace/inputs/clip.mp4 --out-dir /workspace/outputs \
  --model base --projection flat3d --depth-stabilizer nvds
/app/start.sh build-trt --model both --include-nvds   # warm the TRT cache once
```

### Pre-warm TensorRT (recommended first step on a fresh pod)

```bash
/app/start.sh build-trt --model both --include-nvds
```

Engines land in `/workspace/runtime_cache` and persist for the pod's life.

## Whitelisted job options

The runner forwards only validated options (type/choice-checked). 2D→3D:
`model, projection, hole-fill, eye-distance, strength, flat-fov, max-side,
batch, preset, bitrate, provider, gpu-render, pipeline, depth-stabilizer,
nvds-res`, plus boolean toggles `temporal-depth, temporal-norm,
temporal-affine`. Matting: `mode, engine, fps, input-size,
rvm-downsample-ratio, skip-frames, bitrate, preset, cq, matanyone2-size,
matanyone2-prepass, sam3-prompt`. Anything else is ignored; invalid values are
rejected before exec.

## MVP pipeline = ffmpeg

PyNvVideoCodec is **not** installed (it needs CUDA at C++ compile time — the same
build pain as Jasna's vali). `two_dvr` runs with `--pipeline ffmpeg` (NVENC via
the distro ffmpeg). Add PyNvVideoCodec later for the GPU-resident fast path if
throughput needs it.

## AGPL-3.0

This image bundles AGPL source. If you expose it over a network to users, you
must publish the complete corresponding source — this repo (Dockerfile,
start.sh, build-models.sh, job_runner.py, the CI workflow) **and** the upstream
VRPS source it's built from, with your modifications. Keep the GitHub repo public
and linked from the running service. Same network-clause obligation as Jasna.

Also check the license of each ONNX model before charging for output — some DA3
variants (e.g. the Nested Giant-Large) are CC BY-NC 4.0 (non-commercial).
