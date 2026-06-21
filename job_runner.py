#!/usr/bin/env python3
"""VRPS-RunPod job runner.

Reads a single JSON job spec (argv[1] or stdin) and runs the corresponding
offline conversion via the upstream CLIs, using a *list* argv passed straight to
subprocess (never a shell string) so untrusted option values cannot inject
shell commands.

Job spec
--------
{
  "task": "2d3d" | "matting",     # which offline pipeline
  "input": "/workspace/inputs/clip.mp4",   # or a directory for batch
  "batch": false,                  # true -> process a directory recursively
  "out_dir": "/workspace/outputs",
  "options": { ... }               # task-specific, see ALLOWED_* below
}

Only whitelisted options are forwarded, each validated by type/choice, so a
caller (e.g. a web front-end) cannot smuggle arbitrary flags or shell syntax.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

APP = Path("/app")
PY = sys.executable

# ---- Option whitelists -------------------------------------------------------
# Each entry: flag -> validator. Validators return the string value to pass, or
# raise ValueError. value None / missing => flag omitted (use pipeline default).

def _choice(*allowed):
    def v(x):
        s = str(x)
        if s not in allowed:
            raise ValueError(f"must be one of {allowed}, got {s!r}")
        return s
    return v

def _float(lo=None, hi=None):
    def v(x):
        f = float(x)
        if lo is not None and f < lo: raise ValueError(f"< {lo}")
        if hi is not None and f > hi: raise ValueError(f"> {hi}")
        return repr(f)
    return v

def _int(lo=None, hi=None, choices=None):
    def v(x):
        i = int(x)
        if choices is not None and i not in choices: raise ValueError(f"not in {choices}")
        if lo is not None and i < lo: raise ValueError(f"< {lo}")
        if hi is not None and i > hi: raise ValueError(f"> {hi}")
        return str(i)
    return v

def _bitrate(x):
    s = str(x)
    # e.g. "40M", "source", "12000k"
    if s == "source" or (s[:-1].isdigit() and s[-1] in "kKmM") or s.isdigit():
        return s
    raise ValueError("bad bitrate")

# 2D->3D/VR (offline.two_dvr single|batch)
ALLOWED_2D3D = {
    "--model":            _choice("small", "base", "small_hd", "base_hd", "large_hd"),
    "--projection":       _choice("flat3d", "hequirect", "fisheye"),
    "--hole-fill":        _choice("soft_shift", "inverse_warp"),
    "--eye-distance":     _float(1.0, 200.0),
    "--strength":         _float(0.1, 3.0),
    "--flat-fov":         _float(10.0, 160.0),
    "--max-side":         _int(256, 8192),
    "--batch":            _int(1, 64),          # internal frame batch size
    "--preset":           _choice("p1","p2","p3","p4","p5","p6","p7"),
    "--bitrate":          _bitrate,
    "--provider":         _choice("trt", "cuda", "cpu"),
    "--gpu-render":       _choice("auto", "on", "off"),
    "--pipeline":         _choice("auto", "pynv", "ffmpeg"),
    "--depth-stabilizer": _choice("default", "nvds"),   # default = built-in temporal; nvds = NVDS ONNX (16:9 only)
    "--nvds-res":         _choice("512x288", "672x384"),
}

# Boolean toggles for 2d3d (argparse BooleanOptionalAction: --flag / --no-flag).
# Value true -> "--flag", false -> "--no-flag", omitted -> pipeline default.
BOOL_TOGGLES_2D3D = {
    "temporal-depth": "--temporal-depth",   # built-in temporal depth stabilization
    "temporal-norm":  "--temporal-norm",    # temporal near/disparity normalization
    "temporal-affine": "--temporal-affine",
}

# Matting passthrough (offline.convert single|batch)
ALLOWED_MATTING = {
    "--mode":              _choice("green", "alpha"),
    "--engine":            _choice("rvm_fast", "matanyone2_medium", "matanyone2"),
    "--fps":               _float(0.0, 240.0),
    "--input-size":        _int(256, 4096),
    "--rvm-downsample-ratio": _float(0.0625, 1.0),
    "--skip-frames":       _int(choices=[0, 1, 2]),
    "--bitrate":           _bitrate,
    "--preset":            _choice("p1","p2","p3","p4","p5","p6","p7"),
    "--cq":                _int(-1, 51),
    "--matanyone2-size":   _int(choices=[512, 1024]),
    "--matanyone2-prepass": _choice("yolo26m_efficientsam", "yolo26m_birefnet"),
    "--sam3-prompt":       lambda x: str(x)[:128],  # length-capped free text, passed as a single argv element (no shell)
}

# Engines that need heavy (runtime-fetched) models.
HEAVY_ENGINES = {"matanyone2_medium", "matanyone2"}
HEAVY_MODELS_2D3D = {"large_hd"}


def _build_argv(task: str, spec: dict) -> list[str]:
    inp = spec.get("input")
    if not inp:
        raise ValueError("missing 'input'")
    inp_path = Path(inp)
    is_batch = bool(spec.get("batch", False))
    out_dir = spec.get("out_dir", "/workspace/outputs")
    opts = spec.get("options", {}) or {}

    if task == "2d3d":
        module = "offline.two_dvr"
        allowed = ALLOWED_2D3D
    elif task == "matting":
        module = "offline.convert"
        allowed = ALLOWED_MATTING
    else:
        raise ValueError(f"unknown task {task!r}")

    sub = "batch" if is_batch else "single"
    argv = [PY, "-m", module, sub, str(inp_path)]

    if is_batch:
        # batch writes next to each source for two_dvr; convert supports out via cwd.
        if spec.get("recursive", True) is False:
            argv.append("--no-recursive")
    else:
        argv += ["--out-dir", str(out_dir)]
        start = spec.get("start"); dur = spec.get("duration")
        if start is not None: argv += ["--start", repr(float(start))]
        if dur is not None:   argv += ["--duration", repr(float(dur))]

    for flag, validator in allowed.items():
        key = flag.lstrip("-")
        # accept both "--model" and "model" keys in options
        if flag in opts:        raw = opts[flag]
        elif key in opts:       raw = opts[key]
        else:                   continue
        if raw is None:
            continue
        try:
            value = validator(raw)
        except (ValueError, TypeError) as e:
            raise ValueError(f"invalid value for {flag}: {e}")
        argv += [flag, value]

    if spec.get("skip_existing"):
        argv.append("--skip-existing")

    # Boolean toggles (2d3d only).
    if task == "2d3d":
        for key, flag in BOOL_TOGGLES_2D3D.items():
            if key in opts or flag in opts:
                val = opts.get(key, opts.get(flag))
                if val is True:
                    argv.append(flag)
                elif val is False:
                    argv.append(flag.replace("--", "--no-", 1))

    return argv


def _maybe_fetch_heavy(task: str, spec: dict) -> None:
    opts = spec.get("options", {}) or {}
    need = False
    if task == "2d3d":
        model = opts.get("--model") or opts.get("model")
        if model in HEAVY_MODELS_2D3D:
            need = True
    elif task == "matting":
        eng = opts.get("--engine") or opts.get("engine")
        if eng in HEAVY_ENGINES:
            need = True
    if need:
        print("[job_runner] heavy model required; ensuring it is present...", flush=True)
        subprocess.run(["bash", str(APP / "build-models.sh"), "--heavy"], check=False)


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] not in ("-",):
        raw = sys.argv[1]
    else:
        raw = sys.stdin.read()
    try:
        spec = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[job_runner] bad JSON: {e}", file=sys.stderr)
        return 2

    task = spec.get("task", "2d3d")
    try:
        _maybe_fetch_heavy(task, spec)
        argv = _build_argv(task, spec)
    except ValueError as e:
        print(f"[job_runner] rejected: {e}", file=sys.stderr)
        return 2

    print("[job_runner] exec:", " ".join(argv), flush=True)
    # cwd=/app so `python -m offline.*` resolves the package.
    proc = subprocess.run(argv, cwd=str(APP))
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
