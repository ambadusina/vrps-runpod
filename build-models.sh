#!/usr/bin/env bash
# build-models.sh — fetch VRPS ONNX models into /app/models.
#
# Hybrid policy:
#   --baked-only  (build time): DA3 base + base_hd, NVDS, RVM   (~1 GB, into image)
#   --heavy       (runtime)   : MatAnyone2, SAM3, DA3 large_hd  (cached on pod disk
#                               via the /app/models -> /workspace/models symlink)
#   --all                     : both sets
#
# DA3 + NVDS use the app's own downloaders (mirror-aware) via the two_dvr CLI.
# RVM comes from the official GitHub release. MatAnyone2 / SAM3 are pulled from
# their HF repos with huggingface_hub if available, else curl.
set -euo pipefail

MODE="${1:-}"
APP=/app
MODELS="${APP}/models"
PY="${VENV:-/opt/venv}/bin/python"
cd "${APP}"

log() { echo "[build-models] $*"; }

dl_da3() {  # $1 = preset
    log "DA3 download: $1"
    "${PY}" -m offline.two_dvr download --model "$1" || {
        log "DA3 $1 via two_dvr failed (continuing; will retry at runtime)"; return 0; }
}

dl_nvds() {
    log "NVDS download (both tiers)"
    "${PY}" - <<'PYEOF' || true
import urllib.request
from offline import nvds_stabilizer as n
for res in ("512x288", "672x384"):
    w, h = n.resolve_resolution(res)
    for name, dest, urls in n.download_targets(w, h):
        dest.parent.mkdir(parents=True, exist_ok=True)
        ok = False
        for url in urls:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "VRPS/NVDS"})
                with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
                    while (chunk := r.read(1 << 20)):
                        f.write(chunk)
                print(f"[build-models] NVDS {name} ok ({dest.stat().st_size/1e6:.1f} MB)")
                ok = True
                break
            except Exception as e:
                print(f"[build-models] NVDS {name} <- {url} failed: {type(e).__name__}")
        if not ok:
            print(f"[build-models] NVDS {name} unavailable (will retry at runtime)")
PYEOF
}

dl_rvm() {
    mkdir -p "${MODELS}"
    local base="https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0"
    for f in rvm_mobilenetv3_fp32.onnx rvm_resnet50_fp32.onnx; do
        if [[ ! -f "${MODELS}/${f}" ]]; then
            log "RVM download: ${f}"
            curl -fSL --retry 3 -o "${MODELS}/${f}" "${base}/${f}" || \
                log "RVM ${f} failed (continuing)"
        fi
    done
}

dl_matanyone2() {
    log "MatAnyone2 download (512_bs1, 1024_bs1)"
    "${PY}" - <<'PYEOF' || true
import sys
try:
    from huggingface_hub import snapshot_download
except Exception:
    print("[build-models] huggingface_hub missing; install it or fetch MatAnyone2 manually")
    sys.exit(0)
import os
dest = "/app/models"
for sub in ("matanyone2_onnx_512_bs1", "matanyone2_onnx_1024_bs1"):
    try:
        snapshot_download("zerochocobo/matanyone2_onnx", allow_patterns=[f"{sub}/*"],
                          local_dir=dest, local_dir_use_symlinks=False)
        print("[build-models] MatAnyone2", sub, "ok")
    except Exception as e:
        print("[build-models] MatAnyone2", sub, "failed:", e)
PYEOF
}

dl_sam3() {
    log "SAM3 download (encoder/decoder/language + .data)"
    "${PY}" - <<'PYEOF' || true
import sys
try:
    from huggingface_hub import snapshot_download
except Exception:
    print("[build-models] huggingface_hub missing; fetch SAM3 manually")
    sys.exit(0)
try:
    snapshot_download("wkentaro/sam3-onnx-models",
                      local_dir="/app/models/sam3_onnx",
                      local_dir_use_symlinks=False)
    print("[build-models] SAM3 ok")
except Exception as e:
    print("[build-models] SAM3 failed:", e)
PYEOF
}

baked_set() {
    dl_da3 base
    dl_da3 base_hd
    dl_nvds
    dl_rvm
}

heavy_set() {
    dl_da3 large_hd
    dl_matanyone2
    dl_sam3
}

case "${MODE}" in
    --baked-only) baked_set ;;
    --heavy)      heavy_set ;;
    --all)        baked_set; heavy_set ;;
    *) echo "usage: build-models.sh [--baked-only|--heavy|--all]" >&2; exit 2 ;;
esac

log "done (${MODE})"
du -sh "${MODELS}" 2>/dev/null || true
