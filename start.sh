#!/usr/bin/env bash
# start.sh — VRPS-RunPod entrypoint.
#
# Responsibilities:
#   1. Redirect config.ROOT's models/ + runtime_cache/ onto the pod's writable
#      /workspace disk (config.ROOT is hardcoded to /app, so we symlink).
#      Baked-in models (DA3 base/base_hd, NVDS, RVM) are migrated to /workspace
#      on first boot, then the image dir becomes a symlink.
#   2. Optionally fetch heavy models (MatAnyone2, SAM3, DA3 large_hd) on demand.
#   3. Dispatch:
#        - no args / "serve"  -> idle loop (pod stays up; use job_runner via exec
#                                 or the RunPod API to run jobs)
#        - "job" + JSON        -> run one conversion via job_runner.py
#        - "two_dvr" / "convert" / "build-trt" / "download" -> pass through to
#                                 the underlying offline CLI
#        - anything else       -> exec as a raw command (debug)
set -euo pipefail

APP=/app
WS=/workspace
PY="${VENV:-/opt/venv}/bin/python"

log() { echo "[start] $*"; }

# --- 1. Workspace redirection -------------------------------------------------
mkdir -p "${WS}/models" "${WS}/runtime_cache" "${WS}/hf" "${WS}/inputs" "${WS}/outputs"

link_into_workspace() {  # $1 = subdir name under /app (e.g. "models")
    local name="$1"
    local appdir="${APP}/${name}"
    local wsdir="${WS}/${name}"
    if [[ -L "${appdir}" ]]; then
        return  # already a symlink (warm container restart)
    fi
    if [[ -d "${appdir}" ]]; then
        # First boot: migrate any baked content into /workspace, then replace
        # the image dir with a symlink. cp -an = no-clobber, preserve.
        log "migrating baked ${name}/ -> ${wsdir}"
        cp -an "${appdir}/." "${wsdir}/" 2>/dev/null || true
        rm -rf "${appdir}"
    fi
    ln -s "${wsdir}" "${appdir}"
}

link_into_workspace models
link_into_workspace runtime_cache

# --- 2. Optional heavy-model fetch -------------------------------------------
# Set VRPS_FETCH_HEAVY=1 to pull MatAnyone2 + SAM3 + DA3 large_hd at boot.
# Otherwise they are fetched lazily the first time a job requests them (the
# upstream code's ensure_model_available / snapshot_download handles DA3; for
# MatAnyone2/SAM3 the job_runner triggers build-models.sh --heavy on demand).
if [[ "${VRPS_FETCH_HEAVY:-0}" == "1" ]]; then
    log "VRPS_FETCH_HEAVY=1 -> fetching heavy models now"
    bash "${APP}/build-models.sh" --heavy || log "heavy fetch had errors (continuing)"
fi

# --- 3. Dispatch --------------------------------------------------------------
cmd="${1:-serve}"
case "${cmd}" in
    serve|"")
        log "idle serve mode. Pod is up. Run jobs via:"
        log "  docker exec / runpodctl exec -> ${PY} /app/job_runner.py '<json>'"
        log "  or python -m offline.two_dvr single <video> [opts]"
        # Keep the container alive for interactive / API-driven job submission.
        tail -f /dev/null
        ;;
    job)
        shift
        exec "${PY}" "${APP}/job_runner.py" "$@"
        ;;
    two_dvr)
        shift
        cd "${APP}"
        exec "${PY}" -m offline.two_dvr "$@"
        ;;
    convert)
        shift
        cd "${APP}"
        exec "${PY}" -m offline.convert "$@"
        ;;
    build-trt|download)
        cd "${APP}"
        exec "${PY}" -m offline.two_dvr "$@"
        ;;
    fetch-heavy)
        exec bash "${APP}/build-models.sh" --heavy
        ;;
    *)
        # Raw command passthrough (debugging).
        exec "$@"
        ;;
esac
