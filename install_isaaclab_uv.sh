#!/usr/bin/env bash
set -Eeuo pipefail

# Optional debug mode: DEBUG=1 ./install_isaaclab_uv.sh
if [[ "${DEBUG:-0}" == "1" ]]; then
  set -x
fi

if [[ -z "${BASH_VERSION:-}" ]]; then
  echo "ERROR: This script requires bash. Run with: bash ./install_isaaclab_uv.sh" >&2
  exit 1
fi

on_error() {
  local exit_code="$?"
  local line_no="$1"
  echo "ERROR: command failed at line ${line_no} with exit code ${exit_code}" >&2
  exit "${exit_code}"
}
trap 'on_error $LINENO' ERR

# Isaac Lab install script (Linux + uv), based on:
# https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html

# -----------------------------
# Config (override via env vars)
# -----------------------------
ENV_NAME="${ENV_NAME:-env_isaaclab}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$HOME/workspaces}"
REPO_DIR="${REPO_DIR:-$WORKSPACE_DIR/IsaacLab}"
REPO_URL="${REPO_URL:-https://github.com/isaac-sim/IsaacLab.git}"
REPO_REF="${REPO_REF:-main}"
ISAACSIM_PKG="${ISAACSIM_PKG:-isaacsim[all,extscache]==5.1.0}"
INSTALL_LIB="${INSTALL_LIB:-all}"         # all|rl_games|rsl_rl|sb3|skrl|robomimic|none
INSTALL_APT_DEPS="${INSTALL_APT_DEPS:-1}" # 1 to install cmake/build-essential

log() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; }
die() {
  printf '\nERROR: %s\n' "$*" >&2
  exit 1
}
need_cmd() { command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"; }

install_uv_if_needed() {
  if command -v uv >/dev/null 2>&1; then
    log "uv already installed: $(uv --version)"
    return
  fi

  log "uv not found; installing via official installer..."
  need_cmd curl
  sh -c "$(curl -LsSf https://astral.sh/uv/install.sh)"

  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || die "uv install finished but uv is still not on PATH."
  log "uv installed: $(uv --version)"
}

install_apt_deps_if_requested() {
  if [[ "${INSTALL_APT_DEPS}" != "1" ]]; then
    log "Skipping apt dependencies (INSTALL_APT_DEPS=${INSTALL_APT_DEPS})."
    return
  fi
  need_cmd apt-get
  log "Installing apt dependencies: cmake build-essential"
  if [[ "$(id -u)" -eq 0 ]]; then
    apt-get update
    apt-get install -y cmake build-essential
  else
    need_cmd sudo
    sudo apt-get update
    sudo apt-get install -y cmake build-essential
  fi
}

select_torch_spec() {
  local arch
  arch="$(uname -m)"
  case "$arch" in
  x86_64)
    TORCH_SPEC="torch==2.7.0 torchvision==0.22.0"
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
    ;;
  aarch64)
    TORCH_SPEC="torch==2.9.0 torchvision==0.24.0"
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu130"
    ;;
  *)
    die "Unsupported architecture: ${arch}. Supported: x86_64, aarch64."
    ;;
  esac
  log "Detected architecture: ${arch}"
}

install_flatdict_workaround() {
  log "Applying uv build workaround for flatdict==4.0.1 (pkg_resources)"
  # flatdict 4.0.1 imports pkg_resources during build.
  # pkg_resources is removed in newer setuptools, so keep setuptools < 81.
  uv pip install -U "setuptools<81" wheel
  python - <<'PY'
import pkg_resources  # noqa: F401
print("pkg_resources import check: OK")
PY
  uv pip install --no-build-isolation flatdict==4.0.1
}

clone_or_update_repo() {
  need_cmd git
  mkdir -p "${WORKSPACE_DIR}"
  if [[ ! -d "${REPO_DIR}/.git" ]]; then
    log "Cloning IsaacLab into ${REPO_DIR}"
    git clone "${REPO_URL}" "${REPO_DIR}"
  else
    log "IsaacLab repo already exists at ${REPO_DIR}; reusing it."
  fi

  log "Checking out ${REPO_REF}"
  git -C "${REPO_DIR}" fetch --all --tags
  git -C "${REPO_DIR}" checkout "${REPO_REF}"
  # Pull only when local branch tracks a remote branch.
  if git -C "${REPO_DIR}" rev-parse --abbrev-ref --symbolic-full-name "@{u}" >/dev/null 2>&1; then
    git -C "${REPO_DIR}" pull --ff-only
  fi
}

main() {
  log "Starting Isaac Lab + Isaac Sim setup with uv"
  log "Step 1/7: Checking/installing uv"
  install_uv_if_needed
  log "Step 2/7: Installing apt dependencies (if enabled)"
  install_apt_deps_if_requested
  log "Step 3/7: Selecting PyTorch CUDA wheel set"
  select_torch_spec

  log "Step 4/7: Creating Python environment"
  log "Creating virtual environment: ${ENV_NAME} (Python ${PYTHON_VERSION})"
  uv venv --python "${PYTHON_VERSION}" --seed "${ENV_NAME}"

  # shellcheck disable=SC1090
  source "${ENV_NAME}/bin/activate"
  log "Activated venv: ${VIRTUAL_ENV}"

  log "Upgrading pip and build tooling inside venv"
  uv pip install --upgrade pip "setuptools<81" wheel

  log "Installing Isaac Sim package from NVIDIA index"
  uv pip install "${ISAACSIM_PKG}" --extra-index-url https://pypi.nvidia.com

  log "Installing CUDA-enabled PyTorch from ${TORCH_INDEX_URL}"
  uv pip install -U ${TORCH_SPEC} --index-url "${TORCH_INDEX_URL}"

  log "Step 5/7: Installing known build workaround dependencies"
  install_flatdict_workaround

  log "Step 6/7: Cloning/updating Isaac Lab repository"
  clone_or_update_repo

  log "Step 7/7: Installing Isaac Lab packages"
  log "Installing Isaac Lab extensions/frameworks: ${INSTALL_LIB}"
  (
    cd "${REPO_DIR}"
    ./isaaclab.sh --install "${INSTALL_LIB}"
  )

  log "Installation complete."
  cat <<EOF

Next commands:
  source "${ENV_NAME}/bin/activate"
  cd "${REPO_DIR}"
  ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py

Notes:
  - First launch of Isaac Sim can take 10+ minutes and will ask you to accept the NVIDIA EULA.
  - If you don't need robomimic, you can set INSTALL_LIB=none (or another specific lib) before running this script.
EOF
}

main "$@"
