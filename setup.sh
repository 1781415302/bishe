#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)}"

FUNCTIONALPLUS_TAG="v0.2.14-p0"
FUNCTIONALPLUS_COMMIT="aa94989c43bd6680479b3c1cb5f63972d8380d61"
EIGEN_TAG="3.3.9"
EIGEN_COMMIT="0fd6b4f71dd85b2009ee4d1aeb296e2c11fc9d68"
JSON_TAG="v3.7.3"
JSON_COMMIT="e7b3b40b5a95bc74b9a7f662830a27c49ffc01b4"
FRUGALLY_DEEP_TAG="v0.15.17-p0"
FRUGALLY_DEEP_COMMIT="9159a4bdf25df607c3af188884fed2480142eea2"

log() {
    printf '[setup] %s\n' "$*"
}

need_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        printf '[setup] missing required command: %s\n' "$1" >&2
        exit 1
    fi
}

install_apt_prereqs() {
    if [ "${SKIP_APT:-0}" = "1" ]; then
        log "SKIP_APT=1, not installing apt packages"
        return
    fi
    if ! command -v apt-get >/dev/null 2>&1; then
        log "apt-get not found; install system packages manually, then rerun setup"
        return
    fi

    sudo apt-get update
    sudo apt-get install -y \
        build-essential \
        ccache \
        cmake \
        g++ \
        gawk \
        git \
        make \
        wget \
        libtool \
        libxml2-dev \
        libxslt1-dev \
        autoconf \
        texinfo \
        zip \
        genromfs \
        flex \
        bison \
        libncurses5-dev \
        zlib1g-dev \
        xterm \
        python2.7 \
        python-dev \
        python-pip \
        python-setuptools \
        python-future \
        python-lxml \
        python-matplotlib \
        python-serial \
        python-numpy \
        python-scipy \
        python3 \
        python3-pip \
        python3-setuptools \
        python3-venv

    if apt-cache show python-wxgtk3.0 >/dev/null 2>&1; then
        sudo apt-get install -y python-wxgtk3.0
    elif apt-cache show python-wxgtk2.8 >/dev/null 2>&1; then
        sudo apt-get install -y python-wxgtk2.8
    fi
}

install_python_prereqs() {
    if [ "${SKIP_PYTHON:-0}" = "1" ]; then
        log "SKIP_PYTHON=1, not installing Python packages"
        return
    fi

    if command -v pip2 >/dev/null 2>&1; then
        pip2 install --user -U future lxml pymavlink MAVProxy
    else
        log "pip2 not found; sim_vehicle.py in this old ArduPilot tree expects Python 2.7"
    fi

    if command -v python3 >/dev/null 2>&1; then
        python3 -m pip install --user -U pip setuptools wheel
        if [ -f "$ROOT_DIR/simulator/Tools/autotest/requirements.txt" ]; then
            python3 -m pip install --user -r "$ROOT_DIR/simulator/Tools/autotest/requirements.txt"
        else
            python3 -m pip install --user 'pymavlink>=2.4.12'
        fi
    fi
}

ensure_checkout() {
    local name="$1"
    local url="$2"
    local tag="$3"
    local commit="$4"
    local dir="$ROOT_DIR/$name"

    if [ -d "$dir/.git" ]; then
        log "updating $name"
        if [ "${FORCE_DEP_CHECKOUT:-0}" != "1" ]; then
            if ! git -C "$dir" diff --quiet || ! git -C "$dir" diff --cached --quiet; then
                printf '[setup] %s has local changes. Commit/stash them or rerun with FORCE_DEP_CHECKOUT=1.\n' "$name" >&2
                exit 1
            fi
        fi
        git -C "$dir" fetch --tags origin
    elif [ -e "$dir" ]; then
        printf '[setup] %s exists but is not a git checkout. Move it away and rerun setup.\n' "$dir" >&2
        exit 1
    else
        log "cloning $name ($tag)"
        git clone "$url" "$dir"
        git -C "$dir" fetch --tags origin
    fi

    if [ "${FORCE_DEP_CHECKOUT:-0}" = "1" ]; then
        git -C "$dir" reset --hard
    fi
    git -C "$dir" checkout "$commit"

    local actual
    actual="$(git -C "$dir" rev-parse HEAD)"
    if [ "$actual" != "$commit" ]; then
        printf '[setup] %s checkout mismatch: expected %s, got %s\n' "$name" "$commit" "$actual" >&2
        exit 1
    fi
}

cmake_install() {
    local dir="$1"
    shift || true

    log "building $(basename "$dir")"
    mkdir -p "$dir/build"
    (
        cd "$dir/build"
        cmake "$@" ..
        make -j"$JOBS"
        sudo make install
    )
}

install_cpp_dependencies() {
    ensure_checkout "FunctionalPlus" "https://github.com/Dobiasd/FunctionalPlus.git" "$FUNCTIONALPLUS_TAG" "$FUNCTIONALPLUS_COMMIT"
    cmake_install "$ROOT_DIR/FunctionalPlus"

    ensure_checkout "eigen" "https://gitlab.com/libeigen/eigen.git" "$EIGEN_TAG" "$EIGEN_COMMIT"
    cmake_install "$ROOT_DIR/eigen"
    sudo ln -sfn /usr/local/include/eigen3/Eigen /usr/local/include/Eigen

    ensure_checkout "json" "https://github.com/nlohmann/json.git" "$JSON_TAG" "$JSON_COMMIT"
    cmake_install "$ROOT_DIR/json" -DBUILD_TESTING=OFF -DJSON_BuildTests=OFF

    ensure_checkout "frugally-deep" "https://github.com/Dobiasd/frugally-deep.git" "$FRUGALLY_DEEP_TAG" "$FRUGALLY_DEEP_COMMIT"
    cmake_install "$ROOT_DIR/frugally-deep"

    if command -v ldconfig >/dev/null 2>&1; then
        sudo ldconfig
    fi
}

print_next_steps() {
    cat <<'EOF'

Setup finished.

Build the SITL simulator:
  cd simulator
  ./waf configure --board sitl
  ./waf copter

Run a manual copter simulation:
  cd ArduCopter
  python2.7 ../Tools/autotest/sim_vehicle.py --wipe-eeprom --console --map

Run a small mode-comparison smoke test from the repository root:
  python3 simulator/Tools/autotest/run_mode_experiments.py \
    --modes 0,2,4 --missions mission-1.txt --attack-combos 0:0,4:0 --seeds 0 \
    --max-runs 6 --force-cleanup --isolated-sitl-state
EOF
}

main() {
    install_apt_prereqs
    need_cmd git
    need_cmd make
    need_cmd cmake

    install_python_prereqs
    install_cpp_dependencies
    print_next_steps
}

main "$@"
