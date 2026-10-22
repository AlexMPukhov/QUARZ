#!/bin/bash
# Rebuild the QUARZ development environment in a fresh Ubuntu 24.04 container (Claude sessions).
# Installs system packages, builds Kokkos (OpenMP) and openPMD-api (C++: MPI, HDF5, ADIOS2; Python from the pip wheel),
# builds QUARZ and runs the unit tests. Takes ~6 min on 2 cores (tested 2026-10-22: from scratch, quick suite 21/21).
#
# usage: tools/dev/setup_container.sh [PREFIX]      (default PREFIX = /home/claude)
# Every step writes its output to $PREFIX/setup-logs/<step>.log; on failure the script names the step and prints
# the tail of its log (exit code != 0). Re-running is safe: finished builds are skipped (stamp files), an
# interrupted build is resumed. Run it in the foreground or with setsid (a plain nohup job can be killed with
# the shell that started it).
#   options via environment:
#     WITH_EXTRAS=1   also playwright (GUI screenshots),
#                     slurm + sshd on localhost (GUI remote-machine test, section 17 with QUARZ_TEST_REMOTE=1)
#     WITH_CUDA=1     also the CUDA compile check toolchain (pip nvidia-cuda-nvcc, Kokkos 5 with CUDA)
# Afterwards build QUARZ as in CLAUDE.md:
#   cmake -B build -DKokkos_ENABLE_OPENMP=ON -DKokkos_ROOT=$PREFIX/kokkos-install -DopenPMD_ROOT=$PREFIX/openpmd-install
set -eEuo pipefail   # -E: the ERR trap also fires inside functions (run)
PREFIX=${1:-/home/claude}
QUARZ=$(cd "$(dirname "$0")/../.." && pwd)
NJ=${NJ:-2}
KOKKOS_TAG=4.4.01
OPENPMD_TAG=0.17.1
export DEBIAN_FRONTEND=noninteractive
LOGDIR=$PREFIX/setup-logs
mkdir -p "$LOGDIR"
STEP=start
log() { echo "== $(date +%T) $*"; }
# run STEP CMD...: run CMD with output to $LOGDIR/STEP.log
run() { STEP=$1; shift; "$@" >> "$LOGDIR/$STEP.log" 2>&1; }
fail() {
    local rc=$?
    echo "!! setup FAILED in step '$STEP' (exit code $rc); last lines of $LOGDIR/$STEP.log:" >&2
    tail -n 25 "$LOGDIR/$STEP.log" >&2 2>/dev/null || true
    exit $rc
}
trap fail ERR

log "system packages"
rm -f "$LOGDIR"/*.log
# apt-get update: unreachable third-party repositories (e.g. nodesource) only warn; that is fine
run apt apt-get update -q
run apt apt-get install -y -q build-essential cmake git pkg-config \
    openmpi-bin libopenmpi-dev libhdf5-openmpi-dev libadios2-mpi-c++11-dev \
    nlohmann-json3-dev libtoml11-dev python3-dev python3-pip
# vtk: ParaView plugin test in validation section 10 (part of the stored references)
run pip pip install --break-system-packages numpy scipy matplotlib h5py mpi4py openPMD-api vtk

mkdir -p "$PREFIX"
cd "$PREFIX"
# a build counts as done only with its stamp file (an install directory alone may be from an interrupted run)
if [ ! -f "$PREFIX/kokkos-install/.setup-done" ]; then
    log "Kokkos $KOKKOS_TAG (OpenMP + Serial)"
    [ -d kokkos-src/.git ] || { rm -rf kokkos-src; run kokkos git clone -q --depth 1 --branch $KOKKOS_TAG https://github.com/kokkos/kokkos.git kokkos-src; }
    run kokkos cmake -S kokkos-src -B kokkos-build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX/kokkos-install" \
        -DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_SERIAL=ON
    run kokkos cmake --build kokkos-build -j$NJ
    run kokkos cmake --install kokkos-build
    touch "$PREFIX/kokkos-install/.setup-done"
fi
if [ ! -f "$PREFIX/openpmd-install/.setup-done" ]; then
    # C++ library only (MPI, HDF5, ADIOS2); Python uses the pip wheel (installing the module from this
    # build into site-packages would replace the wheel). System json/toml11: FetchContent downloads fail here.
    log "openPMD-api $OPENPMD_TAG (C++: MPI, HDF5, ADIOS2)"
    [ -d openpmd-src/.git ] || { rm -rf openpmd-src; run openpmd git clone -q --depth 1 --branch $OPENPMD_TAG https://github.com/openPMD/openPMD-api.git openpmd-src; }
    run openpmd cmake -S openpmd-src -B openpmd-build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX/openpmd-install" \
        -DopenPMD_USE_MPI=ON -DopenPMD_USE_HDF5=ON -DopenPMD_USE_ADIOS2=ON -DopenPMD_USE_PYTHON=OFF \
        -DopenPMD_USE_INTERNAL_JSON=OFF -DopenPMD_USE_INTERNAL_TOML11=OFF \
        -DBUILD_TESTING=OFF -DopenPMD_BUILD_EXAMPLES=OFF -DopenPMD_BUILD_CLI_TOOLS=ON
    run openpmd cmake --build openpmd-build -j$NJ
    run openpmd cmake --install openpmd-build
    touch "$PREFIX/openpmd-install/.setup-done"
fi

log "QUARZ"
cd "$QUARZ"
run quarz cmake -B build -DKokkos_ENABLE_OPENMP=ON -DKokkos_ROOT="$PREFIX/kokkos-install" \
    -DopenPMD_ROOT="$PREFIX/openpmd-install"
run quarz cmake --build build -j$NJ
STEP=ctest
(cd build && OMP_PROC_BIND=false ctest --output-on-failure > "$LOGDIR/ctest.log" 2>&1)
tail -n 3 "$LOGDIR/ctest.log"

if [ "${WITH_EXTRAS:-0}" = 1 ]; then
    log "extras: playwright, slurm, sshd"
    run extras pip install --break-system-packages playwright
    run extras apt-get install -y -q slurm-wlm munge openssh-server rsync
    # one-node Slurm on this host (no PMI: launch with mpirun inside jobs)
    [ -f /etc/munge/munge.key ] || { dd if=/dev/urandom bs=1 count=1024 of=/etc/munge/munge.key 2>/dev/null; chown munge:munge /etc/munge/munge.key; chmod 400 /etc/munge/munge.key; }
    mkdir -p /run/munge && chown munge:munge /run/munge && (pgrep munged >/dev/null || runuser -u munge -- munged)
    H=$(hostname)
    cat > /etc/slurm/slurm.conf <<EOF
ClusterName=testcluster
SlurmctldHost=$H
ProctrackType=proctrack/linuxproc
TaskPlugin=task/none
SelectType=select/cons_tres
AccountingStorageType=accounting_storage/none
MpiDefault=none
SlurmctldPidFile=/run/slurmctld.pid
SlurmdPidFile=/run/slurmd.pid
StateSaveLocation=/var/spool/slurmctld
SlurmdSpoolDir=/var/spool/slurmd
SlurmUser=root
NodeName=$H CPUs=$(nproc) RealMemory=$(( $(free -m | awk '/Mem:/{print $2}') - 500 )) State=UNKNOWN
PartitionName=debug Nodes=$H Default=YES MaxTime=INFINITE State=UP
EOF
    mkdir -p /var/spool/slurmctld /var/spool/slurmd
    pgrep slurmctld >/dev/null || slurmctld
    pgrep slurmd >/dev/null || slurmd
    mkdir -p ~/.ssh /run/sshd && chmod 700 ~/.ssh
    [ -f ~/.ssh/id_ed25519 ] || ssh-keygen -q -t ed25519 -N "" -f ~/.ssh/id_ed25519
    grep -qf ~/.ssh/id_ed25519.pub ~/.ssh/authorized_keys 2>/dev/null || cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys
    chmod 600 ~/.ssh/authorized_keys
    grep -q "^PermitRootLogin" /etc/ssh/sshd_config || echo "PermitRootLogin prohibit-password" >> /etc/ssh/sshd_config
    ssh-keygen -A >/dev/null; pgrep -x sshd >/dev/null || /usr/sbin/sshd
    ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new localhost true && echo "ssh localhost: OK"
    sinfo
fi

if [ "${WITH_CUDA:-0}" = 1 ]; then
    log "CUDA compile-check toolchain (no GPU here: compile only)"
    run cuda pip install --break-system-packages nvidia-cuda-nvcc nvidia-cuda-runtime nvidia-cuda-cccl
    CU=$(python3 -c 'import nvidia,os;print(os.path.join(os.path.dirname(nvidia.__file__),"cu13"))')
    cd "$PREFIX"
    [ -d kokkos5-src/.git ] || run cuda git clone -q --depth 1 --branch 5.2.2 https://github.com/kokkos/kokkos.git kokkos5-src
    PATH=$CU/bin:$PATH CUDA_HOME=$CU run cuda cmake -S kokkos5-src -B kokkos5-cuda-build -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$PREFIX/kokkos5-cuda-install" -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON \
        -DCMAKE_CXX_COMPILER="$PREFIX/kokkos5-src/bin/nvcc_wrapper"
    PATH=$CU/bin:$PATH CUDA_HOME=$CU run cuda cmake --build kokkos5-cuda-build -j$NJ
    run cuda cmake --install kokkos5-cuda-build
    cd "$QUARZ"
    PATH=$CU/bin:$PATH CUDA_HOME=$CU run cuda cmake -B build-cuda -DKokkos_ROOT="$PREFIX/kokkos5-cuda-install" \
        -DCMAKE_CXX_COMPILER="$PREFIX/kokkos5-src/bin/nvcc_wrapper" -DopenPMD_ROOT="$PREFIX/openpmd-install"
    echo "CUDA compile check: PATH=$CU/bin:\$PATH CUDA_HOME=$CU make -C build-cuda -j$NJ quarz"
fi
log "done"
