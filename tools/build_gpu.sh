#!/bin/bash
# ---------------------------------------------------------------------------
# One-command GPU build of QUARZ (NVIDIA), no root rights needed.
#
#   cd quarz && tools/build_gpu.sh            # build into ./build-gpu, run the unit tests
#
# What it does
#   1. Finds a CUDA compiler: `nvcc` on the PATH / in $CUDA_HOME (e.g. after
#      `module load cuda` on a cluster).  If there is none, it installs NVIDIA's
#      official CUDA 13 compiler wheels from PyPI into ./gpu-deps (pip, user space).
#   2. Detects the GPU architecture with nvidia-smi (override: QUARZ_ARCH=AMPERE80 ...).
#   3. Builds Kokkos with the CUDA backend.  Kokkos' "nvcc_wrapper" is NOT a separate
#      compiler: it is a small script shipped inside Kokkos that calls the normal nvcc.
#   4. Builds QUARZ and runs ctest (and a short GPU run) if a GPU is present.
#
# Environment overrides
#   QUARZ_ARCH=AMPERE80|HOPPER90|ADA89|...   Kokkos architecture name (default: from nvidia-smi)
#   QUARZ_PREFIX=dir                           where CUDA wheels / Kokkos go (default ./gpu-deps)
#   QUARZ_COMPILE_ONLY=1                       machine without GPU/driver: only compile & link
#   QUARZ_JOBS=n                               parallel build jobs (default: nproc)
#   QUARZ_OPENPMD_ROOT=dir                     installed openPMD-api (enables openPMD output)
#   QUARZ_MPI=1|0                              MPI (several GPUs, decomposition along xi).
#                                             Default: on if an MPI installation (mpicxx) is found.
#                                             Any MPI works (CUDA-aware MPI is not needed).
# ---------------------------------------------------------------------------
set -euo pipefail
# old variable names (before the rename QSRZ -> QUARZ) are still accepted
for v in ARCH PREFIX COMPILE_ONLY JOBS OPENPMD_ROOT MPI; do
    o="QSRZ_$v"; n="QUARZ_$v"
    if [ -n "${!o:-}" ] && [ -z "${!n:-}" ]; then export "$n=${!o}"; fi
done
SRC=$(cd "$(dirname "$0")/.." && pwd)
PREFIX=${QUARZ_PREFIX:-$SRC/gpu-deps}
JOBS=${QUARZ_JOBS:-$(nproc)}
mkdir -p "$PREFIX"
say() { echo -e "\n=== $*"; }

# ---- 1. GPU and driver --------------------------------------------------------
HAVE_GPU=0
DRV_MAJOR=0
if [ "${QUARZ_COMPILE_ONLY:-0}" != "1" ] && command -v nvidia-smi > /dev/null 2>&1; then
    HAVE_GPU=1
    DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    DRV_MAJOR=${DRV%%.*}
    CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
    say "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), compute capability $CC, driver $DRV"
fi

# ---- 2. CUDA compiler ------------------------------------------------------------
NVCC=""
if [ -n "${CUDA_HOME:-}" ] && [ -x "$CUDA_HOME/bin/nvcc" ]; then NVCC="$CUDA_HOME/bin/nvcc"; fi
if [ -z "$NVCC" ] && command -v nvcc > /dev/null 2>&1; then NVCC=$(command -v nvcc); fi
if [ -z "$NVCC" ] && [ -x /usr/local/cuda/bin/nvcc ]; then NVCC=/usr/local/cuda/bin/nvcc; fi

if [ -z "$NVCC" ]; then
    if [ "$HAVE_GPU" = "1" ] && [ "$DRV_MAJOR" -lt 580 ]; then
        cat << EOF

No nvcc found, and your driver ($DRV) is older than 580, which the CUDA 13
wheels from PyPI need.  Please get a CUDA 12 toolkit instead, e.g. one of
  * on a cluster:      module avail cuda ; module load cuda/12.x
  * with conda/mamba:  conda install -c conda-forge cuda-nvcc=12.4 cuda-cudart-dev=12.4 cuda-cccl=12.4
  * NVIDIA runfile:    sh cuda_12.4.1_*_linux.run --silent --toolkit --toolkitpath=\$HOME/cuda-12.4
then export CUDA_HOME=<that directory> and rerun this script.
EOF
        exit 1
    fi
    say "No nvcc found: installing NVIDIA's CUDA 13 compiler wheels from PyPI into $PREFIX/cuda-pip"
    python3 -m pip install --quiet --target "$PREFIX/cuda-pip" \
        "nvidia-cuda-nvcc==13.*" "nvidia-cuda-runtime==13.*" "nvidia-cuda-cccl==13.*" "nvidia-cuda-crt==13.*"
    export CUDA_HOME="$PREFIX/cuda-pip/nvidia/cu13"
    ln -sf libcudart.so.13 "$CUDA_HOME/lib/libcudart.so"
    NVCC="$CUDA_HOME/bin/nvcc"
fi
export CUDA_HOME=${CUDA_HOME:-$(cd "$(dirname "$NVCC")/.." && pwd)}
export PATH="$(dirname "$NVCC"):$PATH"
CUDA_VER=$("$NVCC" --version | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1.\2/p')
CUDA_MAJOR=${CUDA_VER%%.*}
say "Using nvcc $NVCC (CUDA $CUDA_VER)"

# driver library: needed for linking.  On a machine without a driver (compile-only
# test) an empty stub is created - the executable then cannot run, only link.
if ! ldconfig -p 2> /dev/null | grep -q "libcuda.so " && [ ! -e "$CUDA_HOME/lib/stubs/libcuda.so" ] \
    && [ ! -e "$CUDA_HOME/lib64/stubs/libcuda.so" ]; then
    if [ "$HAVE_GPU" = "1" ]; then
        echo "WARNING: libcuda.so (NVIDIA driver library) not found by ldconfig"
    else
        say "No NVIDIA driver: creating a link stub for libcuda.so (compile-only build)"
        mkdir -p "$CUDA_HOME/lib/stubs"
        STUB_SYMS="cuGetErrorString cuGetErrorName cuCtxGetDevice cuCtxPushCurrent_v2 cuCtxPopCurrent_v2 cuStreamGetCtx cuDeviceGet cuDeviceGetAttribute cuInit cuDriverGetVersion cuPointerGetAttribute cuMemGetAddressRange_v2"
        { for s in $STUB_SYMS; do echo "int $s(void){return 999;}"; done; } > "$PREFIX/cuda_stub.c"
        gcc -shared -fPIC -o "$CUDA_HOME/lib/stubs/libcuda.so" "$PREFIX/cuda_stub.c"
    fi
fi

# ---- 3. architecture ------------------------------------------------------------
if [ -z "${QUARZ_ARCH:-}" ]; then
    case "${CC:-}" in
        7.0) QUARZ_ARCH=VOLTA70 ;;   7.2) QUARZ_ARCH=VOLTA72 ;;   7.5) QUARZ_ARCH=TURING75 ;;
        8.0) QUARZ_ARCH=AMPERE80 ;;  8.6) QUARZ_ARCH=AMPERE86 ;;  8.7) QUARZ_ARCH=AMPERE87 ;;
        8.9) QUARZ_ARCH=ADA89 ;;     9.0) QUARZ_ARCH=HOPPER90 ;;
        10.0) QUARZ_ARCH=BLACKWELL100 ;; 12.0) QUARZ_ARCH=BLACKWELL120 ;;
        *) QUARZ_ARCH=AMPERE80; echo "WARNING: GPU architecture unknown (${CC:-no GPU}); using AMPERE80. Set QUARZ_ARCH." ;;
    esac
fi
if [ "$CUDA_MAJOR" -ge 13 ]; then KOKKOS_TAG=5.2.2; else KOKKOS_TAG=4.4.01; fi
say "Kokkos $KOKKOS_TAG, architecture $QUARZ_ARCH"

# ---- 4. Kokkos ---------------------------------------------------------------------
KSRC="$PREFIX/kokkos-$KOKKOS_TAG-src"
KINST="$PREFIX/kokkos-$KOKKOS_TAG-cuda-$QUARZ_ARCH"
if [ ! -f "$KINST/lib/cmake/Kokkos/KokkosConfig.cmake" ] && [ ! -f "$KINST/lib64/cmake/Kokkos/KokkosConfig.cmake" ]; then
    [ -d "$KSRC" ] || git -c advice.detachedHead=false clone -q --depth 1 --branch "$KOKKOS_TAG" https://github.com/kokkos/kokkos.git "$KSRC"
    cmake -S "$KSRC" -B "$PREFIX/kokkos-build-$QUARZ_ARCH" -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER="$KSRC/bin/nvcc_wrapper" \
        -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_SERIAL=ON -DKokkos_ARCH_${QUARZ_ARCH}=ON \
        -DCUDAToolkit_ROOT="$CUDA_HOME" -DCMAKE_INSTALL_PREFIX="$KINST" > "$PREFIX/kokkos-configure.log" 2>&1 \
        || { tail -30 "$PREFIX/kokkos-configure.log"; exit 1; }
    say "Building Kokkos (a few minutes)"
    cmake --build "$PREFIX/kokkos-build-$QUARZ_ARCH" -j "$JOBS" --target install > "$PREFIX/kokkos-build.log" 2>&1 \
        || { tail -30 "$PREFIX/kokkos-build.log"; exit 1; }
fi

# ---- 5. QUARZ -----------------------------------------------------------------------
if [ -z "${QUARZ_MPI:-}" ]; then
    if command -v mpicxx > /dev/null 2>&1; then QUARZ_MPI=1; else QUARZ_MPI=0; fi
fi
if [ "$QUARZ_MPI" = "1" ]; then MPI_OPT=ON; else MPI_OPT=OFF; fi
say "Building QUARZ in $SRC/build-gpu (MPI: $MPI_OPT)"
OPMD_OPT=""
[ -n "${QUARZ_OPENPMD_ROOT:-}" ] && OPMD_OPT="-DopenPMD_ROOT=$QUARZ_OPENPMD_ROOT"
cmake -S "$SRC" -B "$SRC/build-gpu" -DCMAKE_BUILD_TYPE=Release -DQUARZ_FETCH_KOKKOS=OFF -DQUARZ_ENABLE_MPI=$MPI_OPT $OPMD_OPT \
    -DKokkos_ROOT="$KINST" -DCMAKE_CXX_COMPILER="$KSRC/bin/nvcc_wrapper" > "$SRC/build-gpu-configure.log" 2>&1 \
    || { tail -30 "$SRC/build-gpu-configure.log"; exit 1; }
grep -q "QUARZ: MPI enabled" "$SRC/build-gpu-configure.log" && echo "MPI enabled" || echo "MPI not used (serial build)"
cmake --build "$SRC/build-gpu" -j "$JOBS" 2>&1 | grep -E "error|Built target" || true
[ -x "$SRC/build-gpu/quarz" ] || { echo "QUARZ build failed"; exit 1; }

if [ "$HAVE_GPU" = "1" ]; then
    say "Unit tests on the GPU"
    (cd "$SRC/build-gpu" && ctest --output-on-failure)
    say "Short GPU run (blowout, one quasi-static solve)"
    (cd "$SRC/validation" && "$SRC/build-gpu/quarz" blowout.in xi.min=-10 output.dir=out_gpu_check | grep -E "execution space|tridiagonal|sweep")
    echo "Compare with the CPU numbers: cd validation && python3 bsum.py out_gpu_check gpu"
else
    say "Compile-only build finished (no GPU here): $SRC/build-gpu/quarz"
fi
echo -e "\nTo run:  $SRC/build-gpu/quarz input.in      (CUDA_HOME=$CUDA_HOME)"
if [ "$MPI_OPT" = "ON" ]; then
    echo "On P GPUs: mpirun -np P $SRC/build-gpu/quarz input.in   (one GPU per rank, chosen automatically)"
fi
