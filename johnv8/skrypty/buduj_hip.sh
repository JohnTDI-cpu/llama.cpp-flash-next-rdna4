#!/usr/bin/env bash
# Budowa HIP. Uzycie: ./buduj_hip.sh <katalog_drzewa> [liczba_watkow] [arch, domyslnie gfx1201]; ROCM_PATH z env (domyslnie /opt/rocm-7.2.4).
# Watkow mniej niz rdzeni, gdy na maszynie dziala cos jeszcze.
set -e
D=$1; J=${2:-16}; ARCH=${3:-gfx1201}
export ROCM_PATH=${ROCM_PATH:-/opt/rocm-7.2.4}
export PATH="$ROCM_PATH/bin:$PATH"
cd "$D"
[ -f build-hip/CMakeCache.txt ] || cmake -B build-hip -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=$ARCH \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DGGML_CUDA_COMPRESSION_MODE=size
cmake --build build-hip -j "$J"
echo "BUILD-OK $D"
