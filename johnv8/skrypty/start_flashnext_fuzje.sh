#!/usr/bin/env bash
# Qwen3.8-Flash-Next UD-IQ3_XXS na jednej karcie + CPU — konfiguracja hostingu (2026-09-03).
# Zmienne: BIN, MODEL, DRAFT, DEV (ROCm0/ROCm1), ROCM_PATH, PORT, CTX, NP, CPU_OD (pierwsza warstwa ekspertow na CPU), WATKI, GRAFY=0.
# PR 28243 (HIP) + glowa MTP unsloth shared Q8, n-max 3, 48 watkow, 25 warstw ekspertow na CPU.
# tg128: 44,4 proza / 45,8 kod przy -c 65536, 2 sloty (26 warstw na CPU). Zmienne: PORT, CTX, NP, CPU_OD (pierwsza warstwa na CPU).
set -u
# [fuzje] drzewo z fuzjami bit-exact (E22/E31): +15% proza, +18% kod z MTP vs PR+latki
B=${BIN:-$(cd "$(dirname "$0")/../.." && pwd)/build-hip/bin}   # binarki z build-hip tego repo
M=${MODEL:-$HOME/models/Qwen3.8-Flash-Next-UD-IQ3_XXS/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf}
MD=${DRAFT:-$HOME/models/Flash-Next-mtp/unsloth/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf}
DEV=${DEV:-ROCm1}; ROCM=${ROCM_PATH:-/opt/rocm-7.2.4}
PORT=${PORT:-8092}; CTX=${CTX:-32768}; NP=${NP:-2}; CPU_OD=${CPU_OD:-22}
WARSTWY=$(seq -s'|' "$CPU_OD" 47)
export ROCM_PATH=$ROCM LD_LIBRARY_PATH="$B:$ROCM/lib"
# [E46] HIP graphs: +2-3%; GRAFY=0 wylacza
[ "${GRAFY:-1}" = "1" ] || export GGML_CUDA_DISABLE_GRAPHS=1
# [E40/E43] watki backendu CPU spinuja zamiast spac: +2% z MTP przy 32 watkach
export OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-active}
# [E45] E10c v3b jawnie ON (niezaleznie od domyslnej wartosci w binarce)
export GGML_JOHNV8_HCBLOCK=${GGML_JOHNV8_HCBLOCK:-1}
export GGML_JOHNV8_MMQ_ID_JMAX=${GGML_JOHNV8_MMQ_ID_JMAX:-32}   # E57/E59: cap kafla J w MMQ mul_mat_id, bit-exact, pp2048 +8-9% (1xGPU)
ss -ltn 2>/dev/null | grep -q "127.0.0.1:$PORT" && { echo "PRZERWANE: port $PORT zajety"; ss -ltnp | grep ":$PORT"; exit 1; }
exec "$B/llama-server" -m "$M" --host 127.0.0.1 --port "$PORT" --alias flash-next-gpu1-fuzje \
  --device $DEV -ngl 99 --fit off \
  -c "$CTX" -np "$NP" --kv-unified --cache-type-k q8_0 --cache-type-v q8_0 \
  -ot "per_layer_token_embd=CPU,blk\.($WARSTWY)\.ffn_.*_exps=CPU" \
  -md "$MD" --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0 -devd $DEV --spec-draft-ngl 99 \
  --flash-attn on -ub 512 -b 1536 -t ${WATKI:-32} -tb 48 --no-warmup --jinja --reasoning-format deepseek
