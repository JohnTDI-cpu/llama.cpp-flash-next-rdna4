#!/usr/bin/env bash
# Bit-exact przez llama-server /completion (temp 0, ten sam prompt) dla dwoch buildow.
# Uzycie: bitexact.sh <binA> <binB> <etykieta> [ZMIENNE=... dla B]
set -u; A=$1; B=$2; E=$3; shift 3
M=${MODEL:-$HOME/models/Qwen3.8-Flash-Next-UD-IQ3_XXS/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf}
ROCM=${ROCM_PATH:-/opt/rocm-7.2.4}; DEV=${DEV:-ROCm1}; CARD=${CARD:-card1}
OT=${OT:-'per_layer_token_embd=CPU,blk\.(2[3-9]|3[0-9]|4[0-7])\.ffn_.*_exps=CPU'}; S=${SCRATCH:-/tmp/bitexact}; mkdir -p "$S"
PK="Silnik 2.0 TDI w Passacie B6: opisz typowe usterki, po kolei, rzeczowo."
PD="$(python3 -c "print(('Oferta: Skoda Octavia III 1.6 TDI, rok 2016, przebieg 187 tys. km, cena 38 900 zl. ' * 60) + 'Ktora z tych ofert jest najtansza? Odpowiedz jednym zdaniem.')")"
gen() {
  local BIN=$1 TAG=$2; shift 2
  env "$@" LD_LIBRARY_PATH=$BIN:$ROCM/lib ROCM_PATH=$ROCM GGML_CUDA_DISABLE_GRAPHS=1 "$BIN/llama-server" -m "$M" --host 127.0.0.1 --port 8097 --device $DEV -ngl 99 --fit off -ot "$OT" -c 8192 -np 1 --cache-type-k q8_0 --cache-type-v q8_0 -fa on -t 48 --no-warmup > /tmp/bitexact_$TAG.log 2>&1 &
  local SV=$!; for i in $(seq 1 200); do curl -sf http://127.0.0.1:8097/health >/dev/null 2>&1 && break; kill -0 $SV 2>/dev/null || { echo "  $TAG: serwer padl"; return 1; }; sleep 2; done
  for W in k d; do local P; [ $W = k ] && P="$PK" || P="$PD"
    python3 - "$P" "$S/gen_${TAG}_$W.txt" <<'PY'
import sys,json,urllib.request
p,out=sys.argv[1],sys.argv[2]
r=urllib.request.Request("http://127.0.0.1:8097/completion",json.dumps({"prompt":p,"n_predict":96 if len(p)<500 else 48,"temperature":0,"cache_prompt":False,"seed":1}).encode(),{"Content-Type":"application/json"})
d=json.loads(urllib.request.urlopen(r,timeout=600).read()); open(out,"w").write(d["content"]); print(f"    {out.split('/')[-1]}: {d['tokens_predicted']} tok, {d['timings']['predicted_per_second']:.1f} t/s")
PY
  done
  kill -9 $SV 2>/dev/null; for i in $(seq 1 60); do U=$(rocm-smi --showmeminfo vram --json 2>/dev/null | python3 -c "import sys,json;print(int(json.load(sys.stdin)['$CARD']['VRAM Total Used Memory (B)'])//2**20)" 2>/dev/null); [ "${U:-9999}" -lt 600 ] && break; sleep 2; done
}
echo "=== $E: A=$A ==="; gen "$A" ${E}_A; echo "=== $E: B=$B $* ==="; gen "$B" ${E}_B "$@"
for W in k d; do for X in A B; do [ -s "$S/gen_${E}_${X}_$W.txt" ] || echo "  [$W] PADL: pusta odpowiedz $X (serwer padl?)"; done; done
for W in k d; do cmp -s "$S/gen_${E}_A_$W.txt" "$S/gen_${E}_B_$W.txt" && echo "  [$W] IDENTYCZNE" || { echo "  [$W] ROZNE:"; diff <(fold -w 100 "$S/gen_${E}_A_$W.txt") <(fold -w 100 "$S/gen_${E}_B_$W.txt") | head -6; }; done
