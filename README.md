# llama.cpp fork — Qwen3.8-Flash-Next (qwen4exp) on AMD RDNA4: bit-exact kernel fusions, MTP, MoE prefill cap

## About this work: a capability test of Claude Fable 5.1

Everything in this fork beyond upstream llama.cpp and PR #28243 — the profiling, the kernel fusions, the MMQ tile cap, the experiments (E0–E66), the scripts, the benchmarks and this documentation — was developed autonomously by **Claude Fable 5.1** (Anthropic) running in Claude Code on the author's machine, on 2026-09-02/03. The author (JohnTDI-cpu) set the task as a test of what the model can do; by his account an earlier attempt at a comparable task with Claude Opus 5 did not succeed.

**The task, as given:** profile the decode path of **Qwen3.8-Flash-Next (`qwen4exp`, UD-IQ3_XXS by unsloth)** on the HIP/ROCm backend and, based on the profile, improve decode throughput *significantly* — and prefill if possible — for two setups: **1× RDNA4 GPU + CPU** (one 32 GB Radeon AI PRO R9700 with part of the MoE experts on the CPU) and **2× RDNA4 GPU + CPU** (both R9700s, only the engram table on the CPU). Hard constraints: no loss of quality (every enabled change must be bit-exact with the stock kernels) and the model had to work from what was on the machine.

**What the model had to work with (the author's private material on this machine):**
- `~/llama-johnv8` — the author's earlier fork (`johnv8: dekodowanie qwen4exp na RDNA4 (+14%)`) with the hyper-connection fusion kernels (`hc-combine`, `hc-mix`) that were ported here as the first fusion commit, and the fork's branches with the RDNA4 mmvq patches (small-K path, nwarps=2), the HIP SWAR patch and the MTP port (`johnv8-mtp`, from PR #27739);
- the author's working folder with the previous rounds of measurements and notes: a max-speed study of Flash-Next with MTP on 2× R9700, quality-evaluation runs of the model (Open PL LLM Leaderboard, 9101 samples), earlier exploratory experiment chains, and the benchmark (`bench_mtp.py`) and bit-exact (`bitexact.sh`) harnesses;
- the model files (UD-IQ3_XXS shards and the MTP draft heads), the stock unsloth Vulkan build used as the baseline, and the ROCm 7.2.4 installation (its headers and the CLR sources were read to understand AQL barrier and graph-capture semantics).

**How it was done:** profile first (rocprof kernel census per token: 4275 kernels/token at the start), then one change at a time, each behind an env switch, each gated by identical-token generation at temperature 0 on two prompts (short and ~1500 tokens) before being enabled by default; A/B timing with repeats and a re-run baseline in every session; 2-GPU sessions only when both cards were free. Several directions were tried and closed with data rather than kept (multi-stream fork/join, HW-queue count, a fusion-plan cache, dense attention for short contexts), one race introduced on the way (E64C) was caught by the bit-exact gate and fixed (E64D).

**Outcome:** decode +10 % (1 GPU) / +17 % (2 GPUs) over the starting point with identical tokens, +22–25 % over the stock unsloth Vulkan build on 1 GPU, prefill +8 % / +19 % from the MMQ tile cap (2.3× the Vulkan build on 1 GPU, above it on 2 GPUs); on 2 GPUs without MTP the Vulkan build still decodes faster. Whether this passes the author's exam is his call: the condition he set was that the model keeps its quality, and every enabled change was verified to produce exactly the same tokens as the stock kernels.

**Notes from the model, for whoever reads this later:** the numbers are from one machine and one ROCm version (7.2.4); the barrier and graph-capture findings are specific to that runtime, and thermal drift of 2–4 % within a session is real, so always re-run the baseline. The single most useful habit was refusing any change that failed the bit-exact gate, even when it looked faster — it is what makes the speed numbers comparable and the model trustworthy.

---

This is a working fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) built on top of **PR #28243** (Qwen3.8-Flash-Next / `qwen4exp` support, commit `2857e511`) with a series of HIP-backend changes for AMD RDNA4 (gfx1201, Radeon AI PRO R9700). Every change that is enabled by default is **bit-exact** with the stock kernels: at temperature 0 the model produces identical tokens on a short and a ~1500-token prompt (gate script in `johnv8/skrypty/bitexact.sh`), so quality is exactly that of the upstream PR.

Measured with **Qwen3.8-Flash-Next UD-IQ3_XXS** (unsloth, 82 GB) + the unsloth MTP draft head (`mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`), context 65536 × 2 slots, ubatch 512, MTP n-max 3, 32 CPU threads, `OMP_WAIT_POLICY=active`, ROCm 7.2.4, EPYC 7543, 125 GB RAM.

## Results vs the stock unsloth build (b10715, Vulkan) — same model, same settings

### 1× GPU (32 GB) + CPU (22 expert layers on GPU, 26 on CPU, KV q8_0)

| build | MTP | pp2048 t/s | decode prose | decode code |
|---|---|---|---|---|
| **this fork (HIP + fusions + HIP graphs + MMQ J cap)** | **yes** | ~416 | **50.5** | **53.6** |
| this fork | no | **418** | 36.0 | 36.0 |
| unsloth b10715 (Vulkan) | yes | 176 | 40.4 | 44.0 |
| unsloth b10715 (Vulkan) | no | 184 | 29.6 | 28.9 |

### 2× GPU (2 × 32 GB, all experts on GPU, `-ts 1,1.2`, KV f16)

| build | MTP | pp2048 t/s | decode prose | decode code |
|---|---|---|---|---|
| **this fork** | **yes** | **820** | **62–65** | **66–70** |
| this fork | no | **942** | 39.7–40.1 | 39.6–40.1 |
| this fork + any-order launches (experimental, off by default) | no | 942 | 41.5 | 41.4 |
| unsloth b10715 (Vulkan) | yes | 690 | 49.9 | 59.5 |
| unsloth b10715 (Vulkan) | no | 894 | 45.6 | 45.5 |

### Where the gains come from (vs the same PR without these changes)

| step | 1× GPU decode (MTP) | 2× GPU decode (MTP) | prefill pp2048 |
|---|---|---|---|
| PR #28243 + 3 RDNA4 patches (starting point) | 45.9 / 48.7 | 53.5 / 56.8 | 1×: 386, 2×: 794 |
| + bit-exact kernel fusions + OpenMP active | 49.7 / 52.9 | 60.2 / 63.9 | — |
| + HIP graphs | 50.5 / 53.6 (+10 %) | 62.8 / 67.0 (+17 %) | — |
| + MMQ column-tile cap for MoE (`GGML_JOHNV8_MMQ_ID_JMAX=32`) | — | — | **1×: 418 (+8 %), 2×: 942 (+19 %)** |

Experiment log (E0–E66, every measurement and verdict): `johnv8/docs/WYNIK_OPTYMALIZACJI.md` (Polish). English summary table: `johnv8/docs/BENCH_SUMMARY_EN.md`.

## What is in the tree

Commits on top of `2857e511` (34 commits, also exported as `johnv8/patches/seria-fuzje/`):

1. RDNA4 mmvq patches (small-K path, nwarps=2 for 2–8 column batches) and HIP SWAR `__vsub4/__vcmpne4` for IQ2/IQ3.
2. Hyper-connection glue fusions (`hc-combine`, `hc-mix`, `scale_silu`), the shared-expert tail (E6b), Q8_1 activation-quantization dedup for mmvq (E6d), GDN prolog and l2norm inside the `gated_delta_net` kernel (E7/E7b), `hc_inject` folded into the combine kernel (E10a), the hyper-connection block as three kernels (E10c v3b), CPU eval-loop timing, an (off) fusion-plan cache (E11).
3. MMQ column-tile cap for `mul_mat_id` (E55/E57): the MoE GEMM picked a 128-column tile from the 512-token ubatch although each expert gets ~10 rows; capping the choice at 32 removes the wasted work with identical accumulation order.

Every fusion has an env switch `GGML_JOHNV8_*` (see `johnv8/docs/README_FUZJE_FINAL.md`); the kernel files are `ggml/src/ggml-cuda/hc-block.cu`, `hc-combine.cu`, `hc-mix.cu`, `shexp-tail.cu`, plus hooks in `ggml-cuda.cu`, `mmvq.cu`, `mmq.cuh`, `gated_delta_net.cu`, `common.cuh` and `src/models/qwen4exp.cpp`.

`johnv8/` folder:

| path | content |
|---|---|
| `docs/README_ODTWORZENIE.md` | step-by-step reproduction on another AMD box (Polish): build, models, run 1×/2× GPU, benchmark method, baseline |
| `docs/WYNIK_OPTYMALIZACJI.md`, `docs/README_FUZJE_FINAL.md`, `docs/BENCH_SUMMARY_EN.md` | measurements, verdicts, lessons |
| `docs/PLAN_ANYORDER.md` | design of hazard-tracked any-order launches (barrier-bit elision on ROCm) |
| `skrypty/` | `buduj_hip.sh` (cmake flags), `start_flashnext_fuzje.sh` (server launcher), `bench_mtp.py` (pp/tg benchmark via llama-server), `bitexact.sh` (bit-exact gate), `spis_jader.py` (kernel census from rocprof); paths via env (`MODEL`, `DRAFT`, `BIN`, `DEV`, `ROCM_PATH`) |
| `patches/` | `seria-fuzje/` (this history as patches), `e64_anyorder.patch` (experimental any-order, off by default), `e55_forkjoin.patch` (multi-stream fork/join — closed, −40 % on ROCm 7.2.4), `e55_jcap.patch` |
| `mikrotesty/` | HIP microbenchmarks: any-order vs barrier launches, graph capture, event/copy semantics, launch cost |

## Build and run (short)

```bash
cmake -B build-hip -S . -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 \
      -DGGML_CUDA_FA=ON -DGGML_CUDA_GRAPHS=ON -DGGML_CUDA_COMPRESSION_MODE=size
cmake --build build-hip -j 16

export OMP_WAIT_POLICY=active GGML_JOHNV8_HCBLOCK=1 GGML_JOHNV8_MMQ_ID_JMAX=32
CPU_OD=22; LAYERS=$(seq -s'|' $CPU_OD 47)      # expert layers 22..47 on CPU (32 GB card); all on GPU: drop the blk part
build-hip/bin/llama-server -m Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf --device ROCm0 -ngl 99 --fit off \
  -c 65536 -np 2 --kv-unified --cache-type-k q8_0 --cache-type-v q8_0 \
  -ot "per_layer_token_embd=CPU,blk\.($LAYERS)\.ffn_.*_exps=CPU" \
  -md mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0 --spec-draft-ngl 99 \
  --flash-attn on -ub 512 -b 1536 -t 32 -tb 48 --jinja --reasoning-format deepseek
```

Two GPUs: `--device ROCm0,ROCm1 -ts 1,1.2 -ot per_layer_token_embd=CPU --cache-type-k f16 --cache-type-v f16`. Details, sizing rules (`CPU_OD` vs VRAM), the benchmark procedure and the baseline setup: `johnv8/docs/README_ODTWORZENIE.md`.

Other architectures: change `AMDGPU_TARGETS`; the fusions were verified bit-exact on gfx1201 only, so run `johnv8/skrypty/bitexact.sh` first (note: on HIP `__fmul_rn/__fadd_rn` do not match `a*b`/`a+b` bit-for-bit, which is why the replicas use plain operators under `#pragma clang fp contract(off)`).

## Things that were tried and do not help on ROCm 7.2.4 / RDNA4

- Multi-stream fork/join (`GGML_CUDA_GRAPH_OPT`-style, generalised for qwen4exp): correct, but every variant with real forks loses 40–50 % decode — cross-stream HIP events cost far more than the ~5 µs AQL barrier they replace.
- `GPU_MAX_HW_QUEUES=8`: within noise.
- Any-order launches with a hazard tracker (`johnv8/patches/e64_anyorder.patch`, `GGML_JOHNV8_ANYORDER=1`): bit-exact, 15 % of launches barrier-free, +2–4 % decode on 2 GPUs without MTP, +1 % with MTP, 0 on 1 GPU; the remaining barriers are real dependency chains, and the flag is dropped by HIP graph capture, so it is used only for graphs of ≥ 400 nodes.

## Branches

- `master` — this work (PR #28243 base + fusions + docs). The previous `master` is kept as `archiwum-master-2026-09-03`.
- `johnv8-mtp` — the earlier round (PR #27742 base + MTP port), kept for reference.

## Credits and license

Upstream llama.cpp and the authors of PR #28243 (qwen4exp), PR #27742 (Flash-Next), the RDNA4 mmvq patches (#27962, #27977); the Qwen3.8-Flash-Next model (Qwen team) and the GGUF/MTP files by unsloth. MIT license, as upstream. Claude (Anthropic) assisted with the analysis, kernel work and measurements throughout.
