# Qwen3.8-Flash-Next (UD-IQ3_XXS, 82 GB) on 2x AMD Radeon AI PRO R9700 — our fused build vs stock unsloth

Quality-preserving variants only: every kernel fusion in our build is bit-exact with the stock kernels (same tokens at temp 0), HIP graphs are bit-exact too.
Baseline = the newest official unsloth build (`b10715-mix-86bd2d3`, 2026-08-31, Linux x64 Vulkan, downloaded from GitHub releases) with the same model, same MTP draft head (Q8_0), same settings.
Common settings: context 65536 x 2 slots, ubatch 512, MTP `draft-mtp` n-max 3 / p-min 0, 32 CPU threads, `OMP_WAIT_POLICY=active`. tg = single-stream tg128 (median of 5); pp = prefill throughput (mean of 4).

## 1x GPU (GPU1 only; 26 of 48 expert layers + engram on CPU; KV cache q8_0)
| build | MTP | pp1024 | pp2048 | decode prose | decode code |
|---|---|---|---|---|---|
| **ours (fused HIP + graphs)** | **yes** | 378* | 416* | **50.5 t/s** | **53.6 t/s** |
| ours (fused HIP + graphs) | no | 372 | 390 | 36.0 | 36.0 |
| unsloth b10715 (Vulkan) | yes | 173 | 176 | 40.4 | 44.0 |
| unsloth b10715 (Vulkan) | no | 177 | 184 | 29.6 | 28.9 |
Ours vs unsloth: decode +25% / +22% (MTP), +22% / +25% (no MTP); prefill x2.2. (*prefill of the MTP config measured on the live server, median of 3.)

## 2x GPU (all experts on GPU, tensor split 1:1.2, only engram on CPU; KV cache f16)
| build | MTP | pp1024 | pp2048 | decode prose | decode code |
|---|---|---|---|---|---|
| **ours (fused HIP + graphs)** | **yes** | 552 | 577 | **62.0 t/s** | **66.4 t/s** |
| ours (fused HIP + graphs) | no | 611 | 622 | 38.9 | 39.0 |
| unsloth b10715 (Vulkan) | yes | 595 | 690 | 49.9 | 59.5 |
| unsloth b10715 (Vulkan) | no | 688 | 894 | 45.6 | 45.5 |
Ours vs unsloth with MTP: decode +24% / +12%. Without MTP the stock Vulkan backend is faster on 2 GPUs (45.6 vs 38.9 decode, 894 vs 622 prefill) — its plain decode and prefill kernels beat HIP on RDNA4; our advantage comes from the cheaper MTP verification path (batch of 4) plus the fusions. MTP acceptance is the same (0.68, mean draft length 3.0) in both builds, i.e. the same tokens.

## Best configuration per setup
- 1x GPU (one card busy, the other free for other work): ours + MTP, 50.5 / 53.6 t/s, prefill ~400 t/s.
- 2x GPU: ours + MTP, 62–63 / 66–67 t/s, prefill ~580 t/s (KV q8_0 gives the same throughput; f16 keeps exact numerics).

## Update 2026-09-03 — MMQ column-tile cap for MoE prefill (bit-exact, `GGML_JOHNV8_MMQ_ID_JMAX=32`)
The MoE matmul (`mul_mat_id`, 59% of 2-GPU prefill time) picked a 128-column tile from the 512-token ubatch although each expert only gets ~10 rows; capping the tile choice at 32 columns removes the wasted work. Same kernels, same accumulation order → identical tokens at temp 0 (gates passed for 8/16/32/64). Decode is unaffected (it uses the mat-vec path). Measured within one session each (baseline re-run in the same session, 3–4 repeats):

| setup | MTP | pp2048 before | pp2048 after | decode |
|---|---|---|---|---|
| 1x GPU (26 expert layers on CPU) | no | 386 | **418 (+8%)** | unchanged (36.5) |
| 2x GPU (all on GPU, KV f16) | no | 794* | **942 (+19%)** | unchanged (39.7 / 39.6) |
| 2x GPU (all on GPU, KV f16) | yes (n-max 3) | 690 | **820 (+19%)** | unchanged (62.0 / 66.0) |

*The 2-GPU no-MTP baseline measured 622 t/s in the earlier session (table above) and 794 t/s in this one with identical settings; the gain is quoted against the same-session baseline. With the cap, our 2-GPU prefill without MTP (942) is above the stock Vulkan build (894); decode without MTP on 2 GPUs (39.7 vs 45.5) remains the open gap. Deployed on the 1-GPU server (port 8092). `GPU_MAX_HW_QUEUES=8` was tested as well: +3.5% decode in one 2-GPU session, 0 in the next and 0 on 1 GPU → noise, not deployed.

**Any-order launches (hazard-tracked, `GGML_JOHNV8_ANYORDER=1`, not deployed):** independent kernels are submitted without the AQL barrier bit; bit-exact (three gates), 15% of launches barrier-free. 2-GPU decode without MTP 40.1 → 41.5 t/s (+2–4%), with MTP +1%, 1 GPU no change. The remaining barriers are real dependency chains (≈400 RAW denials per graph), so further gains would need graph reordering.
