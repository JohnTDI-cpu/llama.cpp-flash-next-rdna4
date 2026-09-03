# Flash-Next na GPU1 + CPU: podsumowanie optymalizacji jader (2026-09-02/03)

Drzewo: `<repo>` (PR 28243 + latki mmvq small-K / nwarps=2 / SWAR + fuzje johnv8). Launcher: `start_flashnext_fuzje.sh`.
Wszystkie wlaczone etapy sa **bit-exact** wobec PR+latki (bramka: te same tokeny na dwoch promptach, temp 0; E22/E31/E45/E47/E48).
Lacznie w konfiguracji hostingu: **45,9/48,7 -> 50,5/53,6 t/s (+10%)** przy identycznych tokenach; bez MTP +20%.

## Wynik (hosting: ctx 65536 x 2 sloty, KV q8_0, ub 512, 26 warstw ekspertow na CPU, MTP n-max 3, t32, OMP_WAIT_POLICY=active)
| build | proza t/s | kod t/s |
|---|---|---|
| PR+latki (stary launcher, t48) | 45,9 | 48,7 |
| fuzje HEAD | 49,7 | 52,9 |
| fuzje HEAD + HIP graphs (E48) | **50,5** | **53,6** |
Bez MTP (ctx 8192, t24): 29,6 -> 35,6 t/s (+20%). Jader na token: 4275 (PR) -> 2228.

## 2x GPU (E50, obie karty wolne): ctx 65536 x 2 sloty, ub 512, MTP n-max 3, t32, OMP active
| build | KV | grafy | pp2048 | proza | kod |
|---|---|---|---|---|---|
| PR+latki | f16 | OFF | 584 | 53,5 | 56,8 |
| PR+latki | f16 | ON | - | 55,6 | 59,8 |
| fuzje HEAD | f16 | OFF | 571 | 60,2 | 63,9 |
| fuzje HEAD | f16 | ON | 577 | **62,8** | **67,0** |
| fuzje HEAD | q8_0 | ON | 584 | 62,6 | 50,9* |
*KV q8_0 nie jest bit-exact z f16 -> na prompcie 'kod' tekst sie rozjechal i akceptacja draftu spadla z 0,68 do 0,44; proza i prefill identyczne, wiec q8_0 nie kosztuje przepustowosci.


## Zestawienie 1x GPU vs 2x GPU (tylko warianty z zachowana jakoscia: wszystkie etapy bit-exact ze stockiem + HIP graphs)
Wspolne: drzewo fuzje HEAD 6735ff5, MTP n-max 3 / p-min 0, ctx 65536 x 2 sloty, ub 512, -t 32, OMP_WAIT_POLICY=active, HIP graphs ON.
| | 1x GPU (GPU1 + CPU) | 2x GPU (GPU0+GPU1) |
|---|---|---|
| gdzie eksperci | 22 warstwy na GPU1, 26 warstw na CPU (+engram) | wszystkie na GPU (ts 1:1,2), tylko engram na CPU |
| KV cache | q8_0 (launcher; f16 nie zmiescilby OD22) | f16 (q8_0: ta sama predkosc) |
| VRAM | GPU1 31,2 GB; GPU0 wolne | ~30 GB + ~30 GB; obie karty zajete |
| tg proza | 50,5 t/s | 62,8 t/s (+24%) |
| tg kod | 53,6 t/s | 67,0 t/s (+25%) |
| pp1024 | 378 t/s | 552 t/s (+46%) |
| pp2048 | 416 t/s | 577 t/s (+39%) |
| dla porownania PR+latki (bez fuzji, bez grafow) | 45,9 / 48,7 | 53,5 / 56,8 |
| zysk naszych fuzji+grafow | +10% | +17% |
Uwagi: tg = pojedynczy strumien (tg128, mediana 5 powtorzen); przy 2 slotach naraz na slot wyjdzie mniej. 1x GPU: prefill z serwera 8092
(/completion, mediana 3). 2x GPU: E50 (obie karty wolne). Roznica 1x->2x = koszt 26 warstw ekspertow na CPU (~0,33 ms/warstwe/token)
+ granice splitow; fuzje daja wiecej na 2x GPU, bo tam dekod jest w pelni launch-bound.

## Zestawienie po angielsku vs bazowy unsloth (E51): patrz BENCH_SUMMARY_EN.md

## Etapy i werdykty
| etap | co | jader/token | bit-exact | zysk |
|---|---|---|---|---|
| hc-mix + scale_silu, hc-combine | fuzje glue hyper-connections (z forka) | -1071 | tak | +9% bez MTP |
| E1 | cpy ogona conv bez ggml_cont | -36 | tak | ~0 |
| E3 | gesta uwaga gdy n_kv <= top-k | -300 (krotki ctx) | NIE | OFF |
| E4 | migawki stanu conv w jednym jadrze (MTP) | -111 | tak | ~0 |
| E6b | ogon shexp: sigmoid*shexp+moe | -96 | tak (po zamianie __fmul_rn na zwykle operatory) | maly |
| E6d | dedup kwantyzacji Q8_1 (attn_qkv|attn_gate) | -65 | tak | maly |
| E7 | prolog GDN (sigmoid beta, softplus(alpha+dt)*A) w jadrze | -144 | tak | maly |
| E7b | l2norm q/k w jadrze GDN (fmaf) | -72 | tak | ~0 |
| E10a | hc_inject (matvec f32) w jadrze combine | -96 | tak | ~0 |
| E10c v3b | blok hc w 3 jadrach (A0/A1/B1) | -158 (vs v0: -629) | tak | +3% bez MTP, +0,6% z MTP |
| E11 | plan fuzji (pomijanie matcherow) | - | NIE | odrzucone |
| OpenMP active + t32 | polityka oczekiwania watkow CPU | - | n/d | +2% z MTP |
| HIP graphs | replay grafu | - | tak (E48: 3 bramki) | +3% |

## Lekcje
1. Koszt "na jadro" nie jest staly: fuzje usuwajace przebiegi po pamieci [10240 x nt] daly najwiecej; drobne elementwise ~0.
2. Bilans tokena (bez MTP, ~29 ms): jadra GPU ~17 ms + luki miedzy launchami ~4-5 us x ~2200 + warstwy CPU ~8 ms (0,33 ms/warstwe).
3. `__fmul_rn/__fadd_rn` na HIP gfx1201 daja inne bity niz `a*b`/`a+b` -> replikowac stock zwyklymi operatorami pod `#pragma clang fp contract(off)`.
4. Pomiary t/s z MTP miedzy buildami o roznej numeryce sa niemiarodajne (tekst i akceptacja draftu sie rozjezdzaja); tylko pary bit-exact albo bez MTP.
5. Po buildach/profilach page cache wypada -> przed A/B `cat shardy > /dev/null`.
6. `ggml_set_output` na tensorze posrednim zmienia alokacje calego grafu i numeryke (ujawnia hazardy) — nie uzywac jako pinu.
7. Replika mmvq musi zachowac siatke blokow jak stock; wspolna prace (normy, kwantyzacja) robic raz, nie w kazdym bloku.

## Przelaczniki env (domyslnie ON, chyba ze napisano inaczej)
GGML_JOHNV8_HC_FUSE, GGML_JOHNV8_MIX_FUSE, GGML_JOHNV8_E1, GGML_JOHNV8_E3 (OFF), GGML_JOHNV8_CONV_SNAP, GGML_JOHNV8_SHEXP_FUSE,
GGML_JOHNV8_Q8_DEDUP, GGML_JOHNV8_GDN_PROLOG, GGML_JOHNV8_GDN_L2 (+_FMA), GGML_JOHNV8_INJECT_FUSE, GGML_JOHNV8_HCBLOCK (0/1/2/3),
GGML_JOHNV8_PLAN_CACHE (OFF), GGML_JOHNV8_TIMING (diagnostyka), GGML_CUDA_DUMP_DISPATCH (diagnostyka fuzji).
Szczegoly i wszystkie pomiary: WYNIK_OPTYMALIZACJI.md (E0-E49).

## Aktualizacja 2026-09-03 — cap kafla J w MMQ (bit-exact, wdrożone), fork/join (zamknięte)

- **Cap J** (`GGML_JOHNV8_MMQ_ID_JMAX=32`, commit fb29fcd, w launcherze 8092): MoE `mul_mat_id` dobierał kafel 128 kolumn od 512 tokenów ubatcha, a ekspert ma ~10 wierszy → cap na 32 usuwa pustą pracę; te same jądra i kolejność akumulacji → identyczne tokeny (bramki 8/16/32/64). Prefill pp2048: 1×GPU 386 → **418** (+8 %), 2×GPU bez MTP 794 → **942** (+19 %, powyżej Vulkan 894), 2×GPU z MTP 690 → **820**. Dekod bez zmian.
- **`GPU_MAX_HW_QUEUES=8`**: +3,5 % dekodu w jednej sesji 2×GPU, 0 w następnej i 0 na 1×GPU → szum, nie wdrożone.
- **Fork/join wielostrumieniowy** (patch E55, bit-exact): każdy wariant z realnymi forkami traci 40–50 % dekodu (zdarzenia między strumieniami HIP za drogie na ROCm 7.2.4) → kierunek zamknięty.
- **Any-order (bez bitu bariery AQL)**: flaga nie przeżywa capture grafu HIP → wyklucza się z grafami (+4 %); plan konserwatywny w `PLAN_ANYORDER.md`, decyzja go/no-go po E61 (reżim bez grafów na 2×GPU) i mikrotestach E63.
- **Any-order (E64–E66, worktree wt-anyorder, `GGML_JOHNV8_ANYORDER=1`)**: tracker hazardów pozwala niezależnym jądrom startować bez bitu bariery AQL; bit-exact (3× bramki), 15 % launchy bez bariery; 2×GPU dekod bez MTP 40,1 → 41,5 t/s (+2…4 %), z MTP +1 %, 1×GPU 0. Reszta luk to prawdziwe łańcuchy zależności — dalszy zysk wymagałby przestawiania grafu. Nie wdrożone (domyślnie OFF).
