# Qwen3.8-Flash-Next (UD-IQ3_XXS) na GPU AMD: jak odtworzyć nasz build z fuzjami i porównać z baseline

Dokument opisuje krok po kroku, jak na innym komputerze z GPU AMD (ROCm) zbudować dokładnie tę wersję llama.cpp, z którą uzyskaliśmy poniższe wyniki, jak ją uruchomić (1× GPU + CPU albo 2× GPU), jak zmierzyć osiągi tą samą metodą i jak porównać je z bazowym buildem unsloth (Vulkan). Wszystkie pełne pomiary (E0–E66) są w `WYNIK_OPTYMALIZACJI.md`; ten plik jest instrukcją i podsumowaniem.

---

## 1. Osiągi vs baseline (skrót)

Model: **Qwen3.8-Flash-Next UD-IQ3_XXS** (unsloth, 82 GB, 3 shardy) + głowa MTP `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` (2,8 GB).
Baseline: **najnowszy oficjalny build unsloth** `b10715-mix-86bd2d3` (Linux x64, Vulkan, `version 0.3.0-dev (build 10715, commit 92cedc867)`), ten sam model, ta sama głowa MTP, te same ustawienia.
Wspólne ustawienia: kontekst 65536 × 2 sloty, ubatch 512, MTP `draft-mtp` n-max 3 / p-min 0, 32 wątki CPU, `OMP_WAIT_POLICY=active`. tg = pojedynczy strumień tg128 (mediana 4–5 powtórzeń, dwa prompty: proza i kod); pp = prefill 2048 tokenów.

### 1×GPU (jedna karta 32 GB + CPU: 22 warstwy ekspertów na GPU, 26 na CPU; KV cache q8_0)

| build | MTP | pp2048 t/s | dekod proza | dekod kod |
|---|---|---|---|---|
| **nasz (HIP + fuzje + grafy HIP + cap J)** | **tak** | ~416* | **50,5** | **53,6** |
| nasz (HIP + fuzje + grafy HIP + cap J) | nie | **418** | 36,0 | 36,0 |
| unsloth b10715 (Vulkan) | tak | 176 | 40,4 | 44,0 |
| unsloth b10715 (Vulkan) | nie | 184 | 29,6 | 28,9 |

Dekod: +25 % / +22 % (z MTP), +22 % / +25 % (bez MTP); prefill ×2,3. (*prefill konfiguracji z MTP zmierzony na żywym serwerze, mediana 3.)

### 2×GPU (2 × 32 GB, wszyscy eksperci na GPU, `-ts 1,1.2`, tylko engram na CPU; KV cache f16)

| build | MTP | pp2048 t/s | dekod proza | dekod kod |
|---|---|---|---|---|
| **nasz (HIP + fuzje + grafy HIP + cap J)** | **tak** | **820** | **62–65** | **66–70** |
| nasz (HIP + fuzje + grafy HIP + cap J) | nie | **942** | 39,7–40,1 | 39,6–40,1 |
| nasz + any-order (eksperymentalne, patrz §8) | nie | 942 | 41,5 | 41,4 |
| unsloth b10715 (Vulkan) | tak | 690 | 49,9 | 59,5 |
| unsloth b10715 (Vulkan) | nie | 894 | 45,6 | 45,5 |

Z MTP: dekod +24 % / +12 %, prefill +19 %. Bez MTP na 2 GPU Vulkan nadal ma szybszy dekod (45,5 vs 40–41,5); prefill po capie J jest po naszej stronie (942 vs 894). Zakresy dekodu z MTP to rozrzut między sesjami (dryf termiczny 2–4 %).

### Skąd biorą się zyski (wobec tego samego PR bez naszych zmian)

| krok | 1×GPU dekod z MTP | 2×GPU dekod z MTP | prefill |
|---|---|---|---|
| PR 28243 + 3 łatki RDNA4 (punkt wyjścia) | 45,9 / 48,7 | 53,5 / 56,8 | 1×: 386, 2×: 794 |
| + fuzje jąder (bit-exact) + OpenMP active | 49,7 / 52,9 | 60,2 / 63,9 | bez zmian |
| + grafy HIP | 50,5 / 53,6 (+10 %) | 62,8 / 67,0 (+17 %) | bez zmian |
| + cap kafla J w MMQ (bit-exact) | bez zmian | bez zmian | **1×: 418 (+8 %), 2×: 942 (+19 %)** |

Wszystkie włączone zmiany są **bit-exact** ze stockowymi jądrami: przy temperaturze 0 model generuje identyczne tokeny na dwóch promptach (krótkim i długim, ~1500 tokenów), sprawdzane skryptem `bitexact.sh` (§6.2). Jakość modelu jest więc dokładnie ta sama co w bazowym PR.

---

## 2. Sprzęt i oprogramowanie, na którym to zmierzono

| element | u nas |
|---|---|
| CPU | AMD EPYC 7543, 32 rdzenie / 64 wątki |
| RAM | 125 GB (model 82 GB musi się zmieścić w page cache razem z częścią na CPU) |
| GPU | 2 × AMD Radeon AI PRO R9700, 32 GB, **gfx1201** (RDNA4) |
| OS / kernel | Linux Mint 22.3, kernel 7.0.0-29 |
| ROCm | **7.2.4** (`/opt/rocm-7.2.4`, HIP 7.2.53211, AMD clang 22.0.0git roc-7.2.4) |
| Vulkan (tylko baseline) | sterownik RADV, `/usr/share/vulkan/icd.d/radeon_icd.json` |

Inne GPU AMD: build działa na każdej architekturze wspieranej przez ROCm (zmień `AMDGPU_TARGETS`, §4), ale **fuzje były weryfikowane jako bit-exact tylko na gfx1201**; na innej architekturze uruchom bramkę z §6.2 przed użyciem. Uwaga na jedną pułapkę: na HIP `__fmul_rn/__fadd_rn` dają inne bity niż `a*b`/`a+b`, dlatego repliki stockowych jąder używają zwykłych operatorów pod `#pragma clang fp contract(off)`.

---

## 3. Co jest w tym katalogu (co skopiować na drugi komputer)

```
johnv8/docs/README_ODTWORZENIE.md       ten plik
johnv8/docs/WYNIK_OPTYMALIZACJI.md      dziennik eksperymentów E0–E66 (tabele, werdykty)
johnv8/docs/README_FUZJE_FINAL.md       podsumowanie fuzji i lekcji
johnv8/docs/BENCH_SUMMARY_EN.md         tabela EN: nasz build vs unsloth
johnv8/docs/PLAN_ANYORDER.md            projekt any-order (eksperymentalny, §8)
johnv8/patches/seria-fuzje/             34 łatki git (format-patch) = historia tego repo ponad bazą PR 28243
johnv8/patches/e64_anyorder.patch       any-order (eksperymentalne, domyślnie OFF)
johnv8/patches/e55_forkjoin.patch       fork/join wielostrumieniowy (zamknięte: −40 % dekodu)
johnv8/skrypty/buduj_hip.sh             build HIP (cmake) — §4
johnv8/skrypty/start_flashnext_fuzje.sh launcher serwera 1×GPU + CPU — §5
johnv8/skrypty/bench_mtp.py             benchmark pp/tg przez llama-server — §6.1
johnv8/skrypty/bitexact.sh              bramka bit-exact dwóch buildów/wariantów — §6.2
johnv8/skrypty/spis_jader.py            spis jąder na token z trace rocprof — §6.3
johnv8/mikrotesty/                      mikrotesty HIP (any-order, capture grafu, event/kopie, koszt launchu)
```

Skrypty biorą ścieżki ze zmiennych środowiskowych: `MODEL` (pierwszy shard modelu), `DRAFT` (głowa MTP), `BIN` (katalog binarek; domyślnie `build-hip/bin` tego repo), `DEV` (`ROCm0`/`ROCm1`), `ROCM_PATH` (domyślnie `/opt/rocm-7.2.4`), `SCRATCH`/`BENCH_DIR` (pliki robocze). Domyślne wartości wskazują na `~/models/...` — ustaw zmienne albo zmień domyślne na początku skryptu.
---

## 4. Build

### 4.1 Zależności

- ROCm 7.2.4 z HIP i hipBLAS (u nas `/opt/rocm-7.2.4`); użytkownik w grupach `render` i `video`.
- cmake ≥ 3.21, gcc/g++, python3, curl, `rocm-smi` (w ROCm).
- Do baseline: sterownik Vulkan (RADV) i binarka unsloth (§7).

### 4.2 Punkt wyjścia: llama.cpp z PR 28243

Baza to **ggml-org/llama.cpp, PR #28243** (obsługa architektury `qwen4exp`, czyli Qwen3.8-Flash-Next) w wersji z 2026-09-02, commit **`2857e511`** (pobraliśmy tarball tego commitu z GitHuba). Drzewo z PR-em nałożonym na jego bazę leży w naszym repo jako pierwszy commit `aa382d3 "PR 28243 base (tarball 2857e511)"`.

Dwie drogi:

**A. Z gotowej serii łatek (najprościej).** Rozpakuj tarball commitu `2857e511` (albo `git fetch origin pull/28243/head && git checkout 2857e511`), zainicjuj repo i nałóż serię:

```bash
cd llama.cpp-2857e511
git init -q && git add -A && git -c user.name=x -c user.email=x@x commit -q -m "PR 28243 base (2857e511)"
git am johnv8/patches/seria-fuzje/*.patch   # albo po prostu: git clone tego repo
```

Seria (34 łatki, w kolejności) zawiera:
1. `0001-cuda-enable-the-mul_mat_vec_q-small-K-path-on-RDNA4` (plik `pr27962.patch`)
2. `0001-cuda-use-nwarps-2-for-2-8-column-batches-on-RDNA4` (plik `pr27977.patch`)
3. `0001-HIP-SWAR-__vsub4-__vcmpne4-for-IQ2-IQ3` (szybsze dot-producty IQ2/IQ3 na HIP)
4. fuzje glue hyper-connections `hc-combine` + `hc-mix` (przeniesione z forka)
5–33. nasze etapy E1–E11 (fuzje ogona conv, ogona eksperta wspólnego E6b, dedup kwantyzacji Q8_1 E6d, prolog i l2norm GDN E7/E7b, `hc_inject` w combine E10a, blok hc w 3 jądrach E10c v3b, pomiar czasu, plan fuzji E11 [OFF]) — każdy z przełącznikiem env `GGML_JOHNV8_*`
34. cap kafla J w MMQ dla `mul_mat_id` (`GGML_JOHNV8_MMQ_ID_JMAX`, E55/E57)

Jeśli `git am` odrzuci którąś łatkę (inna wersja bazy), użyj dokładnie commitu `2857e511` — seria była generowana względem niego.

**B. Ręcznie od upstream.** Nałóż `pr28243.patch` na odpowiadający mu commit upstream, potem serię z punktu A od łatki 1. Droga B ma sens tylko, gdy chcesz przenieść zmiany na nowszy PR.

### 4.3 Kompilacja

`johnv8/skrypty/buduj_hip.sh <katalog_drzewa> [wątki] [arch]` robi dokładnie to:

```bash
export ROCM_PATH=/opt/rocm-7.2.4; export PATH="$ROCM_PATH/bin:$PATH"
cmake -B build-hip -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx1201 \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DGGML_CUDA_COMPRESSION_MODE=size
cmake --build build-hip -j 16
```

- `AMDGPU_TARGETS`: `gfx1201` = R9700 / RX 9070; RX 7900 = `gfx1100`; MI300 = `gfx942` itd.
- `GGML_CUDA_GRAPHS=ON` jest potrzebne do grafów HIP (+3–4 % dekodu; bit-exact).
- Pełny build od zera: ~20–40 min na 16 wątkach. Binarki: `build-hip/bin/llama-server`, `llama-completion`, `llama-bench`.
- Nie przebudowuj drzewa, z którego akurat działa serwer/benchmark (podmiana `.so` pod działającym procesem).

---

## 5. Modele i uruchomienie

### 5.1 Pliki

Z Hugging Face (repozytorium unsloth dla Qwen3.8-Flash-Next w GGUF):
- `UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-0000{1,2,3}-of-00003.gguf` (10,9 MB + 49,6 GB + 32,4 GB = ~82 GB),
- folder `MTP/`: `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` (2,8 GB; wariant „shared” dzieli embeddingi z modelem — używaliśmy tego).

Model ładowany jest przez pierwszy shard (`-m ...-00001-of-00003.gguf`).

### 5.2 Wariant 1×GPU (32 GB) + CPU — launcher `start_flashnext_fuzje.sh`

Kluczowe elementy (pełna komenda w skrypcie):

```bash
export ROCM_PATH=/opt/rocm-7.2.4 LD_LIBRARY_PATH="$B:/opt/rocm-7.2.4/lib"
export OMP_WAIT_POLICY=active            # wątki CPU spinują: +2 % z MTP (E40/E43)
export GGML_JOHNV8_HCBLOCK=1             # blok hc w 3 jądrach (E10c v3b), bit-exact
export GGML_JOHNV8_MMQ_ID_JMAX=32        # cap kafla J w MMQ (E57/E59), bit-exact, prefill +8–19 %
# grafy HIP są domyślnie włączone; GGML_CUDA_DISABLE_GRAPHS=1 je wyłącza

CPU_OD=22; WARSTWY=$(seq -s'|' "$CPU_OD" 47)
llama-server -m "$M" --host 127.0.0.1 --port 8092 \
  --device ROCm1 -ngl 99 --fit off \
  -c 65536 -np 2 --kv-unified --cache-type-k q8_0 --cache-type-v q8_0 \
  -ot "per_layer_token_embd=CPU,blk\.($WARSTWY)\.ffn_.*_exps=CPU" \
  -md "$MD" --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0 -devd ROCm1 --spec-draft-ngl 99 \
  --flash-attn on -ub 512 -b 1536 -t 32 -tb 48 --no-warmup --jinja --reasoning-format deepseek
```

Znaczenie:
- `-ot per_layer_token_embd=CPU` — tablica engramu (per-layer token embedding) zawsze na CPU (duża, rzadko czytana).
- `blk.(22|…|47).ffn_.*_exps=CPU` — eksperci warstw 22–47 na CPU (26 warstw), warstwy 0–21 na GPU. `CPU_OD` = pierwsza warstwa na CPU; przy 32 GB VRAM, ctx 65536 × 2 sloty i KV q8_0 mieści się `CPU_OD=22` (GPU zajęte ~31,2 GB). Mniej VRAM → niższy `CPU_OD` (każda warstwa ekspertów to ok. 1,5 GB), więcej → wyższy.
- `--cache-type-k/v q8_0` — na 1 karcie KV f16 przy tym kontekście się nie mieści; q8_0 nie jest bit-exact z f16 (inne tokeny w długich generacjach), ale przepustowość ta sama.
- MTP: głowa na tym samym GPU (`-devd ROCm1`), n-max 3, p-min 0 (strojenie n-max 2–6 i p-min sprawdzone: 3/0 najlepsze).
- `-t 32` — 32 wątki CPU (na 64-wątkowym EPYC: 24 bez MTP, 32 z MTP; na innym CPU dobierz eksperymentalnie), `-tb 48` dla promptu.
- `--device ROCm1` — u nas druga karta; na jednej karcie użyj `ROCm0`.

Sprawdzenie: `curl http://127.0.0.1:8092/health` → `{"status":"ok"}`; pierwsze ładowanie z dysku trwa kilka minut (82 GB), kolejne z page cache ~1 min.

### 5.3 Wariant 2×GPU (2 × 32 GB)

Te same env; komenda serwera (jak w `bench_mtp.py` przy `CPU_OD=48`):

```bash
llama-server -m "$M" --host 127.0.0.1 --port 8092 \
  --device ROCm0,ROCm1 -ts 1,1.2 -ngl 99 --fit off \
  -c 65536 -np 2 --kv-unified --cache-type-k f16 --cache-type-v f16 \
  -ot per_layer_token_embd=CPU \
  -md "$MD" --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0 --spec-draft-device ROCm1 --spec-draft-ngl 99 \
  --flash-attn on -ub 512 -b 1536 -t 32 -tb 32 --no-warmup
```

`-ts 1,1.2` (proporcja podziału tensorów; karta z głową MTP dostaje więcej) daje ~30 GB + ~30 GB. Wszyscy eksperci na GPU, więc dekod jest w pełni ograniczony launchami jąder — dlatego fuzje dają tu więcej (+17 %) niż na 1×GPU (+10 %).

---

## 6. Jak mierzyliśmy (żeby liczby były porównywalne)

### 6.1 Przepustowość: `bench_mtp.py`

Skrypt startuje `llama-server` z podanymi parametrami, buduje prompty przez `/tokenize` + `/detokenize` (długość dokładna, nie szacowana), mierzy prefill pp512/pp1024/pp2048 i dekod tg128 na dwóch promptach (proza, kod), powtarza, liczy medianę i zapisuje `bench_<ETYK>.json` + log serwera `bench_<ETYK>.log`. Sterowanie przez zmienne środowiskowe:

```bash
# 1×GPU, hosting, z MTP, tylko dekod, 4 powtórki (MODEL/DRAFT wskazują pliki modelu):
MODEL=… DRAFT=… BIN=/…/build-hip/bin ETYK=test1 BACKEND=rocm DEVICES=ROCm1 CPU_OD=22 CTX=65536 NP=2 UB=512 KV=q8_0 \
SPEC=draft-mtp NMAX=3 PMIN=0 WATKI=32 POWT=4 TG_ONLY=1 \
GRAFY_OFF=0 OMP_WAIT_POLICY=active GGML_JOHNV8_HCBLOCK=1 GGML_JOHNV8_MMQ_ID_JMAX=32 python3 bench_mtp.py
# bez MTP: SPEC=none ; z prefillem: TG_ONLY=0 ; 2×GPU: DEVICES="ROCm0,ROCm1" DEVD=ROCm1 TS="1,1.2" CPU_OD=48 KV=f16 GPU0_OK=1
# baseline Vulkan: BACKEND=vulkan BIN=/…/unsloth-llama DEVICES=Vulkan1 (lub "Vulkan0,Vulkan1")
```

Zasady, które okazały się konieczne:
- **Rozgrzej page cache** przed każdą serią (`cat shardy…gguf > /dev/null`) — po buildach/profilach cache wypada i wynik spada o 10–15 %.
- Powtarzaj bazę na końcu serii (`baza2`): dryf termiczny w jednej sesji sięga 2–4 %; porównuj tylko wewnątrz sesji.
- **Porównania t/s z MTP mają sens tylko między buildami bit-exact** — inna numeryka zmienia tekst i akceptację draftu (±8 %). Warianty nie-bit-exact porównuj bez MTP.
- `--no-warmup` + pierwszy pomiar jako rozgrzewka; `rocm-smi` do sprawdzenia, że VRAM zwolnił między przebiegami (skrypt to robi).
- Nic innego nie może pracować na GPU/CPU w czasie pomiaru (sesje 2×GPU robiliśmy tylko przy obu wolnych kartach).

### 6.2 Jakość: bramka bit-exact `bitexact.sh`

```bash
./bitexact.sh <binA> <binB> <etykieta> [ZMIENNA=wartość … dla B]
# np. ten sam build, wariant vs baza:
./bitexact.sh $B $B jcap GGML_JOHNV8_MMQ_ID_JMAX=32
```

Uruchamia serwer A, potem B (na `ROCm1`, ctx 8192, KV q8_0, grafy HIP wyłączone w harnessie, `--temp 0`), generuje 200 tokenów na dwóch promptach: `[k]` krótki („Silnik 2.0 TDI w Passacie B6…”) i `[d]` długi (~1500 tokenów: 60 powtórzeń oferty + pytanie) i porównuje teksty. Werdykt `IDENTYCZNE` na obu = wariant bit-exact. Kontrola: A vs A musi być identyczne; pusty tekst = błąd, nie sukces. Dla zmian sprzętowych (inna architektura GPU) uruchom ją najpierw dla całego builda (`GGML_JOHNV8_*=0` wszystkie vs domyślne).

Grafy HIP sprawdzono osobno (E48, 3 bramki): bit-exact.

### 6.3 Profil jąder (opcjonalnie)

`rocprofv3 --kernel-trace --output-format csv -- llama-completion …` + `spis_jader.py trace.csv` daje spis jąder na token (u nas 4275 → 2228 po fuzjach na 1×GPU) i sumę czasu vs rozpiętość (luki między launchami ~4–5 µs × ~2200 = 10–11 ms/token to koszt bitu bariery AQL na ROCm).

---

## 7. Baseline unsloth (Vulkan)

1. Pobierz z GitHub Releases forka llama.cpp unsloth: `app-b10715-mix-86bd2d3-linux-x64-vulkan.tar.gz`, rozpakuj do katalogu (u nas `<unsloth>`); `LD_LIBRARY_PATH=. ./llama-server --version` → `build 10715, commit 92cedc867`.
2. Uruchom z tymi samymi parametrami co nasz serwer, ale backend Vulkan:

```bash
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json GGML_VK_ALLOW_GRAPHICS_QUEUE=1 LD_LIBRARY_PATH=. \
./llama-server -m "$M" --device Vulkan1 -ngl 99 --fit off -c 65536 -np 2 --kv-unified \
  --cache-type-k q8_0 --cache-type-v q8_0 -ot "per_layer_token_embd=CPU,blk\.(22|…|47)\.ffn_.*_exps=CPU" \
  -md "$MD" --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0 --spec-draft-device Vulkan1 --spec-draft-ngl 99 \
  --flash-attn on -ub 512 -b 1536 -t 32 --no-warmup
```

Uwagi: na 1×GPU unsloth mierzony przy tym samym `CPU_OD=22` (zmieścił się w VRAM; skrypt E51a ma awaryjnie `CPU_OD=20`, ale nie było to potrzebne). 2×GPU: `--device Vulkan0,Vulkan1 -ts 1,1.2`, KV f16. Baseline można też zbudować samemu z HIP (bez naszych łatek) — wtedy „PR+łatki” w tabeli z §1 to punkt odniesienia dla samych fuzji.

---

## 8. Warianty eksperymentalne i to, co nie zadziałało

| co | wynik | stan |
|---|---|---|
| `GPU_MAX_HW_QUEUES=8` | +3,5 % w jednej sesji 2×GPU, 0 w kolejnej i na 1×GPU | szum, nie używamy |
| fork/join wielostrumieniowy (`patches/e55_forkjoin.patch`, `GGML_JOHNV8_FORKJOIN=1`) | bit-exact, ale −40…−50 % dekodu (zdarzenia między strumieniami HIP za drogie na ROCm 7.2.4) | zamknięte |
| any-order z trackerem hazardów (`patches/e64_anyorder.patch`, `GGML_JOHNV8_ANYORDER=1`, `_STATS=1`, `_MINNODES=400`) | bit-exact (3 bramki), 15 % launchy bez bitu bariery AQL; 2×GPU bez MTP 40,1 → 41,5 t/s (+2…4 %), z MTP +1 %, 1×GPU 0 | działa, domyślnie OFF; dalszy zysk wymagałby przestawiania grafu |
| KV q8_0 na 2×GPU | ta sama przepustowość co f16, ale nie bit-exact z f16 | na 2×GPU f16 |
| E3 (gęsta uwaga przy małym n_kv), E11 (plan fuzji) | nie bit-exact / brak zysku | OFF |
| `OMP_WAIT_POLICY=passive` | −15…−22 % z MTP | active |

Any-order wymaga wyłączenia capture grafu HIP dla danego grafu (flaga `hipExtAnyOrderLaunch` jest gubiona przy `hipStreamBeginCapture` na ROCm 7.2.4) — patch robi to automatycznie tylko dla grafów ≥ 400 węzłów (grafy MTP zostają w grafach HIP).

---

## 9. Szybki start (TL;DR)

```bash
# 1. baza + łatki
tar xf llama.cpp-2857e511.tar.gz && cd llama.cpp-2857e511
git init -q && git add -A && git -c user.name=x -c user.email=x@x commit -q -m base
git am johnv8/patches/seria-fuzje/*.patch
# 2. build (gfx1201 → wpisz swoją architekturę)
johnv8/skrypty/buduj_hip.sh "$PWD" 16 gfx1201
# 3. serwer 1×GPU + CPU (ścieżki do modeli i drzewa na początku skryptu)
MODEL=… DRAFT=… DEV=ROCm0 johnv8/skrypty/start_flashnext_fuzje.sh   # PORT=8092, CTX=32768/65536, NP=2, CPU_OD=22, WATKI=32
# 4. bramka jakości i benchmark
MODEL=… DEV=ROCm0 CARD=card0 johnv8/skrypty/bitexact.sh build-hip/bin build-hip/bin kontrola          # A=A → IDENTYCZNE
BIN=$PWD/build-hip/bin ETYK=nasz SPEC=draft-mtp NMAX=3 PMIN=0 CPU_OD=22 CTX=65536 NP=2 UB=512 KV=q8_0 WATKI=32 POWT=4 TG_ONLY=0 GRAFY_OFF=0 \
  OMP_WAIT_POLICY=active GGML_JOHNV8_HCBLOCK=1 GGML_JOHNV8_MMQ_ID_JMAX=32 MODEL=… DRAFT=… python3 johnv8/skrypty/bench_mtp.py
```

Oczekiwane na 1 × R9700 + EPYC: dekod ~50/53 t/s z MTP, ~36 bez, prefill pp2048 ~418 t/s; na 2 × R9700: ~62–65/66–70 z MTP, ~40 bez, prefill 820–942 t/s.
