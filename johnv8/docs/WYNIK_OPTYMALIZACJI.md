# Optymalizacja jąder Qwen3.8-Flash-Next (qwen4exp) na RDNA4 — dziennik eksperymentów E0–E66

Chronologiczny dziennik pracy (2026-09-02/03): każdy etap z pomiarem, werdyktem bit-exact i decyzją. Setupy: 1× R9700 + CPU (część ekspertów na CPU) oraz 2× R9700 + CPU. Wcześniejsze notatki rozpoznawcze (dobór modelu/kwantyzacji, strojenie MTP) pominięto; wynikowe konfiguracje są w README_FUZJE_FINAL.md i BENCH_SUMMARY_EN.md.

### E0 — kalibracja fuzji na PR 28243 (rocprofv3, 40 tokenow)
PR 4275 -> PR+fuzje **3982 (-293)**: k_bin_bcast 941 -> 745 (-196), copyBuffer 249 -> 152 (-98),
**scale_f32 402 -> 402 (0!)**, unary/rms_norm bez zmian. Oczekiwane przy pelnym dzialaniu: scale -387,
bcast -670. WNIOSEK: fuzja hc-combine odpala (~2/warstwe), **fuzja hc-mix NIE odpala wcale** na
drzewie PR — wzorzec sie nie dopasowuje. To jest najtanszy zysk planu (do -190..-480 jader).
KOREKTA E0: oba commity fuzji (c689018e4 hc-combine, f76552838 hc-mix) NIE WESZLY (konflikt, reset) —
"PR+fuzje" = tylko 3 latki jader + reorder johnv8_opt w qwen4exp.cpp. Czyli -293 (bcast -196, kopie -98)
daje sam reorder (multi-ADD upstreamu lapie kolaps strumieni). Fuzje trzeba przeniesc RECZNIE.

### E0b — fuzje forka NA PR 28243 (rocprofv3, 40 tokenow, bez MTP, 25 warstw na CPU)
| build | jader/token | scale | bcast | unary | gated | gen t/s (pod profilerem) |
|---|---|---|---|---|---|---|
| PR | 4275 | 402 | 941 | 297 | 183 | 19,5 |
| PR + latki (reorder johnv8_opt) | 3982 | 402 | 745 | 297 | 183 | 19,0 |
| **PR + latki + fuzje hc-mix/hc-combine** | **3204 (-1071)** | **13** | **357** | **102** | **85** | **22,5 (+18%)** |
Nowe jadra: hc_mix_collapse 98, scale_silu 98, hc_combine 97 -> fuzje lapia WSZYSTKIE miejsca.
E0 zaliczone: analityka (-1062) zgadza sie z pomiarem (-1071). Poprzedni brak zysku na starym forku
= inna baza. Drzewo: <repo>.

### E5 (konkatenacja par wag) — analiza projektowa 2026-09-02 wieczor
Plan z naglowkow GGUF: 144 par (48 warstw), kazda para tego samego typu (Q6_K/Q8_0/F32/BF16), bloki
kwantyzacji wzdluz ne0 -> sklejenie = zlepienie surowych bajtow, bit-exact. Narzedzie: e5_sklej_pary.py
(--plan OK, przepis nie uruchomiony). ALE: pod MTP dekodowanie idzie partia T=1+n_draft=4, wiec widoki
wierszy sklejonego wyniku [rows,T] sa NIECIAGLE; konsumenci wymagajacy ciaglosci (reshape_3d/4d,
topk_moe, ssm_conv?) potrzebuja ggml_cont (+1 jadro) -> zysk netto: uwaga q|k|v -4/warstwe (view_3d
+ norm/rope stridowane OK), indexer q|k -1, GDN qkv|gate -2+1, router i beta|alpha ~0. Realnie ~-100/token
(nie -216). Odlozone za E6b/E6d/E7/E8; przepis 82 GB dopiero, gdy reszta wyczerpana.
Fakt (common.h need_n_rs_seq): pod draft-mtp n_rs_seq = n_max (u nas 3) -> n_slots = 4 migawki stanu conv
na warstwe GDN na KAZDY krok dekodowania. E1 zdejmuje 4x37 blitow (cont), E4 (4 CPY -> 1 jadro) kolejne
3x37 = -111 jader/token. Bez MTP: n_slots = 1, E4 = 0. Kod E4: conv-snap.cu (uzbrojone po E13b).

### PULAPKA METODYCZNA (2026-09-02 ~23:00): page cache
Po serii buildow i profili Cached spadl ze ~106 GB do 67 GB — model IQ3 (82 GB) byl tylko czesciowo
w page cache, a eksperci na CPU sa mmapowane -> dekodowanie czytalo wagi z NVMe. Objaw: A/B w E13
dal 39,0 t/s dla konfiguracji, ktora rano robila 45,3. Wszystkie A/B tg po ~22:30 sa podejrzane
(E6a/E6b), liczby jader (rocprofv3) i bit-exact NIE (nie zaleza od cache). Od E14: rozgrzewka
`cat shardy > /dev/null` przed kazdym A/B + kontrola Cached >= 85 GB.

### E6a (GGML_JOHNV8_OPT=3, rms_norm+mul przez wage 3D) — ODRZUCONE (2026-09-02 23:35)
- bit-exact: ROZNE tokeny vs OPT=1 (fuzja mnozy przed zaokragleniem posrednim; stock zaokragla dwa razy).
- spis jader pod rocprofv3: llama-completion zawisl na 100% CPU (42 min), `timeout 900` zignorowany
  (SIGTERM przechwycony) -> zabity recznie; od teraz wszedzie `timeout -s KILL 900`.
- wniosek: te 96 jader/token da sie zdjac tylko wlasnym jadrem z wylaczona kontrakcja FMA
  (jak hc-mix), replikujacym dwa zaokraglenia stocka. Wpisane jako E6c na pozniej.

### E7 + E6d + naprawa E3 — zaimplementowane, czekaja na build/bramki w E14 (2026-09-02 ~23:55)
- Naprawa E3 (crash pod MTP): `llm_graph_input_qsa::set_input` pomija tensory QSA bez bufora (w rezimie
  gestym nie sa w grafie); klucze indeksera nadal ida do cache'u. Crash byl w `common_context_can_seq_rm`
  (dekod probny przy starcie serwera ze spekulacja) — dlatego padaly tylko biegi z draft-mtp.
- E6b: `__fmul_rn/__fadd_rn` zamiast luznych mnozen (hipcc kontraktuje do FMA -> ROZNE tokeny).
- E7 (commit cdd765a): `ggml_gated_delta_net_prolog` — src[6]=dt, src[7]=A, op_params[1]=1; jadro CUDA i CPU
  licza beta=sigmoid(beta), g=softplus(alpha+dt)*A dokladnie jak unary.cu/binbcast (rn-intrinsics).
  Sciezki bez fuzji (autoregressive/chunking) dostaja prolog policzony jawnie (`gdn_prolog_materialize`).
  Oczekiwane: -4 jadra/warstwe GDN = -144/token. Env GGML_JOHNV8_GDN_PROLOG=0 wylacza.
- E6d: cache kwantyzacji Q8_1 aktywacji w mmvq (klucz: tensor+data+ne/nb+rozmiar, wazny w jednym
  graph_compute, uniewazniany przy zapisie w miejsce). Pary ze wspolnym wejsciem: beta|alpha (36),
  gate_inp|gate_inp_shexp (48), indexer q|k (12), attn_qkv|attn_gate (12, jesli sasiednie w grafie).
  Oczekiwane: -60..-110 quantize_q8_1/token. Bit-exact z konstrukcji. Env GGML_JOHNV8_Q8_DEDUP=0 wylacza.
- E8 zdegradowane: fuzja ssm_conv+silu juz jest w PR (ggml-cuda.cu:4216), zostaje tylko concat (-36).

### KOREKTA (2026-09-02 23:55): wyniki drzewa fuzje od E1+E3 (E13, E6a, E6b, E13b) NIEWAZNE
Kazdy bieg binarki fuzje po commicie 8dc0529 padal na `GGML_ASSERT(buffer)` juz przy rozgrzewce
(llama-completion) lub dekodzie probnym serwera — nie tylko pod MTP. Profile prof5/6/7 maja 82 KB
(brak dekodu), "ROZNE" w E6a/E6b to porownanie pustych odpowiedzi. Naprawa E3 (54ae534) jest w HEAD;
pierwszy wazny build to build z lancucha E4 (23:49), pomiary wazne od E14. E4: latka nie nalozyla sie
na zmieniony ggml-cuda.cu (E6d) -> powtorka jako E15 po E14.

### E14 (2026-09-03 00:00): koncowe A/B z MTP na cieplym cache — PR+latki vs fuzje HEAD (2649d89 bez E4)
Warunki: 1xGPU (ROCm1), OD23 (25 warstw ekspertow na CPU), ctx 8192, np 1, shared head n-max 3, t48, POWT 6, tg128.
Bramki: E7 (GDN_PROLOG) IDENTYCZNE, E6d (Q8_DEDUP) IDENTYCZNE, E6b (SHEXP) ROZNE na kodzie, latki vs fuzje ROZNE -> izolacja w E16.
Spis jader: PR+latki 3982 -> fuzje HEAD 2479 (-1503 = -38%).

| build | tg128 proza | tg128 kod |
|---|---|---|
| A_latki | 39.40 ± 0.48 | 42.70 ± 0.18 |
| B_fuzje | 48.56 ± 0.31 | 46.21 ± 0.25 |
| A_latki2 | 39.61 ± 0.45 | 43.09 ± 0.36 |
| B_fuzje2 | 48.45 ± 0.52 | 46.39 ± 0.48 |

Srednio: proza 39.5 -> 48.5 (+23%), kod 42.9 -> 46.3 (+8%).
UWAGA: A_latki 39,5 t/s jest nizej niz poranne 45,3 dla tej samej konfiguracji na drzewie PR-src -> podejrzenie, ze ktoras
z latek (mmvq small-K / nwarps=2 / SWAR) szkodzi pod MTP (batch weryfikacji T=4). Rozstrzyga E17 (src vs latki vs fuzje).

### E16 (2026-09-03 00:08): izolacja bit-exact, kazdy etap osobno vs PR+latki (serwer, temp 0, 2 prompty)
| etap | proza | kod | werdykt |
|---|---|---|---|
| kontrola latki vs latki | IDENT | IDENT | pipeline deterministyczny |
| wszystko OFF | IDENT | IDENT | przelaczniki dzialaja |
| hc-combine (HC_FUSE) | IDENT | IDENT | OK |
| hc-mix + scale_silu (MIX_FUSE) | IDENT | IDENT | OK |
| E1 (bez ggml_cont) | IDENT | IDENT | OK |
| E3 (gesta uwaga <= top-k) | ROZNE | IDENT | inna kolejnosc redukcji w FA vs sciezka rzadka -> domyslnie OFF |
| E6b (shexp tail) | ROZNE | ROZNE | przyczyna nieznana (arytmetyka identyczna na papierze) -> domyslnie OFF, do zbadania (E19: roznica logitow) |
| E7 (prolog GDN) | IDENT | IDENT | OK |
| E6d (dedup Q8) | IDENT | IDENT | OK |
Commit 2649d89+: E3/E6b domyslnie OFF. Zyski E14 (+23% proza) zawieraly E3 i E6b — powtorka w E17/E18.

### E15 (2026-09-03 00:10, czesciowo): E4 nalozone (55313bf), bramka CONV_SNAP ON/OFF bit-exact IDENTYCZNE.
Spis jader HEAD (E3/E6b OFF, E4, E7b): 2903/token vs 2479 w E14 (E3 OFF oddaje +204 binbcast, +24 mmvf, +24 rms_norm
na krotkim kontekscie; E6b OFF +48). E3 nie jest bit-exact, a w praktyce (prompty 2-5k > budzet top-k 2048) i tak
nieaktywne -> zostaje OFF. A/B CONV_SNAP w toku.

### Typy tensorow hc/GDN (z GGUF, warstwa 2) — do projektu E10
hc_attn_down Q8_0 [10240->320], hc_attn_up Q8_0 [320->10240], hc_attn_inject F32 [10240->4], hc_attn_norm F32 [10240];
ssm_alpha/ssm_beta F32 [2560->48], ffn_gate_inp F32 [2560->512], ffn_gate_inp_shexp F32 [2560->1]; attn_qkv/attn_gate Q8_0.
Wniosek: mul_mat_vec_f (276/token) = inject x2 + alpha + beta + gate_inp + gate_inp_shexp (6/warstwe, wagi F32).
E10 kandydaci (bit-exact realne, bo wagi F32 i replikacja petli mmvf):
- E10a: inject wciagniety do hc-combine (4 iloczyny skalarne 10240 na token) -> -96/token.
- E10b: alpha|beta jako jeden matvec F32 [2560->96] sklejony przy ladowaniu; GDN czyta g/beta z krokiem -> -36/token.
- E10c (duzy): hc_down(Q8_0)+scale_silu+hc_up(Q8_0)+sigmoid+mix w jednym jadrze -> -384..-576/token, replikacja mmvq (dp4a) trudniejsza.
E15 A/B (CONV_SNAP, MTP, OD23, cieply cache): OFF 45,45/50,45, ON 45,40/50,00, OFF2 44,52/49,77, ON2 45,45/50,60 (proza/kod t/s)
-> E4 neutralne (w szumie), bit-exact, zostaje ON. HEAD (E3/E6b OFF, E4, E7b): proza ~45,4, kod ~50,3.

### E17 (2026-09-03 00:18): PR-src vs PR+latki vs fuzje HEAD (55313bf+), MTP n-max 3, OD23, ctx 8192, cieply cache, POWT 6
| build | proza | kod |
|---|---|---|
| SRC (PR 28243 czysty) | 45,15 / 45,11 | 43,21 / 43,36 |
| LATKI (small-K, nwarps=2, SWAR) | 39,60 | 42,97 |
| FUZJE (latki + wszystkie etapy bit-exact) | 45,40 | 50,67 |
| FUZJE bez small-K (env) | 45,18 | 50,58 |
Wnioski: (1) latki bez fuzji SZKODZA prozie pod MTP (-12%); small-K nie jest winne -> nwarps=2 albo SWAR (E21: revert).
(2) fuzje vs czysty PR: kod +17%, proza +0,5% -> proza pod MTP jest ograniczona czyms innym niz liczba jader
(akceptacja draftu / eksperci na CPU?) — do sprawdzenia: tg bez MTP oraz statystyki akceptacji.
KOREKTA E17: akceptacja draftu (proza) SRC 0,70 vs LATKI 0,52 vs FUZJE 0,58 — latki NIE sa bit-exact z czystym PR
(SWAR/nwarps/small-K zmieniaja kolejnosc redukcji), wiec generowany tekst rozjezdza sie i statystyka akceptacji
(a z nia t/s) jest inna. Porownania t/s z MTP miedzy buildami o roznej numeryce sa niemiarodajne; miarodajne sa
tylko miedzy buildami bit-exact (latki vs fuzje: tekst identyczny) albo bez MTP. E21 przerobione: tg bez MTP.
Wniosek praktyczny: liczy sie fuzje vs latki (ten sam tekst): proza 39,6 -> 45,4 (+15%), kod 43,0 -> 50,7 (+18%).

### E18 (2026-09-03 00:21, czesciowo): E7b NIE bit-exact (proza ROZNE; kod IDENT) — roznica w l2norm q/k w jadrze GDN
(prawdopodobnie kontrakcja FMA w norm.cu `tmp += xi*xi`). Dodany przelacznik GGML_JOHNV8_GDN_L2_FMA (1=fmaf, 0=rn) -> E22.
Statystyki E6d z 64 tokenow: hits 3951 / misses 27783 = ~62 trafien/token = pary attn_qkv|attn_gate (Q8_0, 48/warstwe)
+ kilka; alpha/beta/gate_inp/inject to wagi F32 (mul_mat_vec_f, bez kwantyzacji) -> E6d dziala zgodnie z mozliwosciami.
Pozostale 434 kwantyzacje/token: hc_down, hc_up (x2), attn_qkv, ssm_out, shexp gate|up, shexp down, MoE — po jednym wejsciu.
E18 A/B (GDN_L2 ON vs OFF, MTP): ON 45,4/50,6, OFF 42,3/45,8 — ale E7b NIE jest bit-exact (proza ROZNE), wiec roznica
moze byc czesciowo artefaktem akceptacji draftu (inny tekst). Rozstrzygniecie: E22 (FMA=1/0 bramki) + A/B BEZ MTP.

### E19/E20 (2026-09-03 00:40)
- E19: E6b (ogon shexp) zmienial logproby o 0,2-1,5 nata — nie zaokraglenie. Test w izolacji (tail_variants.hip):
  __fmul_rn/__fadd_rn na HIP gfx1201 daja INNE bity niz a*b/a+b w 15% elementow; zwykle operatory pod
  `#pragma clang fp contract(off)` = 0 roznic vs 3 jadra stocku. Poprawione (helpery johnv8_mul_nc/add_nc) w E6b, E7, E7b;
  E6b znow domyslnie ON; bramki w E22.
- E20: E10a (inject w combine) bit-exact IDENT, ale spis jader bez zmian — fuzja nie odpala w realnym grafie
  (podejrzenie: alokator nakłada wyjscie combine na hc_norm, bo w stocku inject liczy sie przed ADD; matcher
  poprawnie odrzuca nakladanie). Dodana diagnostyka; jesli to nakladanie -> ggml_set_output na hc_norm.
- E18/E20 pokazaly tez: A/B z MTP dla etapow NIE bit-exact skacze o +-7% przez zmiane tekstu/akceptacji
  (E7b: 45,4/50,6 vs 42,6/46,0 miedzy dwoma wariantami). Tylko pary bit-exact albo pomiary bez MTP.

### E21 (2026-09-03 00:38): tg128 BEZ MTP (niezalezne od tekstu), OD23, ctx 8192, 2 powtorzenia
| build | proza | kod |
|---|---|---|
| SRC (czysty PR) | 26,66 / 26,78 | 26,73 / 26,55 |
| LATKI | 29,68 / 29,55 (+11%) | 29,28 / 29,29 (+10%) |
| FUZJE HEAD (E7b ON, E10a ON-nieaktywne) | 32,24 / 32,41 (+21% vs SRC, +9% vs latki) | 32,11 / 32,22 |
Latki sa dobre (+11%), fuzje dokladaja +9% bez MTP; z MTP (para bit-exact latki->fuzje): +15% proza, +18% kod.

### E22 (2026-09-03 00:47): po usunieciu __fmul_rn/__fadd_rn
- E7b (l2norm q/k w GDN) z fmaf: IDENTYCZNE (proza i kod) -> bit-exact, domyslnie ON (GDN_L2_FMA=1).
- E6b (ogon shexp) ze zwyklymi operatorami pod contract(off): IDENTYCZNE -> bit-exact, domyslnie ON.
- CALE DRZEWO fuzje HEAD (061a95a+) vs PR+latki: IDENTYCZNE na obu promptach.
- A/B bez MTP L2 ON/OFF: 32,57/32,24 vs 32,50/32,30 -> E7b w szumie (0%). -72 jader/token nie daje mierzalnego zysku;
  koszt jadra zalezy od rodzaju, nie tylko od liczby (E4 tez neutralne). Wniosek: celowac w jadra "drogie" (matvec, norm,
  synchronizacje), nie w najtansze elementwise.
- Spis jader z MTP przez llama-completion nie zadzialal (do sprawdzenia flagi spekulacji w llama-completion).

### E24 (2026-09-03 00:55, czesciowo): E10c (blok hc w 2 jadrach)
- Test w izolacji (losowe dane; potem PRAWDZIWE wagi blk.2/blk.30, amplituda 30, nt=1/4): mixed/xn/lo/inject BIT-EXACT.
- Serwer: HCBLOCK ON vs OFF ROZNE (proza rozjazd po 96 znakach, kod inne od 2. znaku) -> roznica jest, ale mala
  (rozjazd w miejscu bliskiego remisu) i wystepuje TYLKO w pelnym modelu (alokator/scheduler/interakcja?). E26/E27 izoluja.
- Spis jader: 2903 -> 2273/token (-629 = -22%): hcblock_a 99, hcblock_b 98; znikaja mmvq -187, quantize -188, binbcast -191,
  rms_norm -95, scale_silu/hc_mix.
- A/B BEZ MTP: ON 33,12/32,91 vs OFF 32,67/32,59 -> tylko +1,4%/+1,0% za -22% jader!
WNIOSEK (wazny): koszt "na jadro" nie jest staly. hc-mix/combine (-1071) daly +9% bo usunely przebiegi po pamieci
[10240 x nt]; E10c usuwa male jadra obliczeniowe (matvec 3 MB, quantize) i zastepuje je wlasnymi o podobnym czasie.
Pozostale ~15 ms/token (z ~30 ms bez MTP) to prawdopodobnie NIE launche, tylko: 25 warstw ekspertow na CPU
(pasmo DDR4 + ~50 granic splitow GPU<->CPU na token z synchronizacja i budzeniem watkow). E28: watki/poll/slope OD.
E24 A/B Z MTP: ON 40,0/51,3 (2: 41,1/52,3) vs OFF 43,6/47,4 (2: 43,8/47,3) -> sprzeczne kierunki (proza -8%, kod +8%)
= artefakt rozjazdu tekstu (E10c jeszcze nie bit-exact w modelu). Bez MTP: +1,4%. Wniosek: E10c warte ~1-2%, nie 10%.

### E26 (2026-09-03 01:08): E10c w modelu — HCBLOCK=2 (tylko A), =3 (tylko B), =1 (oba) vs 0: wszystko IDENTYCZNE (proza i kod).
Czyli ROZNE z E24 (ten sam kod jader) nie powtorzylo sie -> podejrzenie przejsciowego wyscigu albo artefaktu srodowiska.
E27 (logity) i E31 (3 powtorzenia bramki) maja to rozstrzygnac; do tego czasu E10c zostaje domyslnie ON, ale z flaga "obserwowac".

### E27 (2026-09-03 01:10): logity HCBLOCK=1/2/3 vs 0 — IDENTYCZNE roznice we wszystkich trzech trybach (proza/kod: max|dlogprob|
0,4-1,2; prompt 'def fib': 0). Skoro "tylko A" i "tylko B" daja te same odchylenie, to nie jadra sa winne, tylko wspolny element:
ggml_set_output(lo) (pin). Pin zmienia alokacje calego grafu i ujawnia gdzies wyscig/aliasing (E24 ROZNE vs E26 IDENT przy tym samym
kodzie = niedeterminizm). Decyzja: pin domyslnie OFF (commit); bez pinu matcher B odrzuca fuzje, gdy `mixed` nachodzi na `lo`.
Do zbadania osobno: ktora fuzja ma hazard przy aliasowaniu (kandydat: hc_combine bez sprawdzenia add vs block_out/inject).

### E28 (2026-09-03 01:10): bez MTP, OD23, ctx 8192 — watki/poll/warstwy CPU
| wariant | proza | kod |
|---|---|---|
| t48 | 33,41 / 33,64 | 33,27 / 33,17 |
| t32 | 33,68 | 33,17 |
| t24 | 34,17 | 33,66 |
| t48 --poll 100 | 33,55 | 33,35 |
| t48 --poll 0 | 33,59 | 33,05 |
| t48 OD26 (22 warstwy na CPU) | 34,66 | 34,33 |
| t48 OD29 | OOM VRAM przy ctx 8192 | |
Wnioski: poll bez znaczenia; mniej watkow = odrobine szybciej (24 > 32 > 48); 3 warstwy mniej na CPU = +1,2 t/s
-> ~0,33 ms/token na warstwe CPU -> 25 warstw ~8 ms z ~30 ms/token. GPU + narzuty to ~22 ms/token.

### Analiza czasu jader z trace E24 (bez MTP, pod profilerem ~26 t/s): na token suma czasu jader 17,8 ms, rozpietosc 37,7 ms
-> ~20 ms LUK (GPU bezczynne). Jadra: mmvq 7,55 ms (302 x 25 us, strumieniowanie wag ~pasmo), mmvf 2,47, hcblock_a 1,79
(97 x 18,5 us!), hcblock_b 1,21 (96 x 12,6 us), GDN 0,52, FA 0,45, quantize 0,42, rms 0,35, topk 0,35, binbcast 0,31.
Wnioski: (1) moje jadra E10c sa wolne (kazdy z 40 blokow liczy norme i kwantyzacje od nowa) — zjadaja zysk z usunietych
launchy; (2) luki 20 ms = eksperci na CPU (~8 ms wg E28) + ~12 ms narzutu kolejkowania po stronie CPU: petla eval
przechodzi ~4800 wezlow na token i dla kazdego odpala kilkanascie matcherow fuzji (try_fuse) — to koszt CPU rzedu
2-5 us/wezel. HIPOTEZA GLOWNA: dekod jest CPU-bound (enqueue), nie GPU-bound. Sprawdzenie: perf record na serwerze (E33);
lekarstwo: pamiec podreczna decyzji fuzji per graf (graph reuse -> te same wezly co token) + odchudzenie petli eval.

### E29 (2026-09-03 01:15): hosting (ctx 65536, 2 sloty, KV q8_0, MTP n-max 3), drzewo fuzje HEAD
| konfiguracja | VRAM card1 | proza | kod |
|---|---|---|---|
| OD22, ub 1536 | 32517 MiB | 45,18 | 51,27 |
| OD22, ub 512 | 31221 MiB | 45,23 | 51,70 |
| OD24/25/26, ub 512 | OOM przy ladowaniu | | |
ub 512 oszczedza 1,3 GB bez straty predkosci; jedna warstwa ekspertow ~1,05 GB -> OD23 z ub 512 moze sie zmiescic (+0,4 t/s), E34.

### Launcher (2026-09-03 01:17): przygotowany `start_flashnext_fuzje.sh` (drzewo fuzje, ub 512, -t 24, reszta jak best):
NIE uruchomiony — czeka na E31 (powtarzalnosc bit-exact bez pinu; 2/3 IDENT juz jest), E32 (watki z MTP) i E34 (OD23 przy ub 512).

### E31 (2026-09-03 01:20): E10c BEZ pinu — 3x bramka 0 vs 1 IDENTYCZNE, kontrola IDENTYCZNE, logity 0 vs 1: 0,000 (3 prompty x 4 kroki).
Ale bez pinu jadro B odpala tylko 16/96 razy (alokator naklada `mixed` na `lo` -> matcher B odrzuca); spis 2386/token.
Poprawka (commit): jadro A pisze kopie `lo` do bufora kontekstu, jadro B czyta z niego -> nakladanie nie szkodzi. Walidacja: E36.

### E32 (2026-09-03 01:25): liczba watkow CPU, OD23, ctx 8192 (HEAD z E10c bez pinu)
| watki | bez MTP proza/kod | z MTP proza/kod |
|---|---|---|
| 16 | 32,53 / 32,37 | 37,08 / 40,02 |
| 20 | 33,48 / 33,32 | - |
| 24 | 34,07 / 33,81 | 40,54 / 43,75 |
| 32 | 33,74 / 33,17 | 41,11 / 44,37 |
| 48 | (E28: 33,4-33,6) | 41,51 / 44,56 |
Bez MTP optimum 24 watki (+2%), z MTP 48 (paczka T=4 na CPU = 4x wiecej pracy ekspertow). Hosting z MTP: zostaje -t 48.
UWAGA: te liczby z MTP (41,5/44,6) sa nizsze niz E17 (45,4/50,7) — inny stan drzewa (E10c ON bez pinu) i inny tekst; czysty A/B
E10c ON/OFF z MTP na tym samym buildzie dopiero w E36.

### Czas jader Z MTP (trace E25, HEAD z pinem): na krok suma jader 32,6 ms, rozpietosc 66,5 ms (luki 34 ms).
hcblock_a 55 us x 100 = 5,5 ms/krok (!), hcblock_b 25 us x 101 = 2,6 ms; mmvq 10,6 ms, mmvq_moe 3,7, mmvf 2,4, mul_mat_f 1,1, GDN 1,0.
E10c pod MTP SZKODZI (~ -4 ms/krok ~ -8%): dla nt=4 jadro A ma tylko 10 blokow (konfiguracja mmvq 2 warpy x 2 wiersze x 16
wirtualnych blokow = 32 wierszy/blok) i kazdy blok liczy normy+kwantyzacje 4 tokenow od nowa. Decyzja: HCBLOCK domyslnie OFF
(commit). Refaktor: A0 (normy+kwantyzacja, 1 blok/token -> bufor kontekstu), A1 (matvec+silu, bloki 256 watkow jak stock,
ostatni blok kwantyzuje lo), B1 (matvec+sigmoid+mix, 320 blokow po 8 elementow) -> 3 launche zamiast 7, pelna rownoleglosc.

### E34 (2026-09-03 01:30): hosting OD23 + ub 512 (2 sloty x 64k, MTP): miesci sie, VRAM 32153/32768 MiB (98% — malo zapasu),
40,9/43,9 t/s przy t24 z E10c ON (wolne jadro A) -> liczby nieporownywalne; decyzja: launcher zostaje na OD22 (zapas VRAM), OD23 opcjonalnie.
E33 (perf): perf bez sudo padl (paranoid=4), fallback sudo tez (plik utworzony bez praw) -> powtorka z sudo w E37 + instrumentacja czasu CPU.

### E35 (2026-09-03 01:33): E10a po poprawce (use_count zamiast can_fuse_subgraph): odpala we wszystkich 96 blokach
(mmvf -95, hc_combine -95 -> hc_combine_inject 96), bit-exact IDENTYCZNE. A/B bez MTP: ON 33,77/33,68 i 33,68/33,44 vs
OFF 33,63/33,45 i 33,69/33,38 -> ~+0,3% (szum). Zostaje ON (nieszkodliwe), zysk pomijalny.

### E36 (2026-09-03 01:40): E10c z kopia `lo` w buforze kontekstu — harness (prawdziwe wagi, nt 1/4) 0 roznic, 2x bramka IDENT,
logity 0 vs 1: brak roznic. A/B bez MTP (t24): ON 34,35/34,17 i 34,29/34,15 vs OFF 33,65/33,34 i 33,63/33,50 -> +2,1%/+2,4%.
Z MTP nadal ujemne (jadro A 55 us przy nt=4) -> domyslnie OFF; refaktor A0/A1/B1 (3 launche, pelna rownoleglosc) w planie.

### E37 (2026-09-03 01:45): koszt CPU petli eval + perf
- TIMING (bez MTP, t24): 85 wezlow/graf (graf = split; ~56 splitow/token), 27 fuzji/graf, try_fuse 1,4 us/wezel (~7 ms/token),
  compute_forward 9-13 us/launch (!) -> ~2300 launchy x ~11 us = ~25 ms/token po stronie CPU — tyle co caly token (30 ms).
  Dekod jest na granicy CPU-enqueue i GPU jednoczesnie: GPU zajete 17,8 ms + warstwy CPU ~8 ms, CPU kolejkowanie ~25 ms + matchery 7 ms.
- perf (sudo): 74% probek w libgomp (watki OpenMP backendu CPU krecace sie w oczekiwaniu), 10,8% ggml_vec_dot_iq2_s_q8_K
  (wlasciwa praca ekspertow). Watki spinujace konkuruja z watkiem glownym o rdzenie -> stad 24 watki > 48 bez MTP.
Wnioski/plan: (1) E11 plan fuzji (E38) — zdejmuje ~7 ms CPU/token, jesli dekod jest CPU-bound to duzy zysk;
(2) OMP_WAIT_POLICY/GOMP_SPINCOUNT (E40) — mniej spinowania = wolniejszy watek glowny? do zmierzenia;
(3) koszt 11 us/launch po stronie HIP — sprawdzic sciezke ggml_cuda_kernel_launch (atrybuty PDL/hipLaunchKernelExC) i zredukowac.

### E38 (2026-09-03 01:50): plan fuzji v1 — bit-exact IDENT, graphs reused = 94/96 (reuse grafu dziala), ale plan nigdy nie trafial
(pominiete=0): scheduler wola graph_compute osobno dla kazdego splitu (inny wskaznik cgraph) i plan byl resetowany co split.
Poprawka: mapa planow po wskazniku cgraph (per split). Pomiar: E41.
E38 A/B (plan v1 nieaktywny, HEAD: E10a ON, E10c OFF): bez MTP t24 ~33,7/33,4; z MTP t48: ON 43,8/47,7 i 44,3/47,6 vs OFF 43,8/47,4 i 44,0/47,7
(= stan odniesienia dla E41).

### E39 (2026-09-03 01:55): E10c v2 (A0/A1/B1) — bit-exact (harness nt 1/2/4, 2x serwer). Spis: 2386 -> 2228/token.
Czasy jader (bez MTP): A0 12,2 us, A1 9,3 us, B1 10,9 us (razem 3,1 ms/token ~ tyle co stockowy lancuch).
A/B bez MTP (t24): ON 34,03/33,73 i 34,00/33,73 vs OFF 33,71/33,55 i 33,80/33,55 -> +0,8%.
A/B z MTP (t48, bit-exact, ten sam tekst): ON 43,44/47,07 i 43,45/47,18 vs OFF 43,99/47,72 i 44,08/47,50 -> -1,3%.
Wniosek: E10c nie daje zysku pod MTP nawet w wersji rownoleglej; zostaje OFF. Ostatnia proba (E42): A0 po jednym bloku na
(token, strumien) zamiast na token, B1 w jednym przebiegu (1024 watkow). Jesli nadal <= 0 z MTP -> zamykamy E10c.

### E40 (2026-09-03 02:05): polityka oczekiwania OpenMP (backend CPU), OD23, ctx 8192
| wariant | bez MTP t24 proza/kod | z MTP t48 proza/kod |
|---|---|---|
| default | 33,67 / 33,47 | 44,05 / 47,56 |
| OMP_WAIT_POLICY=passive | 26,14 / 26,24 (-22%) | 37,59 / 41,08 (-15%) |
| GOMP_SPINCOUNT=20000 | 32,33 / 32,39 | 43,27 / 47,33 |
| OMP_WAIT_POLICY=active | 34,39 / 34,31 (+2,3%) | (E43) |
| t32 + passive | - | 38,08 / 41,00 |
Wniosek: opoznienie budzenia watkow na granicach splitow ma znaczenie; active (ciagle spinowanie) daje +2% bez MTP. E43: active z MTP.

### E41 (2026-09-03 02:10): plan fuzji per split — pominiete 179583 / sprawdzone 58589 (dziala), try_fuse 1,39 -> 0,79 us/wezel,
compute_forward 9,2 -> 7,6 us/wezel, ALE bramka [d] ROZNE i bez MTP -2% (33,06/32,44 vs 33,80/33,37); z MTP proza rowne, kod +14% (artefakt tekstu).
Diagnoza: klucz planu = wskaznik wezla; graf promptu (nt=20) i dekodu (nt=1) wspoldziela adresy wezlow (reuse kontekstu grafu),
wiec decyzje "brak fuzji" z promptu (np. mmvq gate+up fuzja tylko dla nt<=8) byly stosowane w dekodzie -> pominiete fuzje
(wolniej) i inna numeryka tam, gdzie fuzja upstream nie jest bit-exact ze sciezka rozdzielna. Poprawka: sygnatura wezla
(op, ne[0..3], src0, src1) w planie. Pomiar: E44.

### E42 (2026-09-03 02:20): E10c v3 (A0 per strumien 4,0 us, A1 9,5 us, B1 na 1024 watkach 24,3 us) — bit-exact, ale wolniej:
bez MTP ON 33,36/32,91 i 33,30/33,16 vs OFF 33,75/33,42 i 33,78/33,57 (-1,3%); z MTP ON 43,4/46,6 i 43,5/46,9 vs OFF 44,2/47,6 i 44,3/47,6 (-2%).
B1 cofniete do 256 watkow (v3b, E45). Jesli v3b tez <= 0 z MTP -> E10c zamykamy (kod zostaje, domyslnie OFF).

### E43 (2026-09-03 02:27): OMP_WAIT_POLICY=active z MTP (OD23, ctx 8192)
| wariant | proza | kod |
|---|---|---|
| t48 default | 44,12 / 44,33 | 47,54 / 47,53 |
| t48 active | 44,18 / 43,97 | 47,97 / 48,42 |
| t32 active | 44,93 | 48,66 |
| bez MTP t24 active | 34,42 | 34,41 |
| bez MTP t32 active | 34,29 | 34,00 |
Wniosek: active + 32 watki = +2% z MTP (najlepsza konfiguracja CPU); bez MTP t24 active +2,5%. Launcher: OMP_WAIT_POLICY=active, -t 32.

### E44 (2026-09-03 02:35): plan fuzji z sygnaturami — nadal [d] ROZNE (2x) i zero zysku (bez MTP 33,6-33,8 vs 33,7; z MTP 43,9-44,1 vs 44,0-44,1);
try_fuse 1,39 -> 1,14 us/wezel. Wniosek: matchery fuzji NIE sa waskim gardlem, a pomijanie ich zmienia numeryke (jakas decyzja
fuzji nie jest stala miedzy ewaluacjami — nie warto szukac). E11 ODRZUCONE (kod zostaje, domyslnie OFF).
Bilans czasu na token (bez MTP, ~29 ms): jadra GPU ~17,8 ms + ~2200 launchy x ~4-5 us luki GPU (~10 ms) + warstwy CPU ~8 ms
(czesciowo nakladaja sie). Dalsze zyski: mniej launchy (kazde -100 ~ +1,5%), szybsze duze jadra (mmvq 7,5 ms - blisko pasma),
mniej warstw na CPU (VRAM). Nastepny krok E12: kwantyzacja Q8_1 wewnatrz mmvq (usuwa 241 launchy quantize_q8_1/token, bit-exact).
E12 (kwantyzacja w mmvq) — analiza: mmvq startuje po jednym bloku na wiersz (np. attn_qkv: 10240 blokow), kazdy blok musialby
kwantyzowac cale y od nowa (10240 x 2560 elementow) -> duzo drozej niz 2 us jadra quantize + luka. ODRZUCONE bez implementacji.
Do sprawdzenia zamiast tego: HIP graphs na drzewie fuzji (wczesniej "zero zysku" mierzone na starym forku; teraz 2x mniej launchy,
ale E6d/E10c alokuja leniwie cudaMalloc -> przy grafach wylaczyc Q8_DEDUP). E46.

### E45 (2026-09-03 02:50): E10c v3b (A0 per strumien 4,0 us, A1 9,5 us, B1 256 watkow 10,9 us) — bit-exact (harness + 2x serwer)
| | proza | kod |
|---|---|---|
| bez MTP t24 ON | 34,90 / 34,87 | 34,52 / 34,59 |
| bez MTP t24 OFF | 33,72 / 33,73 | 33,61 / 33,43 |
| z MTP t48 ON | 44,50 / 44,23 | 48,17 / 47,74 |
| z MTP t48 OFF | 44,05 / 44,13 | 47,60 / 47,72 |
+3,4%/+2,9% bez MTP, +0,6% z MTP -> E10c v3b DOMYSLNIE ON (commit). Suma czasu jader 17,16 ms/token (najnizsza dotad).

### E46 (2026-09-03 02:55): HIP graphs na drzewie fuzji (OD23, ctx 8192, OpenMP active)
| | proza | kod |
|---|---|---|
| bez MTP t24, grafy OFF | 34,36 | 34,31 |
| bez MTP t24, grafy ON (Q8_DEDUP=0) | 35,52 | 35,51 |
| bez MTP t24, grafy ON (dedup ON) | 35,63 | 35,75 |
| z MTP t32, grafy OFF | 44,74 | 48,42 |
| z MTP t32, grafy ON | 45,63 | 49,39 |
HIP graphs daja +3,4% bez MTP i +2% z MTP (wczesniejsze "zero zysku" bylo na starym forku z 2x wiecej launchy). E6d z grafami dziala.
Do zrobienia: bramka bit-exact z grafami (E48), potem grafy ON w launcherze.

### E47 (2026-09-03 03:00): KANDYDAT DO HOSTINGU — hosting ctx 65536 x 2 sloty, ub 512, OD22, MTP n-max 3, t32, OMP active, bez HIP graphs
| build | proza | kod | VRAM |
|---|---|---|---|
| fuzje HEAD 6735ff5 (E10c ON) | 49,68 / 49,61 | 52,82 / 52,91 | 31197 MiB |
| PR+latki | 45,91 / 45,45 | 48,67 / 48,27 | 31177 MiB |
| fuzje t48 default (jak stary launcher) | 48,54 | 51,74 | |
Bramka latki vs HEAD: IDENTYCZNE. Zysk bit-exact: +8,4% proza, +8,6% kod (ten sam tekst). Do tego HIP graphs (E48) ~+2%.

### E48 (2026-09-03 03:05): HIP graphs — bramki: latki(grafy) vs fuzje(grafy) IDENT, fuzje grafy ON vs OFF IDENT, fuzje(grafy) vs latki(bez) IDENT.
Hosting (2x64k, ub 512, OD22, t32 active): grafy ON 50,50/53,72 i 50,50/53,51 vs OFF 48,95/52,36 -> +3%.
WYNIK KONCOWY hostingu: 50,5 proza / 53,6 kod t/s (bit-exact z PR+latki, ktore w tej konfiguracji daja 45,9/48,7; stary launcher ~45/51).

### E49 (2026-09-03 03:05): serwer roboczy 8092 (GPU1) wystartowany z `start_flashnext_fuzje.sh` (CTX 32768 x 2 sloty, ub 512, OD22,
MTP n-max 3, t32, OMP active, HIP graphs ON, pid 4122926). Sanity z timings serwera: proza 50,7 t/s (draft 105/161), kod 53,8 t/s
(draft 108/153), VRAM card1 94%.

### E50 (2026-09-03 10:30-10:38): 2x GPU (ROCm0+ROCm1, ts 1:1.2, tylko engram na CPU), ctx 65536 x 2 sloty, ub 512, MTP n-max 3, t32, OMP active
Na czas pomiaru obie karty byly wolne; serwer testowy 8092 przywrocony po sesji.
| build | KV | grafy | pp2048 t/s | tg proza | tg kod | akceptacja draftu |
|---|---|---|---|---|---|---|
| fuzje HEAD | f16 | ON | 577 | 62,79 / 61,47 | 67,00 / 66,23 | 0,68 / 3,02 |
| fuzje HEAD | q8_0 | ON | 584 | 62,62 / 61,49 | 50,94 / 49,95 | 0,44 / 2,31 (inny tekst!) |
| fuzje HEAD | f16 | OFF | 571 | 60,15 | 63,86 | 0,68 |
| PR+latki | f16 | OFF | 584 | 53,49 / 53,31 | 56,84 / 57,20 | 0,68 |
| PR+latki | f16 | ON | - | 55,55 | 59,84 | 0,68 |
Wnioski: fuzje vs latki (oba bez grafow, ten sam tekst): +12,4%; HIP graphs +4-5%; razem 53,5/56,8 -> 62,8/67,0 (+17%).
KV q8_0: proza identyczna (62,6 vs 62,8) i prefill identyczny -> KV q8_0 nie kosztuje przepustowosci; nizszy "kod" to rozjazd
tekstu (q8_0 nie jest bit-exact z f16: akceptacja draftu 0,44 vs 0,68 na tym prompcie), nie koszt jadra. Prefill ~580 t/s
(pp2048) niezaleznie od buildu — prefill nie jest launch-bound. Wczesniejszy pomiar 2x GPU (53,0/56,9, PR+latki) potwierdzony.

### Zestawienie 1x vs 2x GPU (2026-09-03 10:45)

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

### E51a (2026-09-03 12:15-12:25): 1x GPU hosting (GPU1 + 26 warstw na CPU, ctx 65536 x 2, ub 512, KV q8_0, t32, OMP active)
| build | MTP | pp1024 | pp2048 | tg proza | tg kod |
|---|---|---|---|---|---|
| fuzje HEAD (ROCm, grafy) | bez | 372 | 390 | 36,03 / 35,95 | 35,98 / 35,87 |
| fuzje HEAD (ROCm, grafy) | n-max 3 | (378*) | (416*) | 50,5 (E48) | 53,6 (E48) |
| unsloth b10715-mix (Vulkan, swiezy z GitHub, najnowszy) | n-max 3 | 173 | 176 | 40,70 / 40,00 | 43,85 / 44,24 |
| unsloth b10715-mix (Vulkan) | bez | 177 | 184 | 29,03 / 30,14 | 28,54 / 29,34 |
*prefill fuzje z MTP: z serwera 8092 (/completion, mediana 3). Akceptacja draftu unsloth 0,68/3,02 = jak nasza (ten sam tekst).
Fuzje vs unsloth: z MTP +24%/+22% tg, prefill x2,2; bez MTP +22%/+24% tg, prefill x2,1.

### E51b (2026-09-03 12:25-12:34): 2x GPU (ctx 65536 x 2, ub 512, KV f16, ts 1:1,2, t32 active); obie karty wolne
| build | MTP | pp1024 | pp2048 | tg proza | tg kod |
|---|---|---|---|---|---|
| fuzje HEAD (ROCm, grafy) | n-max 3 | 552 (E50) | 577 (E50) | 62,79 (E50) / 61,19 | 67,00 (E50) / 65,82 |
| fuzje HEAD (ROCm, grafy) | bez | 611 | 622 | 38,81 / 39,03 | 38,91 / 39,05 |
| unsloth b10715-mix (Vulkan0+1) | n-max 3 | 595 | 690 | 50,42 / 49,34 | 59,91 / 59,00 |
| unsloth b10715-mix (Vulkan0+1) | bez | 688 | 894 | 45,50 / 45,67 | 45,50 / 45,46 |
WAZNE: na 2x GPU backend Vulkan (unsloth) BEZ MTP jest szybszy od naszego HIP bez MTP (45,5 vs 39,0 tg; prefill 894 vs 622);
z MTP wygrywamy (62/66 vs 50/60), bo nasz tor weryfikacji T=4 + fuzje sa tansze. Na 1x GPU z offloadem CPU wygrywamy wszedzie.
Wniosek na przyszlosc: zbadac, czemu HIP dekod bez MTP przegrywa z Vulkanem na RDNA4 (mmvq/FA?) — potencjal +15% takze z MTP.

### E52 (2026-09-03 12:40): profil jader HIP, GPU1 + 26 warstw na CPU
PREFILL pp2048 (4 ubatche x 512): 31626 jader, suma 3269 ms: mul_mat_q 1933 ms (3780 x 511 us) = 59%!, gated_delta_net 380 ms
(359 x 1,06 ms = 12%), hipBLASLt GEMM f32 281 ms, flash_attn_tile 126 ms (72 x 1,75 ms), mm_ids_helper 87, quantize_mmq 57, rms 50...
DEKOD bez MTP: 2226 jader, suma 17,1 ms, rozpietosc 38,5 ms -> luki 21 ms (w tym ~8 ms warstw CPU, reszta ~13 ms to bariery
miedzy launchami). mmvq 7,5 ms (300 x 25 us).
Wnioski: prefill = MMQ (dp4a, bez WMMA na RDNA4?) + wolny GDN w prefillu (rekurencyjne jadro po tokenach) + FA tile (nie WMMA);
dekod = przerwy miedzy jadrami (bariera AQL po kazdym jadrze), ktorych Vulkan unika w jednym command bufferze. E53: eksperymenty env
(GGML_CUDA_FORCE_CUBLAS, KV f16 dla FA) na prefillu.

### E53 (2026-09-03 13:05): prefill 1x GPU (OD22, ub 512, bez MTP) — warianty env
| wariant | pp1024 | pp2048 | tg proza/kod |
|---|---|---|---|
| default (MMQ), KV q8_0 | 371 | 391 | 36,1 / 36,0 |
| GGML_CUDA_FORCE_CUBLAS=1 | 369 | 388 | 35,2 / 35,1 |
| KV f16 | 368 | 389 | 36,8 / 36,7 (+2%) |
| GGML_CUDA_FORCE_MMQ=1 | 362 | 386 | 35,8 / 36,1 |
Wniosek: na 1x GPU prefill jest ograniczony przez 26 warstw ekspertow liczonych na CPU (paczki 512 tokenow), wiec wybor jader GPU
nic nie zmienia; roznice HIP vs Vulkan w prefillu (622 vs 894) trzeba badac na 2x GPU (wszyscy eksperci na GPU). KV f16 daje +2%
dekodu na 1x GPU (FA vec na f16 szybsze niz na q8_0), kosztem VRAM.

### Synteza HIP vs Vulkan na RDNA4 (workflow 3 czytelnikow + synteza, 2026-09-03 13:10) — ranking kierunkow
1. Any-order launche z sledzeniem hazardow (hipExtLaunchKernelGGL + hipExtAnyOrderLaunch): HIP wypuszcza kazde jadro na jednym
   strumieniu z bariera AQL (pelne oproznienie poprzedniego), Vulkan wstawia bariery tylko przy realnym konflikcie zakresow bajtow
   (ggml-vulkan.cpp:15744-15766) i przestawia graf (ggml_vk_graph_optimize). Zysk szac. +8-15% dekodu, +5-10% prefillu; bit-exact; naklad: dni.
   MIKROBENCHMARK (anyorder.hip, gfx1201, ROCm 7.2.4): 4000 malych jader na jednym strumieniu: <<<>>> 2,96 us/jadro (puste) i
   16,9 / 59,0 us (jadra 2k/8k iteracji, pelna serializacja) vs hipExtAnyOrderLaunch 2,10 / 2,02 / 2,01 us — flaga DZIALA, niezalezne
   jadra nakladaja sie; dolna granica dispatchu ~2 us/jadro (CP). Warunek: analiza hazardow takze dla buforow z puli (quantize->mmvq).
2. Fork/join wielostrumieniowy (GGML_CUDA_GRAPH_OPT): domyslnie OFF, wychodzi przy 2 GPU (ggml-cuda.cu:4758), dopasowuje tylko
   'attn_norm' z fan-out 3 -> dla qwen4exp nigdy nie dziala. Patch: bramka per split, fan-out 2-6, nazwy hc_norm/hc_mixed/ffn_norm. +5-10%, bit-exact, dzien.
3. Prefill MMQ dla ekspertow: kafel J=128 dobierany z ncols_max=512 tokenow, a ekspert ma srednio ~10 wierszy -> ~10-13x zmarnowanej
   pracy WMMA/LDS (mmq.cuh:908-931, 1476-1500). Cap J przez env (5 linii) -> +15-30% prefillu na 2x GPU; bit-exact (kolejnosc po K bez zmian).
4. Flash-attention D=256: WMMA odciete warunkiem Q->ne[0] <= 128 (fattn.cu:514), 12 warstw QSA ida na skalarny tile; przy dlugim
   kontekscie duzo; NIE bit-exact (inna kolejnosc akumulacji) -> KL-floor.
5. mmvq: prefetch/unroll zachowujacy kolejnosc sum, kopia iq2s_grid do LDS (Vulkan: 2-4 wiersze/wg, pelny unroll). +0-8%, bit-exact.
6. Fuzja TOPK_QSA (radix top-k: 11 launchy x12 warstw + get_rows/permute/cont) jak w Vulkanie. +3-6%, bit-exact z uwaga na remisy.
7. Kwantyzacja q8_1 wciagnieta do producenta (rms_norm/mul/GLU). +2-5%, bit-exact jesli powieli drzewo redukcji.
8. Galki runtime ROCm (segment scheduling grafow, GPU_MAX_HW_QUEUES, spin wait). 0-5%.
Uczciwie: cala przewaga Vulkana bez MTP to ~3,7 ms/token (25,7 vs 22,0), wiec realny cel z 1-2 to ~42-45 t/s bez MTP; z MTP proporcjonalnie mniej.

## E54 — diagnostyka 2×GPU (obie karty wolne na czas pomiaru)

Profil (rocprof, 2×GPU, fuzje, bez MTP):
- prefill 2048 tok: 30959 jąder, suma jąder 3800 ms, rozpiętość 4437 ms → mul_mat_q 2251 ms (59 %, 3780 launchy × 596 µs), gated_delta_net 456 ms (12 %), hipBLASLt f32 GEMM 334 ms (9 %), flash_attn_tile 144 ms (4 %), mm_ids_helper 96 ms.
- dekod (40 tok.): 2242 jąder/token, suma jąder 21,0 ms, rozpiętość 32,3 ms → **luki między jądrami 11,3 ms/token** (bez warstw CPU; to koszt serializacji AQL, nie obliczeń). mmvq 10,4 ms (353 × 29 µs).
- `GGML_CUDA_GRAPH_OPT=1`: zero linii fork/stream → mechanizm nigdy nie odpala dla qwen4exp (jak przewidziano).

A/B dekod tg128 2×GPU, grafy ON, bez MTP (4 powtórki, proza/kod):

| wariant | proza | kod |
|---|---|---|
| baza | 40,05 | 40,02 |
| `GPU_MAX_HW_QUEUES=8` | **41,40** | **41,48** (+3,5 %) |
| `GGML_CUDA_GRAPH_OPT=1` | 40,05 | 40,08 (0) |
| baza2 (dryf) | 40,10 | 40,10 |

Vulkan (unsloth b10715, `GGML_VK_SYNC_LOGGER=1`): 45042 linii sync na prompt+16 tokenów — logger nie rozdziela prefill/dekod, wynik nierozstrzygający.

Test any-order pod capture grafu HIP (`anyorder_graph`, GPU1): replay grafu 16,2 µs/jądro **niezależnie od flagi** (plain 16,20 / anyorder 16,16); na strumieniu plain 16,9 vs any-order 3,05 µs. Wniosek: **hipExtAnyOrderLaunch nie przeżywa capture grafu** → any-order i grafy HIP wykluczają się; projekt any-order musi liczyć się z utratą +3–4 % z grafów i wygrać z tego poziomu.

## E56 — `GPU_MAX_HW_QUEUES=8` na 1×GPU (GPU1, config hostingu: ctx 64k×2, ub 512, KV q8_0, OD22, t32 active, grafy)

Bramka bit-exact ([k],[d]): IDENTYCZNE. A/B tg128 (4 powtórki):

| wariant | bez MTP proza/kod | z MTP n-max 3 proza/kod |
|---|---|---|
| baza | 36,50 / 36,56 | 50,33 / 53,59 |
| `GPU_MAX_HW_QUEUES=8` | 36,44 / 36,52 | 50,31 / 53,00 |
| powtórki (dryf) baza2 / hwq8_2 | 36,25 / 36,18 vs 36,30 / 36,24 | — |

Wniosek: na 1×GPU zero efektu (w szumie); zysk +3,5 % z E54 dotyczy tylko 2×GPU (dwa strumienie/dwa urządzenia, więcej kolejek HW). Do launchera 2×GPU tak, do 8092 (1×GPU) bez znaczenia.

## E57 — cap kafla J w MMQ mul_mat_id (patch E55, `GGML_JOHNV8_MMQ_ID_JMAX`), 1×GPU GPU1, hosting bez MTP

Mechanizm: upstream dobiera J po ne12 = wszystkie tokeny ubatcha (512 → J=128), a ekspert ma średnio ~10 wierszy; blok liczy vec_dot po całym J i tylko maskuje zapis. Cap ogranicza liczbę kolumn używaną wyłącznie do wyboru J (geometria launchu bez zmian). Bramki bit-exact JMAX=16 i JMAX=8: IDENTYCZNE ([k],[d]).

| wariant | pp2048 t/s | tg128 proza/kod |
|---|---|---|
| baza | 385,6 | 36,43 / 36,53 |
| JMAX=8 | 407,5 (+5,7 %) | 36,52 / 36,57 |
| JMAX=16 | 406,5 (+5,4 %) | 36,43 / 36,61 |
| **JMAX=32** | **420,3 (+9,0 %)** | 36,48 / 36,55 |
| baza2 (dryf) | 387,9 | 36,47 / 36,50 |

Dekod bez zmian (mmvq, nie MMQ). Na 2×GPU (MMQ = 59 % prefillu, wszystkie 48 warstw na GPU) spodziewany większy zysk — do zmierzenia w sesji 2×GPU. Do sprawdzenia jeszcze JMAX=64 (E59).

## E59 — cap J: JMAX=64 vs 32 (1×GPU GPU1, pp2048, bez MTP); bramki bit-exact JMAX=64 i 32: IDENTYCZNE

| wariant | pp2048 t/s |
|---|---|
| baza | 386,5 |
| JMAX=32 | **418,4 (+8,3 %)** |
| JMAX=64 | 404,5 / 411,2 (powt.) |

Wybór: **JMAX=32**. Wdrożone do `start_flashnext_fuzje.sh` (`GGML_JOHNV8_MMQ_ID_JMAX=32`), 8092 zrestartowany na buildzie z E55 (commit fb29fcd).

## E55 — patche z workflow implementerów (build-only, osobne worktree)

- **Cap J w MMQ** (`patches/e55_jcap.patch`, worktree wt-jcap) → zmierzony w E57/E59, wdrożony (commit fb29fcd, JMAX=32).
- **Fork/join dla qwen4exp** (`patches/e55_forkjoin.patch`, worktree <worktree fork/join>, commit 41a2e7a, build w wt-forkjoin/build-hip; domyślnie OFF): `GGML_JOHNV8_FORKJOIN=1` (nadrzędne wobec GGML_CUDA_GRAPH_OPT), `_NAMES=attn_norm,ffn_norm,hc_norm,hc_mixed`, `_FANOUT=2,6`, `_MAXROWS=16` (upstream liczył tylko nrows≤1 → wykluczało batche weryfikacji MTP), `_MERGE=1`, `_PULLBACK=2`, `_ENDJOIN`, `_NOGRAPH=1` (fork także bez grafów), `GGML_CUDA_DUMP_DISPATCH=1` (jednorazowy wypis forków/odrzuceń). Kontekst strumieni per split (klucz = wskaźnik cgraph), bez przestawiania węzłów (add_alloc_dep zamiast interleave), fuzje nie przekraczają root/join (klamra n_nodes wokół try_fuse), sprawdzenie zakresów write vs write/read między strumieniami (is_valid_strict), q8-dedup (E6d) wyłączony na strumieniach bocznych. Pomiar: E60 (1×GPU), potem 2×GPU.

## Plan any-order z hazardami (workflow: 2 projekty + sędzia) → `PLAN_ANYORDER.md`

Wybrany projekt konserwatywny (whitelista + odwołanie grantu u źródła: alokacja puli, q8cache, scratch hcblock, hipBLAS, operacje niekernelowe na strumieniu; tracker zakresów bajtów written/read od ostatniego launchu barierowego, per kontekst GPU; join-kernel na końcu splitu; tryb dry z licznikami). Fakty rozstrzygające: flaga any-order jest gubiona przy capture grafu (CLR 7.2: ihipExtLaunchKernel nie zapisuje flags; potwierdzone lokalnie) → any-order tylko przy grafach OFF, każdy etap A/B wobec bazy z grafami; reżim bez grafów jest bliski granicy CPU (~9 µs/węzeł × ~2200 ≈ 20 ms/token) → realny sufit raczej 25 → 20–22 ms/token (46–50 t/s bez MTP na 2×GPU), nie pełne 11–13 ms luk. Najpierw tanie eksperymenty: E0b (event/sync po any-order), E0c (kopie vs any-order), E0f (koszt CPU hipExtLaunchKernel), E0e (serwer z grafami OFF + TIMING: ściana vs suma jąder vs CPU) — dopiero potem etap 0 (infrastruktura + dry-run) i decyzja go/no-go.

## E58 — sesja 2×GPU (obie karty wolne na czas pomiaru): cap J + HW queues, KV f16, wszystko na GPU, 3 powtórki

| wariant | pp2048 t/s | tg128 proza/kod |
|---|---|---|
| baza (bez MTP) | 793,8 | 39,71 / 39,55 |
| JMAX=32 | **942,0 (+18,7 %)** | 39,16 / 38,87 |
| JMAX=32 + HWQ8 | 935,5 | 38,77 / 38,91 |
| JMAX=16 + HWQ8 | 873,7 | 38,82 / 38,95 |
| baza2 (dryf, koniec sesji) | 765,9 | 38,95 / 38,91 |
| MTP n-max 3: baza | 689,5 | 61,99 / 65,98 |
| MTP n-max 3: JMAX=32 + HWQ8 | **819,5 (+18,9 %)** | 61,95 / 66,06 |

Wnioski: cap J=32 daje na 2×GPU +19 % prefillu (MMQ = 59 % czasu prefillu), dekod bez zmian (różnice −1…−2 % mieszczą się w dryfie sesji: baza2 −3,5 % vs baza). `GPU_MAX_HW_QUEUES=8`: +3,5 % z E54 nie powtórzyło się (tu 0) → szum, nie wdrażać. **Prefill 2×GPU bez MTP 942 t/s > unsloth Vulkan 894**; dekod bez MTP 39,7 vs Vulkan 45,5 nadal w tyle (bariery AQL — fork/join E60, any-order wg planu).

## E60 — fork/join (patch E55) na 1×GPU (GPU1, build wt-forkjoin, hosting, grafy ON)

Sanity (`GGML_CUDA_DUMP_DISPATCH=1`): graf główny n_nodes=3787 → 47 forków, śr. 2,5 strumieni bocznych/fork, 94 regiony, 0 odrzuconych; grafy MTP (141 węzłów) po 1 forku. Bramki bit-exact (2×, NOGRAPH=1): IDENTYCZNE — poprawność OK.

| wariant | bez MTP proza/kod | z MTP proza/kod |
|---|---|---|
| baza | 36,22 / 36,40 | 49,31 / 50,57 |
| `GGML_JOHNV8_FORKJOIN=1` | **19,47 / 19,50 (−46 %)** | 45,06 / 47,68 (−8 %) |
| powtórki fj2 / baza2 | 19,37 / 19,54 vs 36,14 / 36,30 | — |

Wniosek: fork/join przez zdarzenia HIP (cudaEventRecord + cudaStreamWaitEvent na każdy fork i join, ~120 par/token) kosztuje na ROCm 7.2.4 znacznie więcej niż oszczędza (bariera AQL ~5 µs vs oczekiwanie na zdarzenie między kolejkami ~dziesiątki µs); to samo tłumaczy, czemu upstream gate'uje GGML_CUDA_GRAPH_OPT do 1 GPU i małych fan-outów. Ablacje w E62 (grafy OFF + NOGRAPH, tylko hc_mixed, mniejsze regiony) dla domknięcia kierunku.

## E62 — ablacje fork/join (1×GPU GPU1, hosting bez MTP, tg128, 3 powtórki)

| wariant | proza / kod |
|---|---|
| grafy OFF, baza | 34,85 / 34,70 |
| grafy OFF, FJ=1 + NOGRAPH=1 | 18,98 / 17,76 |
| grafy ON, FJ tylko `hc_mixed` | 21,81 / 21,85 |
| grafy ON, FJ `attn_norm` fan-out 3–6 | 35,69 / 36,04 (≈ baza 36,2; praktycznie brak forków) |
| grafy ON, FJ MAXROWS=1 MERGE=0 | 20,58 / 20,75 |

Wniosek końcowy: **kierunek fork/join zamknięty** — koszt nie leży w capture grafu (bez grafów jeszcze gorzej), tylko w samych zdarzeniach między strumieniami HIP (cudaEventRecord/cudaStreamWaitEvent ≈ dziesiątki µs każde na ROCm 7.2.4) i w utracie q8-dedup na strumieniach bocznych; każdy wariant z realnymi forkami traci 40–50 %. Patch zostaje w worktree wt-forkjoin (branch e55-forkjoin) jako dokumentacja, nie wchodzi do drzewa głównego. Jedyna droga bez barier to any-order na jednym strumieniu (bez zdarzeń) — patrz PLAN_ANYORDER.md i E61/E63.

## E63 — mikrotesty any-order (GPU1, `scratchpad/e0_anyorder/`, 200 prób każdy)

- **E0b (event/sync po jądrach any-order)**: hipEventRecord+EventSynchronize (timing i DisableTiming), hipStreamSynchronize, pusty kernel any-order + event, EventRecord→StreamWaitEvent(s2)+kernel na s2 — **wszystko PASS** (0 brakujących markerów) → runtime czeka na wszystkie jądra w locie; join-kernel przed event/sync nie jest konieczny (zostaje tylko join na końcu splitu, konstrukcyjnie).
- **E0c (kopie)**: hipMemcpyAsync D2D po 64 jądrach any-order jest uporządkowane (PASS). **Odwrotnie nie**: jądro any-order wydane po kopii 512 MB czyta stare dane w 200/200 prób (OVERTAKE: yes; zwykłe jądro 0/20) → zależność kopia→jądro wymaga bariery (reguła poison po operacjach niekernelowych w planie).
- **E0f (koszt hosta)**: `<<<>>>` 1,35–1,61 µs, hipExtLaunchKernelGGL(0) 1,31–1,63, any-order 1,58–1,70, hipLaunchKernel 1,35–1,58 µs/launch — host bez różnicy; strona GPU: puste jądra 2,96 → 1,91 µs/jądro; jądra ~20 µs: 22,9 µs → 1,9 µs/jądro (pełne nakładanie); kolejka asynchroniczna (host nie blokuje).

Wniosek: semantyka runtime jest przyjazna projektowi konserwatywnemu (E0b), a jedyna pułapka to kopia→jądro (E0c). Sufit zysku wyznacza teraz reżim CPU bez grafów — E61.

## E61 (E0e z planu any-order) — reżim bez grafów HIP na 2×GPU (obie karty wolne na czas pomiaru)

| wariant (tg128, bez MTP, KV f16, 4 powtórki) | proza / kod |
|---|---|
| grafy ON (baza) | 39,68 / 39,58 |
| grafy OFF | 38,20 / 38,19 (−3,7 %) |
| grafy OFF + HWQ8 | 38,20 / 38,23 |
| grafy ON (baza2) | 39,76 / 39,15 |

llama-completion bez grafów: 26,3 ms/token (38,0 t/s), prompt 375 t/s. Suma jąder z profilu E54 ≈ 21 ms/token (pod profilerem, lekko zawyżona) → **odzyskiwalne luki ≈ 5 ms/token**, sufit any-order ≈ 21–22 ms/token (45–47 t/s, +13–18 % vs grafy ON), pod warunkiem że CPU nadąża z wydawaniem (~9 µs/węzeł × ~2240 ≈ 20 ms). Go: etap 0+1 w implementacji (worktree wt-anyorder), potem dry-run (E0d) i bramki.

## E64 — any-order etap 0+1 (worktree <worktree any-order>, branch e64-anyorder, `patches/e64_anyorder.patch`), 1×GPU GPU1, hosting

Implementacja (własna, po 3 nieudanych próbach agentów – 529): tracker zakresów bajtów written/read od ostatniego launchu barierowego (per kontekst GPU, thread_local aktywny w graph_compute bez capture), grant jednorazowy zjadany przez pierwszy launch wezła w `ggml_cuda_kernel_launch` (hipExtLaunchKernelGGL + hipExtAnyOrderLaunch), odwołanie grantu: alokacja puli, mmvq/mmq/mmf/hipBLAS; zatrucie po operacjach niekernelowych (wrappery makr w vendors/hip.h); join-kernel na końcu grafu; whitelista etapu 1: MUL_MAT (zostaje mmvf), ADD/SUB/MUL/DIV, SCALE, RMS_NORM (+fuzje z MUL/ADD), UNARY (bez XIELU), GLU (bez OAI/clamp), ROPE (neox/mrope), GET_ROWS float, fuzje mmvf+GLU/bias, unary_mul, relu_sqr. Env `GGML_JOHNV8_ANYORDER=0|1|dry`, `_STATS=1`.

Wyniki: dry vs OFF IDENTYCZNE; bramki bit-exact ON vs OFF (2×) IDENTYCZNE; 0 asercji. STATS (1×GPU, ~70 splitów/token): 68 węzłów/graf, **any-order 9,3 % launchy** (granty MUL_MAT 8, GET_ROWS 2 na graf; odmowy RAW 10, WAR 5, WAW 3, poison 3, revoke 4).

| wariant (tg128, 4 powt.) | bez MTP proza/kod | z MTP proza/kod |
|---|---|---|
| grafy OFF, baza | 34,82 / 34,72 | — |
| grafy OFF, any-order etap 1 | 35,20 / 35,15 (+1,2 %) | 49,04 / 53,12 |
| grafy ON, baza | 36,40 / 36,34 | 50,58 / 53,46 |
| powtórki off_ao2 / off_baza2 | 35,22 / 35,25 vs 34,66 / 35,13 | — |

Wniosek: mechanizm bezpieczny i bit-exact, ale pokrycie etapu 1 za małe, by pokonać +4 % z grafów. Dalej: etap 2 (jądra hc/gdn/topk/fuzje rope+set_rows z deklarowanymi zakresami scratch) i etap 3 (mmvq: zakresy slotów q8-cache zamiast odwołania) → pomiar na 2×GPU (2 grafy/token).

## E64B — any-order etap 2+3 (commit 0878593 w wt-anyorder): sloty q8 mmvq deklarowane zamiast odwołania, scratch hcblock deklarowany, whitelista MUL_MAT_ID, prosby dla fuzji mmvq-GLU/bias, hc_combine(+inject), hcblock_a, shexp_tail, scale_silu, topk_moe, rms_norm+rope

Poprawność: dry vs OFF IDENTYCZNE, 2 bramki bit-exact IDENTYCZNE, 0 asercji. STATS 1×GPU: any-order **14,9 %** launchy (było 9,3 %); odmowy/graf RAW 10, WAR 5, WAW 16, poison 3; granty MUL_MAT 9, MUL_MAT_ID 0 (ids świeżo zapisane przez topk → RAW/WAW).

| wariant (tg128, 4 powt.) | bez MTP proza/kod | z MTP proza/kod |
|---|---|---|
| grafy OFF, baza | 34,91 / 34,97 | — |
| grafy OFF, any-order 2+3 | **35,60 / 35,57 (+2,0 %)** | 50,16 / 53,07 |
| grafy ON, baza | 36,34 / 36,29 | 49,40 / 53,35 |
| powtórki off_ao2 / off_baza2 | 35,61 / 35,55 vs 35,01 / 35,04 | — |

Na 1×GPU (≈70 splitów/token, każdy startuje zatruty) nadal poniżej grafów ON. Test właściwy: 2×GPU (E65).

## E65 — any-order etap 2+3 na 2×GPU (obie karty wolne na czas pomiaru), KV f16, wszystko na GPU, 4 powtórki

STATS (2 grafy/token): 951 węzłów/graf, launch/graf: bariera 913, any-order 142 (**13,4 %**), ops 62,9 (memcpy/memset → poison 36), odmowy/graf: RAW 183, WAR 82, **WAW 205** (aliasowanie pamięci przez ggml-alloc: dst nowego węzła nachodzi na niedawno zapisany zakres), granty: MUL_MAT 109, GET_ROWS 33, MUL_MAT_ID 0 (ids tuż po topk → RAW).

| wariant (tg128) | bez MTP proza/kod | z MTP proza/kod |
|---|---|---|
| grafy ON, baza | 39,96 / 40,05 | 65,08 / 69,97 |
| grafy OFF, baza | 38,38 / 38,40 | — |
| grafy OFF, any-order 2+3 | 39,88 / 39,91 (run 1), **41,21 / 41,15** (run 2) | 62,68 / 67,27 |
| grafy ON, baza2 | 40,10 / 40,13 | — |

llama-completion bez grafów + any-order: 24,9 ms/token (40,1 t/s) vs 26,3 bez any-order (E61). Wniosek: any-order daje +4…+7 % w reżimie bez grafów i wychodzi na remis/+3 % z grafami ON bez MTP; z MTP −4 % (małe grafy MTP tracą grafy HIP). Pokrycie 13 % ograniczają: łańcuchy zależności (RAW), fałszywe WAW z nadzbiorowych zbiorów W/R (pośrednie węzły fuzji, które nigdy nie są zapisywane) oraz poison po memcpy (cpy/concat). Następne: precyzyjne W/R dla fuzji, grafy HIP tylko dla małych grafów (MTP) + any-order dla dużych, kopie przez kernel zamiast memcpy.

## E64C — any-order: precyzyjne W/R fuzji + kopie przez kernel + grafy HIP tylko dla małych grafów (MINNODES=400) — **bramka bit-exact NIE przeszła**

STATS 1×GPU: any-order 16,1 % launchy, poison 1/graf (kopie przez kernel usunęły zatrucia), RAW 27, WAR 8, WAW 7. Szybkość 1×GPU: grafy dla małych + any-order dla dużych = remis z grafami ON (36,2–36,4 vs 36,3–36,5; z MTP 50,1 vs 50,4). **Ale**: dry vs OFF identyczne, a bramki ON vs OFF: [k] identyczne, **[d] RÓŻNE (2×)** → wyścig: precyzyjne W = „tylko ostatni węzeł grupy" jest niekompletne (któraś fuzja zapisuje węzeł wcześniejszy w grupie). Poprawka: W = ostatni + każdy węzeł grupy z konsumentem poza grupą (ggml_node_get_use_count > użycia wewnątrz) lub z flagą OUTPUT.

## E64D — any-order z poprawionym W fuzji (commit eb9b509): dry vs OFF IDENTYCZNE, 2 bramki ON vs OFF IDENTYCZNE ([k] i [d]), 0 asercji

STATS 1×GPU: any-order 16,1 % launchy, poison 1/graf, RAW 24, WAR 11, WAW 7. 1×GPU (≈70 splitów/token): grafy dla małych + any-order dla dużych = remis (36,30/36,35 vs 36,08/36,29; powtórki 36,28/36,25 vs 36,47/36,42), z MTP 49,6–50,5 vs 50,2. Test właściwy: E66 na 2×GPU.

## E66 — any-order (E64D) na 2×GPU (obie karty wolne na czas pomiaru), KV f16, 4 powtórki

STATS: 951 węzłów/graf, any-order **15,0 %** launchy (granty MUL_MAT 117, GET_ROWS 33, CONT 9, SET_ROWS 7, CPY 1), poison 8, odmowy RAW **395**, WAR 143, WAW 68 → resztę blokują prawdziwe łańcuchy zależności.

| wariant (tg128) | bez MTP proza/kod | z MTP proza/kod |
|---|---|---|
| grafy ON, baza | 39,43 / 40,09 | 61,29 / 65,93 |
| **grafy (małe) + any-order (duże)** | **41,52 / 41,43** | **62,22 / 66,43** |
| grafy OFF + any-order | 40,18 / 40,15 | — |
| powtórki: on_ao2 / on_baza2 | 40,18 / 40,21 vs 40,23 / 38,77 (dryf) | 61,75 / 66,50 |

Werdykt any-order: bit-exact, bezpieczny, **+2…+4 % dekodu bez MTP na 2×GPU, +1 % z MTP, 0 na 1×GPU**. Sufit wyznaczają łańcuchy RAW (395/graf): bez przestawiania grafu (żeby niezależne jądra sąsiadowały — stage 5 z planu, odpowiednik ggml_vk_graph_optimize) więcej się nie wyciśnie. Patch zostaje w worktree wt-anyorder (`patches/e64_anyorder.patch`, domyślnie OFF); do 8092 (1×GPU) nie wdrożony (brak zysku).
