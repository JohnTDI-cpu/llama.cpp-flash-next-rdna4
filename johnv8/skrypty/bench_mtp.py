#!/usr/bin/env python3
"""pp512/pp1024/pp2048 + tg128 dla Flash-Next IQ3, z MTP i bez. GPU1 + CPU, GPU0 nietykane.

Prompty budowane przez /tokenize + /detokenize, wiec dlugosc jest DOKLADNA, nie szacowana.
Zmienne: BIN, ETYK, SPEC (draft-mtp|none), NMAX, PMIN
"""
import json, os, re, signal, subprocess, sys, time, urllib.request

BIN  = os.environ["BIN"].rstrip('/')
ETYK = os.environ["ETYK"]
SPEC = os.environ.get("SPEC", "draft-mtp")
NMAX = os.environ.get("NMAX", "6")
PMIN = os.environ.get("PMIN", "0.7")
PORT = int(os.environ.get("PORT", "8095"))
M  = os.environ.get("MODEL", os.path.expanduser("~/models/Qwen3.8-Flash-Next-UD-IQ3_XXS/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf"))
MD = os.environ.get("DRAFT", os.path.expanduser("~/models/Flash-Next-mtp/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"))
LOG = os.path.join(os.environ.get("BENCH_DIR", "."), f"bench_{ETYK}.log")
ROCM = os.environ.get("ROCM_PATH", "/opt/rocm-7.2.4")

def vram():
    try:
        d=json.loads(subprocess.run(["rocm-smi","--showmeminfo","vram","--json"], capture_output=True, text=True, timeout=20).stdout)
        return {k:int(v.get("VRAM Total Used Memory (B)",0))//2**20 for k,v in d.items()}
    except Exception: return {}
BAZA = vram()

env=dict(os.environ)
BACKEND=os.environ.get("BACKEND","rocm")
if BACKEND=="vulkan":
    DEV=os.environ.get("DEVICES","Vulkan1"); env["VK_ICD_FILENAMES"]="/usr/share/vulkan/icd.d/radeon_icd.json"; env["GGML_VK_ALLOW_GRAPHICS_QUEUE"]="1"
    env["LD_LIBRARY_PATH"]=f"{BIN}"
else:
    DEV=os.environ.get("DEVICES","ROCm1"); env["ROCM_PATH"]=ROCM
    env["LD_LIBRARY_PATH"]=f"{BIN}:{ROCM}/lib"; (env.__setitem__("GGML_CUDA_DISABLE_GRAPHS","1") if os.environ.get("GRAFY_OFF","1")!="0" else env.pop("GGML_CUDA_DISABLE_GRAPHS",None))
CPU_OD = int(os.environ.get("CPU_OD","16"))
OT = ("per_layer_token_embd=CPU" if CPU_OD>=48 else r"per_layer_token_embd=CPU,blk\.(" + "|".join(str(i) for i in range(CPU_OD,48)) + r")\.ffn_.*_exps=CPU")

cmd=[f"{BIN}/llama-server","-m",M,"--host","127.0.0.1","--port",str(PORT),"--alias","bench",
     "-c",os.environ.get("CTX","8192"),"-np",os.environ.get("NP","1"),"--kv-unified","-ngl","99","--fit","off","--device",DEV,"-ot",OT,
     "--cache-type-k",os.environ.get("KV","q8_0"),"--cache-type-v",os.environ.get("KV","q8_0"),
     *(["-ts",os.environ["TS"]] if os.environ.get("TS") else []),"--flash-attn","on","-ub",os.environ.get("UB","1536"),"-b","1536","-t",os.environ.get("WATKI","32"),"-tb",os.environ.get("WATKI_B",os.environ.get("WATKI","32")),"--no-warmup","--seed","1234",
     "--temp","1.0","--top-p","0.95","--top-k","20","--min-p","0.0","-lv","3"]
if os.environ.get("MLOCK"): cmd += ["--mlock"]
if os.environ.get("OTD"): cmd += ["-otd", os.environ["OTD"]]
if os.environ.get("EXTRA"): import shlex as _sx; cmd += _sx.split(os.environ["EXTRA"])
if SPEC != "none":
    cmd += ["-md",MD,"--spec-draft-ngl","99","--spec-draft-device",os.environ.get("DEVD",DEV.split(",")[0]),"--spec-type",SPEC,
            "--spec-draft-n-max",NMAX,"--spec-draft-p-min",PMIN]
else:
    cmd += ["--spec-type","none"]

lf=open(LOG,"w")
p=subprocess.Popen(cmd,env=env,stdout=lf,stderr=subprocess.STDOUT,preexec_fn=os.setsid)
def ubij():
    try: os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except Exception: pass
    for _ in range(120):
        time.sleep(1)
        # port MUSI byc wolny, nie tylko VRAM - inaczej kolejny przebieg pada na bind
        port_wolny = subprocess.run(["ss","-ltn"],capture_output=True,text=True).stdout.find(f"127.0.0.1:{PORT}") < 0
        v=vram()
        if not v: continue   # rocm-smi padl/timeout -> nie uznawaj za wolne
        if port_wolny and v.get("card1",0) <= BAZA.get("card1",0)+500 and v.get("card0",0) <= BAZA.get("card0",0)+500: return

def post(sciezka, obj, t=900):
    r=urllib.request.Request(f"http://127.0.0.1:{PORT}{sciezka}",json.dumps(obj).encode(),
                             {"Content-Type":"application/json"})
    with urllib.request.urlopen(r,timeout=t) as x: return json.loads(x.read())

try:
    t0=time.time()
    while time.time()-t0 < 600:
        if p.poll() is not None:
            print(f"  PADL rc={p.returncode}: "+"".join(open(LOG).readlines()[-6:])); sys.exit(1)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health",timeout=2) as r:
                if r.status==200: break
        except Exception: time.sleep(2)
    po=vram(); d0=po.get("card0",0)-BAZA.get("card0",0)
    if d0>300 and not os.environ.get("GPU0_OK"): print(f"  ALARM: GPU0 urosl o {d0} MiB"); ubij(); sys.exit(2)
    print(f"  {ETYK} [{BACKEND} spec={SPEC}, {48-CPU_OD} warstw na CPU] VRAM card0 {po.get('card0',0)-BAZA.get('card0',0)} / card1 {po.get('card1',0)-BAZA.get('card1',0)} MiB, "
          f"GPU0 delta {d0} MiB", flush=True)

    zrodlo = (open("~/").read()
              + open("~/").read()) * 3
    toks = post("/tokenize", {"content": zrodlo})["tokens"]
    def prompt_o_dlugosci(n):
        return post("/detokenize", {"tokens": toks[:n]})["content"]

    def med(xs): xs=sorted(xs); return xs[len(xs)//2]
    def sr(xs):
        m=sum(xs)/len(xs)
        return m,(sum((x-m)**2 for x in xs)/(len(xs)-1))**0.5 if len(xs)>1 else 0.0

    w={"etykieta":ETYK,"spec":SPEC,"nmax":NMAX,"pmin":PMIN}
    # rozgrzewka
    post("/completion", {"prompt":"Silnik", "n_predict":16, "cache_prompt":False})

    for n in ([] if os.environ.get("TG_ONLY","0") not in ("","0") else (512, 1024, 2048)):
        pf=[]; realne=0
        for _ in range(4):
            r=post("/completion", {"prompt":prompt_o_dlugosci(n), "n_predict":1, "cache_prompt":False})
            pf.append(r["timings"]["prompt_per_second"]); realne=r["timings"]["prompt_n"]
        a,s=sr(pf)
        w[f"pp{n}"]=med(pf); w[f"pp{n}_all"]=[round(x,1) for x in pf]; w[f"pp{n}_tok"]=realne
        print(f"    pp{n:<5} ({realne:>4} tok)  {a:7.1f} +- {s:5.1f} t/s   {[round(x,1) for x in pf]}", flush=True)

    PROMPTY={"proza":"Opisz budowe silnika wysokopreznego.",
             "kod":"Napisz w Pythonie funkcje, ktora parsuje CSV z ofertami aut (marka, model, rok, przebieg, cena) i zwraca mediane ceny dla kazdego rocznika. Z obsluga bledow i typowaniem."}
    for nazwa,tresc in PROMPTY.items():
        tg=[]
        for _ in range(int(os.environ.get("POWT","5"))):
            r=post("/completion", {"prompt":tresc, "n_predict":128, "cache_prompt":False})
            tg.append(r["timings"]["predicted_per_second"])
        a,s=sr(tg)
        w[f"tg128_{nazwa}"]=med(tg); w[f"tg128_{nazwa}_all"]=[round(x,2) for x in tg]
        if nazwa=="proza": w["tg128"]=med(tg)
        print(f"    tg128 {nazwa:<6}     {a:7.2f} +- {s:5.2f} t/s   {[round(x,2) for x in tg]}", flush=True)

    log=open(LOG).read()
    ak=re.findall(r"draft acceptance =\s*([0-9.]+)", log)
    dl=re.findall(r"mean len =\s*([0-9.]+)", log)
    w["akceptacja"]=ak[-1] if ak else None; w["mean_len"]=dl[-1] if dl else None
    if SPEC!="none": print(f"    draft: akceptacja {w['akceptacja']}, dlugosc {w['mean_len']}", flush=True)
    json.dump(w, open(f"~/","w"), indent=2)
finally:
    ubij()
