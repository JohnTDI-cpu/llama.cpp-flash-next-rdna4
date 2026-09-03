#!/usr/bin/env python3
# spis_jader.py <trace.csv> [<trace_ref.csv>]  -> jader/token (srednia z 40 ostatnich tokenow) + roznice rodzin
import csv, collections, re, sys
def spis(T):
    rows=sorted(csv.DictReader(open(T)), key=lambda r:int(r["Start_Timestamp"]))
    f=[re.sub(r"\(.*","",re.sub(r"<.*","",r["Kernel_Name"])).split("::")[-1].replace("void ","")[:34] for r in rows]
    idx=[i for i,x in enumerate(f) if x.startswith("topk_moe")]; n=len(idx)//48; g=[idx[k*48] for k in range(n)]
    c=collections.Counter(f[g[-40]:]); return {k:v/40 for k,v in c.items()}, sum(c.values())/40
a,na=spis(sys.argv[1])
if len(sys.argv)>2:
    b,nb=spis(sys.argv[2]); print(f"  jader/token: ref {nb:.0f} -> ten {na:.0f}  (roznica {na-nb:+.0f})")
    print("  %-38s %8s %8s %8s"%("rodzina","ref","ten","delta"))
    for k in sorted(set(a)|set(b), key=lambda k:-(b.get(k,0)+a.get(k,0))):
        d=a.get(k,0)-b.get(k,0)
        if abs(d)>=0.5 or b.get(k,0)>=100: print("  %-38s %8.1f %8.1f %+8.1f"%(k,b.get(k,0),a.get(k,0),d))
else:
    print(f"  jader/token: {na:.0f}")
    for k,v in sorted(a.items(), key=lambda kv:-kv[1])[:12]: print("  %-38s %8.1f"%(k,v))
