# Minimal dependency-free GGUF tensor-info reader.
import struct, sys
from collections import defaultdict

TYPES = {0:"F32",1:"F16",2:"Q4_0",3:"Q4_1",6:"Q5_0",7:"Q5_1",8:"Q8_0",9:"Q8_1",
         10:"Q2_K",11:"Q3_K",12:"Q4_K",13:"Q5_K",14:"Q6_K",15:"Q8_K",
         16:"IQ2_XXS",17:"IQ2_XS",18:"IQ3_XXS",19:"IQ1_S",20:"IQ4_NL",
         21:"IQ3_S",22:"IQ2_S",23:"IQ4_XS",24:"I8",25:"I16",26:"I32",
         27:"I64",28:"F64",29:"IQ1_M",30:"BF16"}

f = open(sys.argv[1], "rb")
rd = lambda fmt: struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

assert f.read(4) == b"GGUF"
rd("<I")                       # version
n_tensors = rd("<Q")
n_kv      = rd("<Q")

def skip_val(t):
    if   t in (0,1):  f.read(1)
    elif t in (2,3):  f.read(2)
    elif t in (4,5,6):f.read(4)
    elif t == 7:      f.read(1)
    elif t == 8:      f.read(rd("<Q"))
    elif t == 9:
        et = rd("<I"); n = rd("<Q")
        for _ in range(n): skip_val(et)
    elif t in (10,11,12): f.read(8)
    else: raise SystemExit("bad kv type %d" % t)

for _ in range(n_kv):
    f.read(rd("<Q"))           # key
    skip_val(rd("<I"))

by_type = defaultdict(lambda: [0, 0])          # name -> [count, elements]
downgraded = defaultdict(lambda: [0, 0])
samples = []
for _ in range(n_tensors):
    name = f.read(rd("<Q")).decode("utf-8", "replace")
    nd   = rd("<I")
    dims = [rd("<Q") for _ in range(nd)]
    tt   = TYPES.get(rd("<I"), "?")
    rd("<Q")                                   # offset
    ne = 1
    for d in dims: ne *= d
    by_type[tt][0] += 1
    by_type[tt][1] += ne
    if tt in ("Q5_0", "Q8_0"):
        downgraded[(tt, dims[0], dims[0] % 256)][0] += 1
        downgraded[(tt, dims[0], dims[0] % 256)][1] += ne
        if len(samples) < 6: samples.append((name, tt, dims))

tot = sum(v[1] for v in by_type.values())
print("type       count      Gelem   share-of-weights")
for k,(c,e) in sorted(by_type.items(), key=lambda kv:-kv[1][1]):
    print("%-9s %6d %10.2f %8.1f%%" % (k, c, e/1e9, 100*e/tot))

print("\nQ5_0 / Q8_0 tensors by ncols  (llama-quant downgrades when ncols %% 256 != 0):")
for (t,nc,mod),(c,e) in sorted(downgraded.items(), key=lambda kv:-kv[1][1]):
    flag = "<-- NOT divisible by 256" if mod else "   divisible (downgrade NOT explained by ncols)"
    print("  %-5s ncols=%-6d mod=%-4d count=%-5d %7.2f Gelem  %s" % (t,nc,mod,c,e/1e9,flag))

print("\nsamples:")
for n,t,d in samples: print("  %-46s %-5s %s" % (n,t,d))
