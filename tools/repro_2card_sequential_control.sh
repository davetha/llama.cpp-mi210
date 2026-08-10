#!/usr/bin/env bash
# Is the trigger "the 2nd request" rather than concurrency? Fire 3 STRICTLY
# SEQUENTIAL requests at -np 1 on 2 cards. Also fixes the harness race that gave
# a false "no fault": wait for the container to actually exit before reading logs.
set -uo pipefail
M=/models/nemotron-heretic-gguf/NVIDIA-Nemotron-3-Super-120B-A12B-BF16-heretic.i1-Q4_K_M.gguf
NP=${1:-1}
PORT=8099

docker rm -f seqtest >/dev/null 2>&1
docker run -d --name seqtest --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --shm-size 16g -p $PORT:$PORT \
  -v /home/dave/llamacpp-ssd:/src -v /mnt/llm-storage:/models -w /src \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/core-7.14/lib \
  --entrypoint bash llama-rocm714-rpc:tune -c \
  "./build/bin/llama-server -m $M -ngl 99 -sm layer -c 8192 -np $NP \
     -b 4096 -ub 2048 -fa 1 -ctk q8_0 -ctv q8_0 -t 24 --host 0.0.0.0 --port $PORT" >/dev/null

for i in $(seq 1 120); do
  curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok && { echo "ready (np=$NP)"; break; }
  sleep 10
done

for r in 1 2 3; do
  OUT=$(curl -s -m 300 "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
    -d '{"prompt":"Name three colours.","n_predict":24,"temperature":0,"seed":1234}' 2>&1)
  if echo "$OUT" | grep -q predicted_per_second; then
    echo "request $r: OK   $(echo "$OUT" | python3 -c 'import sys,json;print(round(json.load(sys.stdin)["timings"]["predicted_per_second"],2),"tok/s")' 2>/dev/null)"
  else
    echo "request $r: FAILED"
  fi
  sleep 2
done

echo "=== server log (after settle) ==="
sleep 5
docker logs seqtest 2>&1 | grep -icE "memory access fault" | sed 's/^/fault_lines=/'
docker logs seqtest 2>&1 | grep -iE "memory access fault" | head -2
docker rm -f seqtest >/dev/null 2>&1
