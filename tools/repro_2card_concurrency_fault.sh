#!/usr/bin/env bash
# Aggregate throughput under concurrency. Everything measured so far is
# single-stream; this is how the box is actually driven (open-webui/litellm).
# Watch for the documented 2-card fault on the 2nd sequential request.
set -uo pipefail

M=/models/nemotron-heretic-gguf/NVIDIA-Nemotron-3-Super-120B-A12B-BF16-heretic.i1-Q4_K_M.gguf
NP=${1:-4}
PORT=8099

docker rm -f conctest >/dev/null 2>&1
docker run -d --name conctest --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --shm-size 16g -p $PORT:$PORT \
  -v /home/dave/llamacpp-ssd:/src -v /mnt/llm-storage:/models -w /src \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/core-7.14/lib \
  --entrypoint bash llama-rocm714-rpc:tune -c \
  "./build/bin/llama-server -m $M -ngl 99 -sm layer -c 16384 -np $NP \
     -b 4096 -ub 2048 -fa 1 -ctk q8_0 -ctv q8_0 -t 24 --host 0.0.0.0 --port $PORT" >/dev/null

echo "waiting for server (np=$NP)..."
for i in $(seq 1 120); do
  if curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok; then echo "ready after ${i}0s"; break; fi
  sleep 10
done

req() {   # $1 = tag
  curl -s -m 600 "http://127.0.0.1:$PORT/completion" \
    -H 'Content-Type: application/json' \
    -d '{"prompt":"Explain how a binary search tree works, then give an example.","n_predict":128,"temperature":0,"seed":1234}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); t=d.get('timings',{}); print('$1', round(t.get('predicted_per_second',0),2), 'tok/s  prompt', round(t.get('prompt_per_second',0),1))"
}

echo "=== 1 request alone ==="
S=$(date +%s.%N); req single; E=$(date +%s.%N)

echo "=== $NP concurrent ==="
S=$(date +%s.%N)
for i in $(seq 1 $NP); do req "conc$i" & done
wait
E=$(date +%s.%N)

echo "=== faults (after settle) ==="
# settle first: reading docker logs immediately gave a FALSE "no fault" once,
# because the fault had not been flushed yet when the header appeared.
sleep 8
echo "fault_lines=$(docker logs conctest 2>&1 | grep -icE 'memory access fault')"
docker logs conctest 2>&1 | grep -iE "memory access fault" | head -2
docker rm -f conctest >/dev/null 2>&1
