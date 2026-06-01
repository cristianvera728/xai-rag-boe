#!/bin/bash
#SBATCH --job-name=xai_rag_ragas
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/xai_rag_ragas_%j.out
#SBATCH --error=logs/xai_rag_ragas_%j.err

# ── Entorno ───────────────────────────────────────────────────
nvidia-smi

source "$SLURM_SUBMIT_DIR/.venv312/bin/activate"
echo "Python: $(which python) — $(python --version)"

cd "$SLURM_SUBMIT_DIR"

# Caché local de modelos (bge-m3 ya está aquí)
export SENTENCE_TRANSFORMERS_HOME="$SLURM_SUBMIT_DIR/models"
export HF_HOME="$SLURM_SUBMIT_DIR/models/huggingface"
export HF_HUB_OFFLINE=1

# bge-m3 en CPU para dejar la GPU íntegra a vLLM
export SENTENCE_TRANSFORMERS_DEVICE=cpu

# Forzar motor V0 de vLLM: el V1 tiene incompatibilidad ABI con torch+cu124
export VLLM_USE_V1=0

# ── Llama 3.1-8B: ruta del modelo ────────────────────────────
# Si tienes el modelo descargado en ~/.cache/huggingface, usa HF_HUB_OFFLINE=1
# Si usas un path local directo, cambia MODEL_ID por la ruta absoluta
MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"

# ── Arrancar vLLM en background ──────────────────────────────
echo ""
echo "=== Arrancando vLLM (${MODEL_ID}) ==="
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --port 8000 \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    > logs/vllm_${SLURM_JOBID}.out 2>&1 &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# ── Esperar a que vLLM esté listo ────────────────────────────
echo "Esperando a que vLLM responda en localhost:8000..."
MAX_WAIT=300
ELAPSED=0
until curl -s http://localhost:8000/v1/models > /dev/null 2>&1; do
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "ERROR: vLLM no respondió en ${MAX_WAIT}s — abortando"
        kill $VLLM_PID 2>/dev/null
        exit 1
    fi
done
echo "vLLM listo (${ELAPSED}s)"
curl -s http://localhost:8000/v1/models | python -c "import sys,json; d=json.load(sys.stdin); print('Modelo:', d['data'][0]['id'])"

# ── Fase 1 + 2: RAG inference + evaluación RAGAS ─────────────
echo ""
echo "=== Evaluación RAGAS ==="
python eval/ragas_eval.py \
    --vllm-url  http://localhost:8000 \
    --model     "$MODEL_ID" \
    --dataset   dataset_qa_boe.json \
    --answers   eval/rag_answers.json \
    --out       eval/ragas_results.json \
    --top-k     5 \
    --wait-vllm 0

EXIT_CODE=$?

# ── Apagar vLLM ──────────────────────────────────────────────
echo ""
echo "=== Deteniendo vLLM ==="
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

# ── Resumen ───────────────────────────────────────────────────
echo ""
echo "=== ragas_results.json ==="
python -c "
import json, pathlib, sys
p = pathlib.Path('eval/ragas_results.json')
if not p.exists():
    print('No se generó ragas_results.json')
    sys.exit(1)
r = json.loads(p.read_text())
agg = r.get('aggregate', {})
for k, v in agg.items():
    print(f'  {k:<40} {v:.4f}')
print(f'Items evaluados: {len(r.get(\"per_item\", []))}')
"

exit $EXIT_CODE
