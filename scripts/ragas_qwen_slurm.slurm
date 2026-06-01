#!/bin/bash
#SBATCH --job-name=xai_rag_ragas_qwen
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/xai_rag_ragas_qwen_%j.out
#SBATCH --error=logs/xai_rag_ragas_qwen_%j.err

# Evaluación RAGAS usando Qwen2.5-14B-Instruct como juez.
# Reutiliza eval/rag_answers.json (generado con Llama) → solo Fase 2.
# Salida: eval/ragas_results_qwen_judge.json
# Al finalizar imprime comparativa Llama-judge vs Qwen-judge.

set -euo pipefail

nvidia-smi

source "$SLURM_SUBMIT_DIR/.venv312/bin/activate"
echo "Python: $(which python) — $(python --version)"

cd "$SLURM_SUBMIT_DIR"

# ── Entorno base ──────────────────────────────────────────────
export SENTENCE_TRANSFORMERS_HOME="$SLURM_SUBMIT_DIR/models"
export HF_HOME="$SLURM_SUBMIT_DIR/models/huggingface"
export SENTENCE_TRANSFORMERS_DEVICE=cpu
export VLLM_USE_V1=0   # V1 tiene ABI mismatch con torch+cu124

JUDGE_MODEL="Qwen/Qwen2.5-14B-Instruct"
OUT_FILE="eval/ragas_results_qwen_judge.json"
LLAMA_FILE="eval/ragas_results.json"

# ── Verificar que rag_answers.json existe ─────────────────────
if [ ! -f "eval/rag_answers.json" ]; then
    echo "ERROR: eval/rag_answers.json no existe."
    echo "  Lanza primero ragas_slurm.slurm para generar las respuestas RAG."
    exit 1
fi
N_ANSWERS=$(python -c "import json; print(len(json.load(open('eval/rag_answers.json'))))")
echo "Respuestas RAG disponibles: ${N_ANSWERS} items (--skip-inference activado)"

# ── Descarga Qwen2.5-14B-Instruct si no está en caché ────────
echo ""
echo "=== Verificando caché de ${JUDGE_MODEL} ==="
MODEL_CACHE_DIR="$HF_HOME/hub/models--Qwen--Qwen2.5-14B-Instruct"

if [ -d "$MODEL_CACHE_DIR" ]; then
    echo "Modelo ya en caché: $MODEL_CACHE_DIR"
else
    echo "Descargando ${JUDGE_MODEL} (~28GB safetensors)..."
    # HF_HUB_OFFLINE no debe estar activo durante la descarga
    unset HF_HUB_OFFLINE
    python -c "
from huggingface_hub import snapshot_download
import os
path = snapshot_download(
    repo_id='$JUDGE_MODEL',
    ignore_patterns=['*.pt', 'original/*'],
)
print(f'Descargado en: {path}')
"
    echo "Descarga completada."
fi

# Activar modo offline para todo lo que sigue
export HF_HUB_OFFLINE=1

# ── Arrancar vLLM con Qwen2.5-14B ────────────────────────────
echo ""
echo "=== Arrancando vLLM (${JUDGE_MODEL}) ==="

# Qwen2.5-14B en bfloat16 ocupa ~28GB → 40GB A100 es suficiente con util=0.90
# trust_remote_code necesario para el tokenizador de Qwen
python -m vllm.entrypoints.openai.api_server \
    --model "$JUDGE_MODEL" \
    --port 8000 \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    > logs/vllm_qwen_${SLURM_JOBID}.out 2>&1 &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# ── Esperar a que vLLM esté listo (max 400s — Qwen tarda más) ─
echo "Esperando a que vLLM responda en localhost:8000..."
MAX_WAIT=400
ELAPSED=0
until curl -s http://localhost:8000/v1/models > /dev/null 2>&1; do
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "ERROR: vLLM no respondió en ${MAX_WAIT}s"
        echo "--- últimas líneas del log vLLM ---"
        tail -20 logs/vllm_qwen_${SLURM_JOBID}.out
        kill $VLLM_PID 2>/dev/null
        exit 1
    fi
done
echo "vLLM listo (${ELAPSED}s)"
curl -s http://localhost:8000/v1/models \
    | python -c "import sys,json; d=json.load(sys.stdin); print('Modelo activo:', d['data'][0]['id'])"

# ── Fase 2 únicamente: evaluación RAGAS con juez Qwen ────────
echo ""
echo "=== Evaluación RAGAS (juez: ${JUDGE_MODEL}) ==="
python eval/ragas_eval.py \
    --vllm-url      http://localhost:8000 \
    --model         "$JUDGE_MODEL" \
    --dataset       dataset_qa_boe.json \
    --answers       eval/rag_answers.json \
    --out           "$OUT_FILE" \
    --skip-inference \
    --wait-vllm     0

EXIT_CODE=$?

# ── Apagar vLLM ──────────────────────────────────────────────
echo ""
echo "=== Deteniendo vLLM ==="
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

# ── Comparativa Llama-judge vs Qwen-judge ─────────────────────
echo ""
echo "=== COMPARATIVA DE JUECES ==="
python - <<'PYEOF'
import json, pathlib, sys, math

METRICS = {
    "faithfulness":                         "Faithfulness",
    "answer_relevancy":                     "Answer Relevancy",
    "llm_context_precision_with_reference": "Context Precision",
    "context_recall":                       "Context Recall",
}

def load(path):
    p = pathlib.Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("aggregate", {})

def fmt(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "  —   "
    return f"{v:.4f}"

llama = load("eval/ragas_results.json")
qwen  = load("eval/ragas_results_qwen_judge.json")

if llama is None and qwen is None:
    print("No se encontraron resultados.")
    sys.exit(1)

col_w = 10
header = f"{'Métrica':<44} {'Llama-8B':>{col_w}} {'Qwen-14B':>{col_w}} {'Δ (Qwen-Llama)':>{col_w+2}}"
print(header)
print("-" * len(header))

for key, label in METRICS.items():
    v_llama = (llama or {}).get(key)
    v_qwen  = (qwen  or {}).get(key)
    if v_llama is not None and v_qwen is not None:
        delta = v_qwen - v_llama
        delta_str = f"{delta:+.4f}"
    else:
        delta_str = "  —   "
    print(f"{label:<44} {fmt(v_llama):>{col_w}} {fmt(v_qwen):>{col_w}} {delta_str:>{col_w+2}}")

print()
n_llama = len(json.loads(pathlib.Path("eval/ragas_results.json").read_text()).get("per_item", [])) if llama else 0
n_qwen  = len(json.loads(pathlib.Path("eval/ragas_results_qwen_judge.json").read_text()).get("per_item", [])) if qwen else 0
print(f"  Items Llama-judge : {n_llama}")
print(f"  Items Qwen-judge  : {n_qwen}")
PYEOF

echo ""
echo "Resultados Qwen guardados en: $OUT_FILE"

exit $EXIT_CODE
