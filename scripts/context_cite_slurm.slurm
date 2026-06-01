#!/bin/bash
#SBATCH --job-name=xai_rag_context_cite
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/xai_rag_context_cite_%j.out
#SBATCH --error=logs/xai_rag_context_cite_%j.err

# ContextCite attribution evaluation for RAG-BOE.
#
# Starts vLLM with Llama 3.1-8B-Instruct (same setup as RAGAS) and runs the
# ContextCite ablation algorithm on the 200 QA pairs from rag_answers.json.
#
# Algorithm (per item):
#   1. Split 5 context chunks into sentences (~25-30 parts/item).
#   2. Sample N_ABLATIONS=64 random binary masks over parts.
#   3. For each mask, call vLLM completions (echo=True) to get log P(answer | ablated_ctx).
#   4. Fit Ridge regression → attribution weight per sentence.
#   5. Aggregate to chunk level → save to context_cite_results.json.
#
# Prerequisites: eval/rag_answers.json must exist (run ragas_slurm.slurm first).
# Output: eval/context_cite_results.json

set -euo pipefail

nvidia-smi

source "$SLURM_SUBMIT_DIR/.venv312/bin/activate"
echo "Python: $(which python) — $(python --version)"

cd "$SLURM_SUBMIT_DIR"

# ── Entorno ───────────────────────────────────────────────────────────────────
export SENTENCE_TRANSFORMERS_HOME="$SLURM_SUBMIT_DIR/models"
export HF_HOME="$SLURM_SUBMIT_DIR/models/huggingface"
export HF_HUB_OFFLINE=1
export SENTENCE_TRANSFORMERS_DEVICE=cpu
export VLLM_USE_V1=0   # V1 tiene ABI mismatch con torch+cu124

MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
OUT_FILE="eval/context_cite_results.json"

# ── Verificar prerequisitos ────────────────────────────────────────────────────
if [ ! -f "eval/rag_answers.json" ]; then
    echo "ERROR: eval/rag_answers.json no existe."
    echo "  Lanza primero ragas_slurm.slurm para generar las respuestas RAG."
    exit 1
fi
N_ANSWERS=$(python -c "import json; print(len(json.load(open('eval/rag_answers.json'))))")
echo "Respuestas disponibles: ${N_ANSWERS} items en rag_answers.json"

# ── Verificar dependencias ─────────────────────────────────────────────────────
echo ""
echo "=== Verificando dependencias ==="
python -c "import sklearn; print(f'scikit-learn {sklearn.__version__}')"
python -c "import openai; print(f'openai {openai.__version__}')"
python -c "from transformers import AutoTokenizer; print('transformers OK')"

# ── Arrancar vLLM ─────────────────────────────────────────────────────────────
echo ""
echo "=== Arrancando vLLM (${MODEL_ID}) ==="
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --port 8000 \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    > logs/vllm_cc_${SLURM_JOBID}.out 2>&1 &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# ── Esperar a que vLLM esté listo ─────────────────────────────────────────────
echo "Esperando a que vLLM responda en localhost:8000..."
MAX_WAIT=300
ELAPSED=0
until curl -s http://localhost:8000/v1/models > /dev/null 2>&1; do
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "ERROR: vLLM no respondió en ${MAX_WAIT}s"
        echo "--- últimas líneas del log vLLM ---"
        tail -20 logs/vllm_cc_${SLURM_JOBID}.out
        kill $VLLM_PID 2>/dev/null
        exit 1
    fi
done
echo "vLLM listo (${ELAPSED}s)"
curl -s http://localhost:8000/v1/models \
    | python -c "import sys,json; d=json.load(sys.stdin); print('Modelo activo:', d['data'][0]['id'])"

# ── Evaluación ContextCite ─────────────────────────────────────────────────────
echo ""
echo "=== ContextCite attribution (${N_ANSWERS} items, 64 ablaciones/item) ==="
python eval/context_cite_eval.py \
    --vllm-url      http://localhost:8000 \
    --model         "$MODEL_ID" \
    --answers       eval/rag_answers.json \
    --out           "$OUT_FILE" \
    --num-ablations 64 \
    --mask-prob     0.5 \
    --wait-vllm     0

EXIT_CODE=$?

# ── Apagar vLLM ───────────────────────────────────────────────────────────────
echo ""
echo "=== Deteniendo vLLM ==="
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

# ── Resumen ────────────────────────────────────────────────────────────────────
echo ""
echo "=== RESUMEN context_cite_results.json ==="
python - <<'PYEOF'
import json, pathlib, sys

p = pathlib.Path("eval/context_cite_results.json")
if not p.exists():
    print("No se generó context_cite_results.json")
    sys.exit(1)

r = json.loads(p.read_text())
agg = r.get("aggregate", {})
cfg = r.get("config", {})

print(f"  Modelo             : {cfg.get('model')}")
print(f"  Ablaciones/item    : {cfg.get('num_ablations')}")
print(f"  Items válidos      : {agg['n_valid']}/{agg['n_items']}  (errores: {agg['n_errors']})")
print()
print(f"  Mean top-1 score   : {agg['mean_top1_score']:.4f}   (fracción de atribución en el chunk más citado)")
print(f"  Mean entropy       : {agg['mean_entropy']:.4f}   (baja = concentrada, alta = difusa)")
print(f"  Mean concentration : {agg['mean_concentration']:.4f}   (1 = todo en un chunk, 0 = uniforme)")
print(f"  Mean chunks >10%   : {agg['mean_n_chunks_above_10pct']:.2f}    (avg chunks con >10% atribución)")
print(f"  Mean chunks >20%   : {agg['mean_n_chunks_above_20pct']:.2f}    (avg chunks con >20% atribución)")
print(f"  Mean ablaciones OK : {agg['mean_valid_ablations']:.1f} / {cfg.get('num_ablations')}")

# Distribución de concentration por topic
items = [x for x in r["per_item"] if "error" not in x]
by_topic = {}
for x in items:
    t = x.get("topic", "unknown")
    by_topic.setdefault(t, []).append(x["concentration"])

print()
print("  Concentración por topic:")
for topic, vals in sorted(by_topic.items(), key=lambda kv: -sum(kv[1])/len(kv[1])):
    print(f"    {topic:<25} {sum(vals)/len(vals):.4f}  (n={len(vals)})")
PYEOF

echo ""
echo "Resultados guardados en: $OUT_FILE"

exit $EXIT_CODE
