#!/bin/bash
# run_context_cite.sh — Ejecuta ContextCite en servidor sin SLURM.
#
# Uso:
#   bash run_context_cite.sh [--smoke]
#
# --smoke : procesa solo 5 items (16 ablaciones) para verificar que funciona
# Sin flag : evaluación completa (200 items, 64 ablaciones)
#
# Requisitos previos:
#   - venv activado: source .venv/bin/activate
#   - Llama 3.1-8B descargado (ver setup_nuevo_servidor.sh)
#   - eval/rag_answers.json presente (viene del git)

set -euo pipefail

SMOKE=0
if [[ "${1:-}" == "--smoke" ]]; then
    SMOKE=1
fi

# ── Configuración de rutas ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export HF_HOME="${HF_HOME:-$SCRIPT_DIR/models/huggingface}"
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-$SCRIPT_DIR/models}"
export SENTENCE_TRANSFORMERS_DEVICE=cpu
export HF_HUB_OFFLINE=1

# VLLM_USE_V1=0 solo es necesario en x86_64 con torch+cu124 (ABI mismatch).
# En aarch64 (DGX Spark) con vLLM moderno no hace falta.
VLLM_V1_FLAG="${SCRIPT_DIR}/.vllm_v1_needed"
if [ -f "$VLLM_V1_FLAG" ] && [ "$(cat "$VLLM_V1_FLAG")" = "1" ]; then
    export VLLM_USE_V1=0
    echo "VLLM_USE_V1=0 (x86_64 + cu124)"
fi

MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"

# Buscar el snapshot del modelo en el caché local (compatible con cache_dir y con HF_HOME/hub/).
# Si se encuentra, pasar la ruta absoluta a vLLM para evitar problemas de HF_HOME vs cache_dir.
MODEL_LOCAL=$(find "$SCRIPT_DIR/models" -name "config.json" -path "*Llama-3.1-8B*" 2>/dev/null \
    | head -1 | xargs -I{} dirname {} 2>/dev/null || true)
if [ -n "$MODEL_LOCAL" ] && [ -f "$MODEL_LOCAL/config.json" ]; then
    VLLM_MODEL="$MODEL_LOCAL"
    echo "Modelo local encontrado: $VLLM_MODEL"
else
    VLLM_MODEL="$MODEL_ID"
    echo "Usando modelo HuggingFace ID: $VLLM_MODEL"
fi

# ── Verificar GPU ─────────────────────────────────────────────────────────────
echo "=== GPU disponible ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
    || echo "nvidia-smi no disponible — ¿CPU only?"

# ── Verificar prerequisitos ────────────────────────────────────────────────────
echo ""
echo "=== Verificando prerequisitos ==="
python -c "import vllm, sklearn, openai, transformers; print(f'vllm={vllm.__version__} sklearn={sklearn.__version__} openai={openai.__version__} transformers={transformers.__version__}')"
if [ ! -f "eval/rag_answers.json" ]; then
    echo "ERROR: eval/rag_answers.json no existe — hacer git pull primero"
    exit 1
fi
echo "rag_answers.json: $(python -c "import json; print(len(json.load(open('eval/rag_answers.json'))), 'items')")"

# ── Arrancar vLLM ─────────────────────────────────────────────────────────────
mkdir -p logs
echo ""
echo "=== Arrancando vLLM (${VLLM_MODEL}) ==="
python -m vllm.entrypoints.openai.api_server \
    --model "$VLLM_MODEL" \
    --port 8000 \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    > logs/vllm_context_cite_$(date +%Y%m%d_%H%M%S).out 2>&1 &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# Asegurar que vLLM se mata al salir del script
trap "echo 'Deteniendo vLLM...'; kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null" EXIT

# ── Esperar a que vLLM esté listo ─────────────────────────────────────────────
echo "Esperando a que vLLM responda en localhost:8000..."
MAX_WAIT=300
ELAPSED=0
until curl -s http://localhost:8000/v1/models > /dev/null 2>&1; do
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "ERROR: vLLM no respondió en ${MAX_WAIT}s"
        exit 1
    fi
done
echo "vLLM listo (${ELAPSED}s)"
curl -s http://localhost:8000/v1/models \
    | python -c "import sys,json; d=json.load(sys.stdin); print('Modelo activo:', d['data'][0]['id'])"

# ── Ejecutar ContextCite ──────────────────────────────────────────────────────
echo ""
if [ $SMOKE -eq 1 ]; then
    echo "=== SMOKE TEST: 5 items, 16 ablaciones ==="
    python eval/context_cite_eval.py \
        --vllm-url      http://localhost:8000 \
        --model         "$MODEL_ID" \
        --answers       eval/rag_answers.json \
        --out           eval/context_cite_smoke.json \
        --num-ablations 16 \
        --mask-prob     0.5 \
        --max-items     5 \
        --wait-vllm     0
else
    echo "=== EVALUACIÓN COMPLETA: 200 items, 64 ablaciones ==="
    python eval/context_cite_eval.py \
        --vllm-url      http://localhost:8000 \
        --model         "$MODEL_ID" \
        --answers       eval/rag_answers.json \
        --out           eval/context_cite_results.json \
        --num-ablations 64 \
        --mask-prob     0.5 \
        --wait-vllm     0
fi

# ── Resumen final ─────────────────────────────────────────────────────────────
echo ""
echo "=== RESUMEN ==="
if [ $SMOKE -eq 1 ]; then
    python - <<'PYEOF'
import json, pathlib
p = pathlib.Path("eval/context_cite_smoke.json")
if not p.exists(): print("No se generó el JSON"); exit(1)
r = json.loads(p.read_text())
agg = r["aggregate"]
print(f"Items: {agg['n_valid']}/{agg['n_items']}  errores={agg['n_errors']}")
for item in r["per_item"]:
    if "error" in item:
        print(f"  {item['id']}: ERROR — {item['error']}")
    else:
        print(f"  {item['id']} top1={item['top1_score']:.3f} chunk={item['top1_chunk']} conc={item['concentration']:.3f} chunks:{[f'{s:.2f}' for s in item['chunk_scores_norm']]}")
PYEOF
else
    python - <<'PYEOF'
import json, pathlib
p = pathlib.Path("eval/context_cite_results.json")
if not p.exists(): print("No se generó el JSON"); exit(1)
r = json.loads(p.read_text())
agg = r["aggregate"]
print(f"Items válidos:      {agg['n_valid']}/{agg['n_items']}")
if agg["mean_top1_score"]:
    print(f"Mean top-1 score:   {agg['mean_top1_score']:.4f}")
    print(f"Mean concentration: {agg['mean_concentration']:.4f}")
    print(f"Mean entropy:       {agg['mean_entropy']:.4f}")
    print(f"Mean chunks >10%:   {agg['mean_n_chunks_above_10pct']:.2f}")
items = [x for x in r["per_item"] if "error" not in x]
by_topic = {}
for x in items:
    by_topic.setdefault(x.get("topic","?"), []).append(x["concentration"])
print("\nConcentración por topic:")
for t, vs in sorted(by_topic.items(), key=lambda kv: -sum(kv[1])/len(kv[1])):
    print(f"  {t:<25} {sum(vs)/len(vs):.4f}  (n={len(vs)})")
PYEOF
fi
