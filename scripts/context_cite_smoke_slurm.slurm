#!/bin/bash
#SBATCH --job-name=xai_rag_cc_smoke
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/xai_rag_cc_smoke_%j.out
#SBATCH --error=logs/xai_rag_cc_smoke_%j.err

# Smoke test de ContextCite: procesa solo 5 items con 16 ablaciones.
# Si termina OK y el JSON es coherente, lanzar context_cite_slurm.slurm.

set -euo pipefail

nvidia-smi

source "$SLURM_SUBMIT_DIR/.venv312/bin/activate"
echo "Python: $(which python) — $(python --version)"

cd "$SLURM_SUBMIT_DIR"

export HF_HOME="$SLURM_SUBMIT_DIR/models/huggingface"
export HF_HUB_OFFLINE=1
export SENTENCE_TRANSFORMERS_DEVICE=cpu
export VLLM_USE_V1=0

MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"

# ── Arrancar vLLM ─────────────────────────────────────────────────────────────
echo "=== Arrancando vLLM ==="
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --port 8000 \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    > logs/vllm_smoke_${SLURM_JOBID}.out 2>&1 &
VLLM_PID=$!

MAX_WAIT=300; ELAPSED=0
until curl -s http://localhost:8000/v1/models > /dev/null 2>&1; do
    sleep 5; ELAPSED=$((ELAPSED + 5))
    [ $ELAPSED -ge $MAX_WAIT ] && { echo "ERROR: vLLM timeout"; kill $VLLM_PID 2>/dev/null; exit 1; }
done
echo "vLLM listo (${ELAPSED}s)"

# ── Smoke test: 5 items, 16 ablaciones ────────────────────────────────────────
echo ""
echo "=== Smoke test ContextCite (5 items, 16 ablaciones) ==="
python eval/context_cite_eval.py \
    --vllm-url      http://localhost:8000 \
    --model         "$MODEL_ID" \
    --answers       eval/rag_answers.json \
    --out           eval/context_cite_smoke.json \
    --num-ablations 16 \
    --mask-prob     0.5 \
    --max-items     5 \
    --wait-vllm     0

EXIT_CODE=$?

kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo ""
echo "=== Resultado smoke test ==="
python - <<'PYEOF'
import json, pathlib
p = pathlib.Path("eval/context_cite_smoke.json")
if not p.exists():
    print("FALLO: no se generó el JSON"); exit(1)

r = json.loads(p.read_text())
agg = r["aggregate"]
print(f"Items procesados: {agg['n_valid']}/{agg['n_items']}  errores={agg['n_errors']}")

for item in r["per_item"]:
    if "error" in item:
        print(f"  {item['id']}: ERROR — {item['error']}")
        continue
    scores = item["chunk_scores_norm"]
    print(
        f"  {item['id']} ({item['topic']}) "
        f"top1={item['top1_score']:.3f} chunk={item['top1_chunk']} "
        f"conc={item['concentration']:.3f} "
        f"abl_ok={item['n_valid_ablations']}/16 "
        f"chunks: {[f'{s:.2f}' for s in scores]}"
    )

if agg["n_errors"] == 0:
    print()
    print("✓ Smoke test OK — puedes lanzar context_cite_slurm.slurm")
else:
    print()
    print(f"✗ {agg['n_errors']} errores — revisar antes de lanzar el job completo")
PYEOF

exit $EXIT_CODE
