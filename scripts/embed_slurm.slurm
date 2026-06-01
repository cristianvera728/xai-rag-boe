#!/bin/bash
#SBATCH --job-name=xai_rag_embed
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/xai_rag_embed_%j.out
#SBATCH --error=logs/xai_rag_embed_%j.err

nvidia-smi

# Conda no está instalado en los nodos DGX — usar el venv del proyecto
source "$SLURM_SUBMIT_DIR/.venv312/bin/activate"
echo "Python: $(which python) — $(python --version)"

python -c "
import torch, sys
v = torch.__version__
print(f'torch {v}  CUDA disponible: {torch.cuda.is_available()}')
if 'cu124' not in v:
    print('ERROR: torch no es cu124 — relanza setup_venv.slurm primero', file=sys.stderr)
    sys.exit(1)
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

# Parchear device a cuda; usar caché local para no descargar en cada job
export SENTENCE_TRANSFORMERS_DEVICE=cuda
export SENTENCE_TRANSFORMERS_HOME="$SLURM_SUBMIT_DIR/models"
export HF_HOME="$SLURM_SUBMIT_DIR/models/huggingface"

cd "$SLURM_SUBMIT_DIR"

python ingest_from_kimi.py --dataset dataset_qa_boe.json

echo ""
echo "=== ingest_report.json ==="
python -c "
import json, pathlib
r = json.loads(pathlib.Path('ingest_report.json').read_text())
print(f\"Docs OK    : {r['docs_ok']}/{r['docs_total']}\")
print(f\"Chunks gen : {r['chunks_generated']}\")
print(f\"Upserted   : {r['chunks_upserted']}\")
print(f\"Total DB   : {r['total_in_chroma']}\")
if r['docs_failed']:
    print(f\"Fallidos   : {r['docs_failed']}\")
"
