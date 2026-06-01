#!/bin/bash
#SBATCH --job-name=xai_rag_download_llama
#SBATCH --partition=dgx
#SBATCH --gres=gpu:0
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/xai_rag_download_llama_%j.out
#SBATCH --error=logs/xai_rag_download_llama_%j.err

# Descarga Llama 3.1-8B-Instruct desde HuggingFace Hub al caché local del proyecto.
# Requiere que HF_TOKEN esté definido en el entorno o pasado con --export:
#   sbatch --export=ALL,HF_TOKEN=hf_xxxx scripts/download_llama_slurm.slurm

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source .venv312/bin/activate
echo "Python: $(python --version)"

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN no definido. Lanza con:"
    echo "  sbatch --export=ALL,HF_TOKEN=hf_xxxx scripts/download_llama_slurm.slurm"
    exit 1
fi

MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
export HF_HOME="$SLURM_SUBMIT_DIR/models/huggingface"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

echo "=== Descargando $MODEL_ID ==="
echo "HF_HOME: $HF_HOME"

python -c "
from huggingface_hub import snapshot_download
import os

model_id = '$MODEL_ID'
print(f'Descargando {model_id}')
path = snapshot_download(
    repo_id=model_id,
    token=os.environ['HUGGING_FACE_HUB_TOKEN'],
    ignore_patterns=['*.pt', 'original/*'],   # solo safetensors
)
print(f'OK: {path}')
"

echo ""
echo "=== Config encontrado ==="
find "$HF_HOME" -path "*/Llama-3.1-8B-Instruct/config.json" | head -3

echo ""
echo "=== Espacio ocupado ==="
du -sh "$HF_HOME/hub/models--meta-llama--Llama-3.1-8B-Instruct/" 2>/dev/null || echo "(path no encontrado)"

echo "Descarga completada."
