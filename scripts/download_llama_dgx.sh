#!/bin/bash
# Descarga Llama 3.1-8B-Instruct al caché HF local del proyecto.
# Requiere HF_TOKEN con acceso al modelo meta-llama/Llama-3.1-8B-Instruct.
#
# Uso:
#   export HF_TOKEN=hf_xxxx
#   bash scripts/download_llama_dgx.sh
#
# O lanzar como job SLURM sin GPU:
#   sbatch scripts/download_llama_slurm.slurm

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
HF_HOME_LOCAL="$PROJECT_DIR/models/huggingface"

echo "=== Descarga Llama 3.1-8B-Instruct ==="
echo "Destino: $HF_HOME_LOCAL"
echo "Modelo:  $MODEL_ID"
echo ""

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN no definido."
    echo "  export HF_TOKEN=hf_xxxx"
    exit 1
fi

source "$PROJECT_DIR/.venv312/bin/activate"

export HF_HOME="$HF_HOME_LOCAL"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

python -c "
from huggingface_hub import snapshot_download
import os

model_id = '$MODEL_ID'
hf_home  = os.environ['HF_HOME']

print(f'Descargando {model_id} → {hf_home}')
path = snapshot_download(
    repo_id=model_id,
    token=os.environ['HUGGING_FACE_HUB_TOKEN'],
    ignore_patterns=['*.pt', 'original/*'],   # solo safetensors
)
print(f'OK: {path}')
"

echo ""
echo "=== Verificando config.json ==="
find "$HF_HOME_LOCAL" -path "*/Llama-3.1-8B-Instruct/config.json" | head -3
echo "Descarga completada."
