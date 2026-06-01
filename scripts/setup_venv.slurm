#!/bin/bash
#SBATCH --job-name=xai_rag_setup
#SBATCH --partition=dgx
#SBATCH --gres=gpu:0
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/xai_rag_setup_%j.out
#SBATCH --error=logs/xai_rag_setup_%j.err

cd "$SLURM_SUBMIT_DIR"
VENV=".venv312"

echo "Python3 del nodo: $(python3 --version) — $(which python3)"
echo "Creando $VENV ..."

# python3 -m venv primero; fallback a virtualenv si ensurepip no está
if ! python3 -m venv "$VENV" 2>/dev/null; then
    echo "python3 -m venv falló, probando virtualenv..."
    virtualenv -p python3 "$VENV"
fi

source "$VENV/bin/activate"
echo "venv Python: $(python --version)"

echo "=== pip install requirements ==="
pip install --quiet -r requirements.txt

echo "=== pip install torch 2.6.0+cu124 (driver 570.x, CUDA 12.8) ==="
pip install --quiet "torch==2.6.0+cu124" --index-url https://download.pytorch.org/whl/cu124

echo "=== verificación ==="
python -c "
import requests, chromadb, sentence_transformers, torch
print(f'requests {requests.__version__}  OK')
print(f'chromadb {chromadb.__version__}  OK')
print(f'torch {torch.__version__}  CUDA disponible: {torch.cuda.is_available()}')
"
echo "Setup completado."
