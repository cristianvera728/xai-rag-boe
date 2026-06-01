#!/bin/bash
# setup_nuevo_servidor.sh — Configura el entorno en un servidor sin SLURM.
#
# Ejecutar DESDE el directorio raíz del proyecto (después del git clone):
#   bash setup_nuevo_servidor.sh
#
# Lo que hace:
#   1. Crea virtualenv .venv con Python 3.12
#   2. Instala torch 2.6.0+cu124 (o CPU si no hay CUDA 12.x)
#   3. Instala vLLM 0.8.5.post1 + transformers 4.x + RAGAS + demás deps
#   4. Descarga bge-m3 y Llama-3.1-8B al caché local
#
# Después de ejecutar este script:
#   source .venv/bin/activate
#   bash run_context_cite.sh --smoke   # verificar
#   bash run_context_cite.sh           # evaluación completa

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Detectar CUDA ─────────────────────────────────────────────────────────────
echo "=== Verificando GPU y CUDA ==="
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    CUDA_VER=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" || echo "unknown")
    echo "CUDA Version: $CUDA_VER"
    HAS_GPU=1
else
    echo "No se detectó GPU NVIDIA — instalando PyTorch CPU (ContextCite será lento)"
    HAS_GPU=0
fi

# ── Crear virtualenv ──────────────────────────────────────────────────────────
echo ""
echo "=== Creando virtualenv .venv ==="
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo ".venv creado"
else
    echo ".venv ya existe, continuando"
fi
source .venv/bin/activate
echo "Python: $(which python) — $(python --version)"

# ── Instalar PyTorch ──────────────────────────────────────────────────────────
echo ""
echo "=== Instalando PyTorch ==="
pip install --upgrade pip --quiet

if [ $HAS_GPU -eq 1 ]; then
    # torch 2.6.0+cu124 funciona con CUDA 12.x (driver >= 525)
    pip install --quiet \
        "torch==2.6.0+cu124" \
        "torchaudio==2.6.0+cu124" \
        --index-url https://download.pytorch.org/whl/cu124
else
    pip install --quiet "torch==2.6.0" --index-url https://download.pytorch.org/whl/cpu
fi

python -c "import torch; print(f'torch {torch.__version__}  CUDA: {torch.cuda.is_available()}')"

# ── Instalar vLLM ─────────────────────────────────────────────────────────────
echo ""
echo "=== Instalando vLLM 0.8.5.post1 ==="
# IMPORTANTE: vLLM 0.8.5.post1 requiere torch==2.6.0 exactamente
# Las versiones más nuevas de vLLM requieren CUDA 13 (incompatible con drivers < 570)
pip install --quiet "vllm==0.8.5.post1"

# transformers 5.x eliminó all_special_tokens_extended que usa vLLM 0.8.x
echo "=== Instalando transformers 4.x compatible ==="
pip install --quiet "transformers>=4.51.1,<5.0.0"

# ── Instalar resto de dependencias ────────────────────────────────────────────
echo ""
echo "=== Instalando dependencias del proyecto ==="
pip install --quiet \
    ragas==0.4.3 \
    langchain-openai \
    langchain-community \
    sentence-transformers \
    chromadb \
    "scikit-learn>=1.3.0" \
    openai \
    huggingface_hub \
    requests \
    lxml \
    numpy \
    tqdm

# ── Verificación del stack ────────────────────────────────────────────────────
echo ""
echo "=== Verificación del stack ==="
python -c "
import torch, vllm, ragas, sklearn, openai
print(f'torch         {torch.__version__}  CUDA: {torch.cuda.is_available()}')
print(f'vllm          {vllm.__version__}')
print(f'ragas         {ragas.__version__}')
print(f'scikit-learn  {sklearn.__version__}')
print(f'openai        {openai.__version__}')
"

# ── Crear directorios de caché ─────────────────────────────────────────────────
mkdir -p models/huggingface logs

export HF_HOME="$SCRIPT_DIR/models/huggingface"
export SENTENCE_TRANSFORMERS_HOME="$SCRIPT_DIR/models"

# ── Descargar bge-m3 ──────────────────────────────────────────────────────────
echo ""
echo "=== Descargando BAAI/bge-m3 (~570MB) ==="
python -c "
from sentence_transformers import SentenceTransformer
import os
os.environ['SENTENCE_TRANSFORMERS_HOME'] = '$SCRIPT_DIR/models'
m = SentenceTransformer('BAAI/bge-m3', device='cpu')
print('bge-m3 descargado OK')
"

# ── Descargar Llama 3.1-8B ────────────────────────────────────────────────────
echo ""
echo "=== Descargando Llama 3.1-8B-Instruct (~15GB) ==="
echo "NOTA: requiere HF_TOKEN con acceso aceptado a meta-llama/Llama-3.1-8B-Instruct"
echo "      Si no tienes token, acepta la licencia en: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct"
echo ""

if [ -z "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN no está definido. Opciones:"
    echo "  export HF_TOKEN=hf_xxx && bash setup_nuevo_servidor.sh"
    echo "  O ejecutar manualmente:"
    echo "  python -c \"from huggingface_hub import snapshot_download; snapshot_download('meta-llama/Llama-3.1-8B-Instruct', token='hf_xxx', ignore_patterns=['*.pt','original/*'])\""
    echo ""
    echo "Continuando sin descargar Llama (necesario para ContextCite)."
else
    python -c "
from huggingface_hub import snapshot_download
import os
path = snapshot_download(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    token=os.environ.get('HF_TOKEN'),
    ignore_patterns=['*.pt', 'original/*'],
    cache_dir='$SCRIPT_DIR/models/huggingface',
)
print(f'Llama descargado en: {path}')
"
fi

# ── Resumen final ─────────────────────────────────────────────────────────────
echo ""
echo "==========================================="
echo "  Setup completado"
echo "==========================================="
echo ""
echo "Próximos pasos:"
echo "  1. source .venv/bin/activate"
echo "  2. bash run_context_cite.sh --smoke   # smoke test (5 items)"
echo "  3. bash run_context_cite.sh           # evaluación completa (200 items)"
echo ""
echo "Para transferir ChromaDB desde el DGX (si necesitas RAGAS o MACS):"
echo "  scp -r cvera@dgx.iuii.ua.es:/path/to/xai-rag-boe/chroma_boe/ ."
