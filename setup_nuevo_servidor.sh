#!/bin/bash
# setup_nuevo_servidor.sh — Configura el entorno en un servidor sin SLURM.
#
# Compatible con:
#   - x86_64 + CUDA 12.x  (ruedas cu124 de PyTorch + vLLM 0.8.5.post1)
#   - aarch64 + CUDA 13.x (NVIDIA GB10 / DGX Spark — ruedas PyPI + vLLM latest)
#
# Uso:
#   bash setup_nuevo_servidor.sh
#   export HF_TOKEN=hf_xxx && bash setup_nuevo_servidor.sh   # para descargar Llama

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Detectar plataforma ───────────────────────────────────────────────────────
ARCH=$(uname -m)
echo "=== Plataforma: $ARCH ==="

if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || true
    CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9]+" || echo "0")
    echo "CUDA major: $CUDA_VER"
    HAS_GPU=1
else
    echo "No GPU NVIDIA detectada"
    CUDA_VER=0
    HAS_GPU=0
fi

# ── Crear virtualenv ──────────────────────────────────────────────────────────
echo ""
echo "=== Creando virtualenv .venv ==="
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo ".venv creado"
else
    echo ".venv ya existe — actualizando"
fi
source .venv/bin/activate
echo "Python: $(which python) — $(python --version)"
pip install --upgrade pip --quiet

# ── Instalar PyTorch ──────────────────────────────────────────────────────────
echo ""
echo "=== Instalando PyTorch ==="

if [ $HAS_GPU -eq 0 ]; then
    echo "CPU only — instalando torch sin CUDA"
    pip install --quiet torch

elif [ "$ARCH" = "aarch64" ]; then
    # ARM (NVIDIA GB10 / DGX Spark / Jetson):
    # Las ruedas cu124 no existen para aarch64. Usar PyPI directamente
    # que provee torch con soporte CUDA nativo para ARM.
    echo "aarch64 detectado — instalando torch desde PyPI (soporta CUDA $CUDA_VER)"
    pip install --quiet torch

else
    # x86_64: pinear cu124 para compatibilidad con DGX IUII (CUDA 12.8)
    # Si CUDA >= 13, también podemos usar cu124 (CUDA es backward-compatible)
    echo "x86_64 detectado — instalando torch 2.6.0+cu124"
    pip install --quiet \
        "torch==2.6.0+cu124" \
        --index-url https://download.pytorch.org/whl/cu124
fi

python -c "import torch; print(f'torch {torch.__version__}  CUDA disponible: {torch.cuda.is_available()}')"

# ── Instalar vLLM ─────────────────────────────────────────────────────────────
echo ""
echo "=== Instalando vLLM ==="

if [ "$ARCH" = "aarch64" ]; then
    # aarch64 + CUDA 13: vLLM necesita compilar fastsafetensors (extensión C++).
    # Requiere python3.12-dev. Intentamos instalar si falta.
    echo "aarch64: verificando python3.12-dev (necesario para compilar vLLM deps)"
    if ! dpkg -l python3.12-dev &>/dev/null 2>&1; then
        echo "  Instalando python3.12-dev..."
        sudo apt-get install -y python3.12-dev 2>/dev/null \
            || { echo "  AVISO: no hay sudo o apt falló — intentando igualmente"; }
    else
        echo "  python3.12-dev ya instalado"
    fi
    echo "aarch64: instalando vLLM sin pin de versión"
    pip install --quiet vllm
    VLLM_V1_NEEDED=0   # vLLM moderno usa V1 por defecto sin problemas en ARM
else
    # x86_64 con torch 2.6.0+cu124: solo vLLM 0.8.5.post1 es compatible
    echo "x86_64: instalando vLLM 0.8.5.post1 (compatible con torch 2.6.0)"
    pip install --quiet "vllm==0.8.5.post1"
    # transformers 5.x eliminó all_special_tokens_extended que usa vLLM 0.8.x
    pip install --quiet "transformers>=4.51.1,<5.0.0"
    VLLM_V1_NEEDED=1   # V1 tiene ABI mismatch en cu124 — forzar V0
fi

# ── Resto de dependencias ─────────────────────────────────────────────────────
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

# ── Guardar flag de arquitectura para run_context_cite.sh ────────────────────
echo "$VLLM_V1_NEEDED" > .vllm_v1_needed
echo "ARCH=$ARCH  VLLM_USE_V1=$VLLM_V1_NEEDED"

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
if [ -z "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN no definido — saltando descarga de Llama."
    echo "Para descargar:"
    echo "  export HF_TOKEN=hf_xxx && bash setup_nuevo_servidor.sh"
    echo "  (acepta la licencia en https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)"
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

# ── Resumen ────────────────────────────────────────────────────────────────────
echo ""
echo "==========================================="
echo "  Setup completado"
echo "==========================================="
echo ""
echo "Próximos pasos:"
echo "  1. source .venv/bin/activate"
echo "  2. bash run_context_cite.sh --smoke   # verifica (5 items, ~15 min)"
echo "  3. bash run_context_cite.sh           # evaluación completa (200 items)"
