#!/bin/bash
# ============================================================
# download_llama_dgx.sh
# Ejecutar en el NODO DE ACCESO del DGX (sin GPU, sin SLURM)
# Requiere: huggingface-cli, token HF con acceso a Llama 3.1
# ============================================================

set -euo pipefail

MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
CACHE_DIR="${HOME}/.cache/huggingface/hub"

echo "=================================================="
echo "  Descarga: ${MODEL_ID}"
echo "  Destino:  ${CACHE_DIR}"
echo "=================================================="

# --- 1. Verificar token --------------------------------
if [ -z "${HF_TOKEN:-}" ]; then
  echo ""
  echo "ERROR: Variable HF_TOKEN no definida."
  echo "Ejecuta primero:  export HF_TOKEN=hf_xxxxxxxxxxxx"
  exit 1
fi

# --- 2. Login en HuggingFace --------------------------
echo "[1/3] Autenticando en HuggingFace..."
huggingface-cli login --token "${HF_TOKEN}"

# --- 3. Descargar modelo ------------------------------
echo "[2/3] Descargando ${MODEL_ID}..."
echo "      (Los pesos pesan ~16 GB — puede tardar 20-40 min según red)"
echo ""

huggingface-cli download "${MODEL_ID}" \
  --cache-dir "${CACHE_DIR}" \
  --include "*.safetensors" "*.json" "tokenizer*" \
  --resume-download

# --- 4. Verificar descarga ----------------------------
echo ""
echo "[3/3] Verificando archivos descargados..."
find "${CACHE_DIR}" -path "*Llama-3.1-8B-Instruct*" -name "*.safetensors" | sort

echo ""
echo "✓ Descarga completada."
echo ""
echo "Ruta para vLLM:"
echo "  --model ${CACHE_DIR}/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/<hash>"
echo ""
echo "Próximo paso: scripts/launch_vllm.sh"
