#!/bin/bash
#SBATCH --job-name=xai_rag_setup_ragas
#SBATCH --partition=dgx
#SBATCH --gres=gpu:0
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/xai_rag_setup_ragas_%j.out
#SBATCH --error=logs/xai_rag_setup_ragas_%j.err

cd "$SLURM_SUBMIT_DIR"
source .venv312/bin/activate
echo "Python: $(python --version)"

# Reinstalar torch 2.6.0+cu124 (driver 570.x, CUDA 12.8)
# vllm 0.8.5.post1 requiere torch==2.6.0; 0.21.0 requiere torch==2.11.0+cu13 (incompatible)
echo "=== pip install torch 2.6.0+cu124 ==="
pip install --quiet --force-reinstall \
    "torch==2.6.0+cu124" "torchaudio==2.6.0+cu124" \
    --index-url https://download.pytorch.org/whl/cu124

echo "=== pip install vllm==0.8.5.post1 ==="
pip install --quiet "vllm==0.8.5.post1"

# transformers 5.x eliminó all_special_tokens_extended que usa vLLM 0.8.x
echo "=== pip install transformers 4.x compatible ==="
pip install --quiet "transformers>=4.51.1,<5.0.0"

echo "=== pip install langchain deps ==="
pip install --quiet langchain-openai langchain-community

echo "=== verificación ==="
python -c "
import torch, vllm, ragas, langchain_openai
print(f'torch    {torch.__version__}   CUDA: {torch.cuda.is_available()}')
print(f'vllm     {vllm.__version__}  OK')
print(f'ragas    {ragas.__version__}  OK')
"

echo ""
echo "=== Modelo Llama en caché ==="
export HF_HOME="$SLURM_SUBMIT_DIR/models/huggingface"
find "$HF_HOME" -path "*/Llama-3.1-8B-Instruct/config.json" 2>/dev/null \
    | head -3 || echo "(no encontrado)"

echo "Setup RAGAS completado."
