# XAI-RAG-AAPP — Contexto del proyecto

## Objetivo
Paper: "Beyond Citation: Evaluating Explanation Faithfulness in 
Spanish-Language RAG Systems for Public Administration"
Cátedra ENIA, Universidad de Alicante.

## Stack
- Modelo: Llama 3.1-8B-Instruct (DGX vía vLLM)
- Corpus: BOE indexado en ChromaDB con bge-m3
- Evaluación: RAGAS + ContextCite + MACS

---

## ⚠️ INFRAESTRUCTURA: DGX IUII - Universidad de Alicante

**REGLA CRÍTICA: Este proyecto corre en una máquina compartida (DGX IUII-UA).
NUNCA ejecutar cómputo intensivo directamente en el nodo de acceso.
TODO proceso que use GPU, entrene modelos o procese datasets grandes
DEBE lanzarse mediante SLURM con sbatch.**

### Gestor de colas: SLURM
- Partición: `dgx` (OBLIGATORIO en todos los jobs)
- Tiempo máximo: 2 días
- Sin `--mem`, el job no se planifica hasta nodo completamente libre

### Plantilla SLURM obligatoria para este proyecto
```bash
#!/bin/bash
#SBATCH --job-name=xai_rag_<nombre>
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/xai_rag_<nombre>_%j.out
#SBATCH --error=logs/xai_rag_<nombre>_%j.err

nvidia-smi
source miniconda3/bin/activate
conda activate xai-rag

# código aquí
```

### Entorno
- Gestor: Conda (`~/miniconda3/`)
- Entorno del proyecto: `xai-rag`
- Activar: `source miniconda3/bin/activate && conda activate xai-rag`

### Comandos útiles
```bash
sbatch scripts/nombre.slurm    # lanzar job
squeue -u $USER                # ver estado
scancel <job_id>               # cancelar
tail -f logs/<job_id>.out      # ver logs en tiempo real
```

### Checklist antes de cualquier sbatch
- [ ] `--partition=dgx` presente
- [ ] `--mem` especificado
- [ ] `--cpus-per-task` especificado  
- [ ] `--time` estimado
- [ ] GPUs en `--gres` = GPUs que el código realmente usa
- [ ] Logs en `logs/` con `%j` en el nombre

### Docker (si aplica)
- Nombre del contenedor DEBE empezar por nombre de usuario
- Formato: `$USER_<identificador>`
- Contenedores sin este prefijo son eliminados por el IUII

---

## Estructura del proyecto
```
xai-rag-boe/
├── CLAUDE.md                    ← este fichero
├── dataset_qa_boe.json          ← 200 pares QA generados por Kimi
├── boe_pipeline/
│   └── boe_pipeline.py          ← scraping + chunking + ChromaDB
├── scripts/
│   ├── download_llama_dgx.sh    ← descarga Llama al DGX
│   └── *.slurm                  ← TODOS los jobs GPU van aquí
├── eval/                        ← (próximo) RAGAS + ContextCite + MACS
├── logs/                        ← salidas de jobs SLURM
└── requirements.txt
```