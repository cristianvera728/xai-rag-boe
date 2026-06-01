# Proceso completo de construcción y evaluación del sistema RAG-BOE

**Proyecto:** "Beyond Citation: Evaluating Explanation Faithfulness in Spanish-Language RAG Systems for Public Administration"  
**Cátedra ENIA — Universidad de Alicante**  
**Fecha:** 2026-05-18

---

## Visión general

El objetivo es construir un sistema RAG (Retrieval-Augmented Generation) sobre normativa española y europea publicada en el BOE, evaluarlo con métricas estándar (RAGAS) y, posteriormente, aplicar métricas de explicabilidad (ContextCite, MACS) para el paper. Este documento describe paso a paso todo lo que se ha hecho para llegar a los resultados RAGAS finales.

```
dataset_qa_boe.json         ← punto de partida: 200 preguntas con respuesta
        │
        ▼
Ingesta de documentos        ← descarga BOE/EUR-Lex → chunking → ChromaDB
        │
        ▼
RAG Inference (Fase 1)       ← retrieval bge-m3 + generación Llama 3.1-8B
        │
        ▼
RAGAS Evaluation (Fase 2)    ← 4 métricas con Llama como juez
        │
        ▼
eval/ragas_results.json      ← resultados finales
```

---

## 1. Infraestructura: DGX IUII — Universidad de Alicante

### Máquina de cómputo

Todo el trabajo de GPU se ejecuta en el **DGX IUII** de la Universidad de Alicante, una máquina compartida que requiere el uso obligatorio de **SLURM** para cualquier proceso intensivo. Las especificaciones relevantes:

- **GPUs:** NVIDIA A100-SXM4-40GB
- **Driver NVIDIA:** 570.133.20 → CUDA máximo soportado: **12.8**
- **OS:** Linux (Debian/Ubuntu), Python del sistema: 3.12.3
- **Almacenamiento:** directorio home en NFS compartido entre todos los nodos

### Restricción crítica de SLURM

Ningún proceso con GPU puede ejecutarse directamente en el nodo de login. Todo se lanza mediante `sbatch`. La plantilla usada:

```bash
#SBATCH --partition=dgx          # obligatorio
#SBATCH --gres=gpu:1             # número real de GPUs usadas
#SBATCH --mem=48G                # sin esto el job no se planifica
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/nombre_%j.out
#SBATCH --error=logs/nombre_%j.err
```

### Entorno Python

El sistema Python del nodo de login es 3.11, pero los nodos de cómputo tienen 3.12.3. Se creó un **virtualenv** (no conda) en el propio directorio del proyecto para que esté disponible en NFS para todos los nodos:

```bash
python3.12 -m venv .venv312    # creado en el nodo DGX vía sbatch
```

**Problema crítico con PyTorch:** Al instalar `torch` sin versión fija, pip resolvía a `torch==2.12.0+cu130` (CUDA 13), incompatible con el driver 570.x (máximo CUDA 12.8). La solución fue pinear explícitamente:

```bash
pip install "torch==2.6.0+cu124" --index-url https://download.pytorch.org/whl/cu124
```

El sufijo `+cu124` indica que la librería está compilada para CUDA 12.4, que es compatible con cualquier driver que soporte CUDA 12.x o superior.

**Jobs de setup ejecutados:**
| Job | Descripción | Script |
|---|---|---|
| `setup_venv.slurm` | Crea `.venv312`, instala dependencias base + torch 2.6.0+cu124 | `scripts/setup_venv.slurm` |
| `setup_ragas.slurm` (×3) | Instala vLLM + langchain + transformers 4.x compatible | `scripts/setup_ragas.slurm` |

---

## 2. Dataset de evaluación

### Origen

El dataset `dataset_qa_boe.json` fue generado externamente por el modelo **Kimi** (Moonshot AI) a partir de los documentos normativos. **No se modifica en ningún momento** — es el gold standard del experimento.

### Estructura

- **200 pares QA** sobre normativa española y europea
- **10 topics** × 20 preguntas: `admin_electronica`, `contratacion_publica`, `empleo_publico`, `funcion_publica`, `hacienda`, `procedimiento_adm`, `proteccion_datos`, `regimen_local`, `subvenciones`, `transparencia`
- **3 niveles de dificultad:** baja (80), media (80), alta (40)
- **22 documentos únicos** referenciados (21 BOE + 1 DOUE)
- Cada item contiene: `id`, `question`, `ground_truth`, `doc_id`, `topic`, `difficulty`, `doc_url`

### Documentos normativos cubiertos (selección)

| doc_id | Documento |
|---|---|
| BOE-A-2015-10565 | Ley 39/2015, Procedimiento Administrativo Común |
| BOE-A-2015-11719 | Ley 40/2015, Régimen Jurídico del Sector Público |
| BOE-A-2017-12902 | Ley 9/2017, Contratos del Sector Público (LCSP) |
| BOE-A-2003-20977 | Ley 38/2003, General de Subvenciones |
| BOE-A-2013-12887 | Ley 19/2013, Transparencia y Buen Gobierno |
| BOE-A-1985-5392 | Ley 7/1985, Bases del Régimen Local |
| BOE-A-1985-151 | Ley Orgánica 2/1979, Tribunal Constitucional |
| DOUE-L-2016-80807 | Reglamento (UE) 2016/679 — RGPD |
| BOE-A-2018-16673 | Ley Orgánica 3/2018, LOPDGDD |

---

## 3. Pipeline de ingesta: BOE → ChromaDB

### Descripción general

El script `ingest_from_kimi.py` lee el dataset, extrae los `doc_id` únicos, descarga cada documento normativo, lo chunkea por artículo y lo indexa en ChromaDB con embeddings bge-m3.

### 3.1 Descarga de documentos

**Documentos BOE:** Se usa la API XML oficial del BOE:
```
https://www.boe.es/diario_boe/xml.php?id={doc_id}
```
El módulo `boe_pipeline.py` parsea el XML y extrae texto por artículo, conservando metadatos (`doc_id`, `titulo`, `fecha`, `url`).

**Documentos EUR-Lex (RGPD):** El RGPD no tiene `doc_id` de tipo `BOE-*`. Se descarga desde EUR-Lex en formato Formex4 XML:
```
https://eur-lex.europa.eu/legal-content/ES/TXT/XML/?uri=CELEX:32016R0679
```
Un parser específico (`_parse_eurlex_xml`) extrae artículos del formato Formex4, normalizando encabezados `Artículo N.` para garantizar la compatibilidad con el chunker.

### 3.2 Estrategia de chunking

El chunking se realiza en `boe_pipeline.py` con dos estrategias en cascada:

1. **Estrategia `articulo` (prioritaria):** Detecta artículos mediante regex (`Artículo \d+`) y divide el texto artículo por artículo. Cada chunk contiene un artículo completo con su título. Esta estrategia se aplica cuando el documento tiene estructura articulada reconocible.

2. **Estrategia `ventana` (fallback):** Si el texto no tiene artículos detectables, aplica ventana deslizante con solapamiento.

**Todos los 22 documentos** usaron la estrategia `articulo`, lo que garantiza que cada chunk representa una unidad semántica coherente (un artículo de ley).

### 3.3 Metadatos por chunk

Cada chunk almacenado en ChromaDB incluye:
```python
{
    "doc_id":     "BOE-A-2015-10565",
    "titulo":     "Ley 39/2015...",
    "fecha":      "20151002",
    "url":        "https://www.boe.es/...",
    "estrategia": "articulo",
    "chunk_idx":  0,
    "art_num":    "9",
    "art_titulo": "Sistemas de identificación de los interesados...",
    "qa_ids":     "qa_001,qa_002,qa_003"   # QA relacionadas (CSV)
}
```

El campo `qa_ids` permite trazar qué preguntas del dataset están relacionadas con cada chunk.

### 3.4 Modelo de embeddings: BAAI/bge-m3

- **Modelo:** `BAAI/bge-m3` — multilingüe, optimizado para recuperación en español
- **Device:** CPU durante la ingesta (para ceder la GPU íntegra al modelo de lenguaje)
- **Configuración:** `normalize_embeddings=True`
- **Caché local:** Descargado una vez a `models/huggingface/` y reutilizado en todos los jobs mediante `HF_HOME=$SLURM_SUBMIT_DIR/models/huggingface`

La variable de entorno `SENTENCE_TRANSFORMERS_DEVICE=cuda/cpu` controla el dispositivo sin modificar el código de `boe_pipeline.py`.

### 3.5 ChromaDB

- **Tipo:** `PersistentClient` con directorio `./chroma_boe`
- **Colección:** `boe_normativa`
- **Función de embedding:** `SentenceTransformerEmbeddingFunction` con bge-m3
- **Inserción:** `collection.upsert()` — idempotente, permite relanzar sin duplicar
- **Resume logic:** Antes de procesar cada documento, se consulta ChromaDB para obtener los `doc_id` ya presentes y saltarlos. Esto permite retomar la ingesta si un job SLURM se cancela por tiempo.

### 3.6 Resultado final de la ingesta

| Métrica | Valor |
|---|---|
| Documentos procesados | 22/22 (100%) |
| Documentos fallidos | 0 |
| Chunks generados | 5.427 |
| Chunks en ChromaDB | 5.427 |
| Estrategia dominante | artículo (100% de los documentos) |

**Evolución de jobs de ingesta:**

El proceso de ingesta requirió **4 iteraciones de jobs SLURM** por problemas que se fueron resolviendo:

| Job | Resultado | Problema | Solución |
|---|---|---|---|
| `embed 53047` | Cancelado (1h) | Torch sin pinear → descarga 49min de ruedas cu130 | Pinear `torch==2.6.0+cu124` en setup |
| `embed 53152` | Cancelado (2h) | Procesó 16/22 docs, se quedó sin tiempo | Aumentar `--time=04:00:00` |
| `embed 53243` | 0 chunks (bug) | Resume leía `ingest_report.json` del dry-run → todos skipped | Cambiar resume: consultar ChromaDB directamente |
| `embed 53387` | **OK** ✓ | — | 5.427 chunks indexados |

---

## 4. Descarga del modelo Llama 3.1-8B-Instruct

Para que vLLM no descargue el modelo en cada job (lento y dependiente de red), se descargó una vez al caché local del proyecto:

```bash
sbatch --export=ALL,HF_TOKEN=<token> scripts/download_llama_slurm.slurm
```

- **Destino:** `models/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/`
- **Tamaño:** ~15 GB (solo safetensors, se excluyen `.pt` y archivos `original/`)
- **Acceso:** requirió aceptar la licencia Llama 3.1 Community License en HuggingFace
- En todos los jobs posteriores se activa `HF_HUB_OFFLINE=1` para evitar llamadas a red

---

## 5. Instalación de vLLM: problemas de compatibilidad

La instalación de vLLM fue el proceso más complejo, con **3 iteraciones** de `setup_ragas.slurm` para resolver incompatibilidades de versiones:

### Intento 1: `pip install vllm` → vLLM 0.21.0

- pip instaló la versión más reciente disponible: **vLLM 0.21.0**
- Esta versión requiere `torch==2.11.0+cu13` (CUDA 13)
- **Problema:** el driver 570.x del DGX soporta CUDA máximo 12.8 → CUDA 13 no es compatible
- vLLM arrancaba pero crasheaba silenciosamente (log vacío)

### Intento 2: Downgrade a vLLM 0.8.5.post1 + torch 2.6.0+cu124

Se identificó que **vLLM 0.8.5.post1** requiere exactamente `torch==2.6.0` (compatible con cu124):

```bash
pip install --force-reinstall "torch==2.6.0+cu124" --index-url https://download.pytorch.org/whl/cu124
pip install "vllm==0.8.5.post1"
```

- **Nuevo problema:** `transformers==5.8.1` (instalado por otras dependencias) eliminó el atributo `all_special_tokens_extended` que vLLM 0.8.x usa internamente → `AttributeError` al inicializar el tokenizador

### Intento 3: Downgrade de transformers a 4.x

```bash
pip install "transformers>=4.51.1,<5.0.0"
```

- **Nuevo problema:** El motor **V1** de vLLM 0.8.5.post1 falla con `undefined symbol` en `torch_c_dlpack_addon_torch26-cuda.so` — incompatibilidad de ABI entre el binario precompilado y nuestro torch+cu124

### Solución final: forzar motor V0

```bash
export VLLM_USE_V1=0
```

El motor V0 de vLLM no carga la librería `torch_c_dlpack_ext` problemática. Con esta configuración, vLLM arrancó correctamente en **130 segundos** y comenzó a servir peticiones.

**Stack final de dependencias:**
```
torch==2.6.0+cu124
vllm==0.8.5.post1
transformers>=4.51.1,<5.0.0
ragas==0.4.3
langchain-openai
langchain-community
sentence-transformers (bge-m3)
```

---

## 6. Pipeline de evaluación RAGAS

El script `eval/ragas_eval.py` implementa dos fases secuenciales.

### 6.1 Fase 1 — RAG Inference

Para cada una de las 200 preguntas:

1. **Retrieval:** Se consulta ChromaDB con la pregunta en texto libre. bge-m3 genera el embedding de la consulta y se recuperan los **top-5 chunks** más similares (similitud coseno).

2. **Generación:** Los 5 chunks se formatean como contexto y se envían a Llama 3.1-8B-Instruct vía la API OpenAI-compatible de vLLM:

```python
SYSTEM_PROMPT = (
    "Eres un asistente jurídico especializado en normativa española y europea. "
    "Responde a la pregunta basándote ÚNICAMENTE en los fragmentos de normativa "
    "proporcionados. Sé preciso y cita el texto relevante cuando sea posible. "
    "Si la información necesaria no aparece en los fragmentos, indícalo explícitamente."
)
```

Parámetros de generación: `temperature=0.0`, `max_tokens=512`.

3. **Guardado intermedio:** Las respuestas se guardan en `eval/rag_answers.json` para poder reutilizarlas sin repetir la inferencia (flag `--skip-inference`).

### 6.2 Fase 2 — Evaluación RAGAS

Con las respuestas generadas, RAGAS evalúa 4 métricas usando **el mismo Llama 3.1-8B como juez** (single-model evaluation):

| Métrica RAGAS | Qué mide | Entradas necesarias |
|---|---|---|
| **Faithfulness** | ¿La respuesta se basa solo en el contexto recuperado? | respuesta + contextos |
| **ResponseRelevancy** | ¿La respuesta es relevante para la pregunta? | pregunta + respuesta + embeddings |
| **LLMContextPrecisionWithReference** | ¿Los chunks recuperados son relevantes para la referencia? | pregunta + contextos + ground_truth |
| **LLMContextRecall** | ¿El contexto contiene la información necesaria para la respuesta correcta? | contextos + ground_truth |

El `LLMContextPrecisionWithReference` y el `LLMContextRecall` usan el `ground_truth` del dataset como referencia, por eso requieren el campo `reference` en los `SingleTurnSample`.

Para `ResponseRelevancy`, RAGAS genera preguntas sintéticas a partir de la respuesta y mide similitud con la pregunta original usando los **embeddings de bge-m3 en CPU**.

### 6.3 Arquitectura del job SLURM de evaluación

```
ragas_slurm.slurm
│
├── nvidia-smi                          # verificar GPU disponible
├── source .venv312/bin/activate
├── export HF_HOME, HF_HUB_OFFLINE=1
├── export VLLM_USE_V1=0               # motor V0 por compatibilidad ABI
├── export SENTENCE_TRANSFORMERS_DEVICE=cpu
│
├── python -m vllm.entrypoints.openai.api_server \
│       --model meta-llama/Llama-3.1-8B-Instruct \
│       --port 8000 --dtype bfloat16 \
│       --max-model-len 8192 \
│       --gpu-memory-utilization 0.85  &   ← background
│
├── [espera hasta que localhost:8000 responde, max 300s]
│   → tardó 130s en cargar el modelo
│
├── python eval/ragas_eval.py \
│       --vllm-url http://localhost:8000 \
│       --model meta-llama/Llama-3.1-8B-Instruct \
│       --top-k 5 --wait-vllm 0         ← 51 minutos en total
│
└── kill vLLM → guardar ragas_results.json → resumen
```

**Recursos utilizados:** 1× A100-SXM4-40GB, 48GB RAM, 8 CPUs. Throughput de vLLM: ~270-325 tokens/s.

### 6.4 Iteraciones del job RAGAS

| Job | Resultado | Problema |
|---|---|---|
| `ragas 53436` | Fallo (300s) | vLLM 0.21.0 con torch+cu13 → crash silencioso (log vacío) |
| `ragas 53449` | Fallo | vLLM 0.8.5.post1 + transformers 5.x → `AttributeError: all_special_tokens_extended` |
| `ragas 53453` | Fallo | vLLM V1 engine → `undefined symbol` en librtorch_c_dlpack |
| `ragas 53457` | **OK** ✓ | `VLLM_USE_V1=0` → 51 min de evaluación exitosa |

---

## 7. Ficheros generados

```
eval/
├── rag_answers.json        ← respuestas RAG por pregunta (Fase 1)
├── ragas_results.json      ← scores RAGAS por item + agregado (Fase 2)
├── ragas_analysis.md       ← análisis detallado de resultados
└── pipeline_process.md     ← este documento

logs/
├── xai_rag_setup_venv_*.out/err
├── xai_rag_embed_*.out/err
├── xai_rag_setup_ragas_*.out/err
├── xai_rag_download_llama_*.out/err
├── xai_rag_ragas_*.out/err
└── vllm_*.out              ← log del servidor vLLM

models/
└── huggingface/
    └── hub/
        ├── models--BAAI--bge-m3/            ← embeddings (caché)
        └── models--meta-llama--Llama-3.1-8B-Instruct/  ← 15GB

chroma_boe/                 ← base de datos vectorial persistente
ingest_report.json          ← informe de la ingesta
```

---

## 8. Scripts SLURM del proyecto

| Script | GPU | Tiempo | Propósito |
|---|---|---|---|
| `scripts/setup_venv.slurm` | 0 | 30 min | Crea `.venv312`, instala torch 2.6.0+cu124 y dependencias base |
| `scripts/embed_slurm.slurm` | 1 | 4h | Descarga documentos BOE/DOUE, chunkea e indexa en ChromaDB |
| `scripts/download_llama_slurm.slurm` | 0 | 30 min | Descarga Llama 3.1-8B-Instruct a caché local (requiere HF_TOKEN) |
| `scripts/setup_ragas.slurm` | 0 | 30 min | Instala vLLM 0.8.5.post1, transformers 4.x, langchain |
| `scripts/ragas_slurm.slurm` | 1 | 6h | Lanza vLLM, ejecuta RAG inference + evaluación RAGAS |

---

## 9. Resumen de decisiones técnicas clave

| Decisión | Alternativa descartada | Motivo |
|---|---|---|
| `torch==2.6.0+cu124` fijo | `pip install torch` sin pin | Sin pin instala cu130, incompatible con driver 570.x |
| `vllm==0.8.5.post1` | vllm 0.21.0 (latest) | 0.21.0 requiere torch 2.11.0+cu13, incompatible |
| `transformers<5.0.0` | transformers 5.8.1 | v5.x eliminó `all_special_tokens_extended` usado por vLLM |
| `VLLM_USE_V1=0` | Motor V1 por defecto | V1 falla por ABI mismatch en `torch_c_dlpack_addon` |
| bge-m3 en CPU durante RAGAS | bge-m3 en GPU | Dejar GPU íntegra a vLLM para maximizar throughput de generación |
| Resume via ChromaDB | Resume via `ingest_report.json` | El report del dry-run tenía todos los docs como "ok", causó skip total |
| `HF_HOME` local al proyecto | `~/.cache/huggingface` | Los modelos se descargan una vez y quedan en NFS, disponibles en todos los nodos sin re-descarga |
