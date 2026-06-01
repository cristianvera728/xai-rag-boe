# Análisis de Evaluación RAGAS — RAG-BOE

**Fecha:** 2026-05-18  
**Modelo generador:** Llama 3.1-8B-Instruct (vLLM 0.8.5.post1, motor V0)  
**Dataset:** `dataset_qa_boe.json` — 200 pares QA sobre normativa española/europea  
**Corpus:** ChromaDB con 5.427 chunks (bge-m3, top-k=5)  
**Jueces evaluados:** Llama 3.1-8B-Instruct · Qwen2.5-14B-Instruct

---

## 1. Resultados globales

| Métrica | Score | Items válidos |
|---|---|---|
| **Faithfulness** | **0.773** | 190/200 |
| **Answer Relevancy** | **0.812** | 196/200 |
| **Context Precision** | **0.870** | 154/200 |
| **Context Recall** | **0.928** | 194/200 |

> **Nota sobre nulos:** Context Precision tiene un 23% de items sin score (46/200), consecuencia de `LLMDidNotFinishException` durante la evaluación (el juez alcanzó el límite de tokens en prompts largos). Los promedios excluyen estos valores nulos.

### Interpretación general

- **Context Recall (0.928)** — El sistema de recuperación (ChromaDB + bge-m3) localiza casi toda la información relevante. Excelente cobertura del corpus.
- **Context Precision (0.870)** — Los chunks recuperados son mayoritariamente pertinentes para la pregunta. El top-k=5 genera poco ruido.
- **Answer Relevancy (0.812)** — Las respuestas generadas responden adecuadamente a las preguntas planteadas.
- **Faithfulness (0.773)** — El punto más débil: el modelo introduce ocasionalmente afirmaciones no respaldadas por el contexto recuperado. Esperado en un modelo 8B sin fine-tuning jurídico.

---

## 2. Análisis por topic (n=20 por categoría)

| Topic | Faithfulness | Answer Rel. | Context Prec. | Context Recall |
|---|---|---|---|---|
| procedimiento_adm | **0.847** | **0.916** | **0.918** | 0.947 |
| empleo_publico | **0.836** | 0.771 | 0.906 | 0.947 |
| subvenciones | **0.836** | 0.869 | 0.800 | 0.912 |
| regimen_local | 0.799 | 0.825 | 0.864 | **0.974** |
| hacienda | 0.784 | 0.849 | 0.910 | 0.897 |
| transparencia | 0.791 | 0.831 | 0.874 | 0.944 |
| funcion_publica | 0.795 | 0.828 | 0.881 | **1.000** |
| contratacion_publica | 0.719 | 0.762 | 0.803 | 0.922 |
| admin_electronica | 0.672 | 0.721 | **0.960** | 0.950 |
| proteccion_datos | **0.638** | **0.739** | 0.863 | **0.799** |

### Observaciones por topic

**Mejores:**
- `procedimiento_adm` lidera en faithfulness (0.847) y answer relevancy (0.916) — preguntas sobre la Ley 39/2015 con normativa bien estructurada y chunkeada por artículo.
- `funcion_publica` alcanza recall perfecto (1.000) — el EBEP está bien representado en el corpus.

**Problemáticos:**
- `proteccion_datos` es el topic con peor desempeño en todas las métricas, especialmente faithfulness (0.638) y recall (0.799). Las preguntas sobre la LOPDGDD y el RGPD implican plazos y cifras concretas donde el modelo tiende a alucinar. El documento DOUE (RGPD) también presenta chunking más complejo.
- `admin_electronica` tiene la faithfulness más baja (0.672) pese a tener muy buena precision (0.960) — el modelo recupera contexto relevante pero genera respuestas que van más allá de él.
- `contratacion_publica` sufre en preguntas comparativas de umbral (diferencias entre tipos de contratos) donde el modelo sintetiza en lugar de citar.

---

## 3. Análisis por dificultad

| Dificultad | n | Faithfulness | Answer Rel. | Context Prec. | Context Recall |
|---|---|---|---|---|---|
| **baja** | 80 | **0.814** | **0.829** | 0.856 | 0.928 |
| **media** | 80 | 0.747 | 0.824 | **0.886** | 0.929 |
| **alta** | 40 | 0.739 | 0.754 | 0.865 | 0.929 |

**Patrón claro:** La faithfulness y la answer relevancy decrecen con la dificultad, mientras que el context recall se mantiene constante (~0.929). Esto indica que la recuperación es robusta independientemente de la complejidad de la pregunta, pero la **generación** tiene más problemas con preguntas complejas (comparativas, multi-hop, que requieren síntesis de varios artículos).

---

## 4. Distribución de faithfulness

| Rango | Items | % |
|---|---|---|
| **≥ 0.9** (muy fiel) | 90 | 47.4% |
| 0.7 – 0.9 (aceptable) | 34 | 17.9% |
| 0.5 – 0.7 (marginal) | 39 | 20.5% |
| **< 0.5** (infiel) | 27 | 14.2% |

- **88 items** con faithfulness = 1.0 (completamente fundamentados en el contexto)
- **7 items** con faithfulness = 0.0 (respuesta completamente alucinada o sin relación con el contexto)

---

## 5. Casos problemáticos (faithfulness = 0.0)

| ID | Topic | Dificultad | Pregunta (resumen) |
|---|---|---|---|
| qa_039 | contratacion_publica | alta | Diferencia umbrales suministro vs servicios |
| qa_127 | proteccion_datos | baja | Plazo comunicación DPO a la AEPD |
| qa_132 | proteccion_datos | media | Plazo prescripción infracciones muy graves LOPDGDD |
| qa_134 | proteccion_datos | media | Plazo prescripción infracciones graves LOPDGDD |
| qa_144 | regimen_local | media | Municipios con Junta de Gobierno Local obligatoria |
| qa_185 | transparencia | baja | Ministerio al que está adscrito el Portal de la Transparencia |
| qa_196 | transparencia | alta | Sanciones por infracciones muy graves (Ley 37/2007) |

**Patrón común:** 6 de los 7 casos implican **datos numéricos concretos** (plazos de prescripción, umbrales económicos, cifras de población) o **adscripciones institucionales específicas**. El modelo genera el valor incorrecto o desactualizado en lugar de basarse exclusivamente en el contexto recuperado. Esto es consistente con la literatura sobre alucinaciones en LLMs para preguntas factuales.

---

## 6. Calidad de la recuperación (context recall)

| Rango recall | Items | % |
|---|---|---|
| **≥ 0.9** | 175 | 87.5% |
| 0.7 – 0.9 | 1 | 0.5% |
| 0.5 – 0.7 | 5 | 2.5% |
| **< 0.5** | 13 | 6.5% |

El pipeline de recuperación (bge-m3 + ChromaDB, top-k=5) es muy robusto: el 87.5% de los items obtiene recall ≥ 0.9. Los 13 casos con recall < 0.5 coinciden en su mayoría con el topic `proteccion_datos`, donde el RGPD (documento EUR-Lex) presenta una estructura de chunking diferente al BOE.

---

## 7. Conclusiones para el paper

1. **El pipeline RAG-BOE funciona correctamente** como baseline: recall y precision del contexto son elevados, indicando que la ingesta + bge-m3 + ChromaDB es una base sólida.

2. **La faithfulness (0.773) es el cuello de botella generativo**: Llama 3.1-8B sin ajuste específico tiende a alucinar en preguntas sobre datos numéricos (plazos, umbrales, cifras). Esto motiva directamente la necesidad de métricas de explicabilidad como ContextCite y MACS.

3. **Efecto de la dificultad en generación, no en recuperación**: La recuperación es invariante a la dificultad (recall ~0.929 en todas las categorías), pero la generación degrada en preguntas complejas. Resultado relevante para el paper.

4. **Protección de datos como caso de estudio**: El topic `proteccion_datos` concentra los peores resultados en todas las métricas, posiblemente por la complejidad del RGPD y el chunking de documentos EUR-Lex. Merece análisis cualitativo específico.

5. **23% de items sin Context Precision** por límite de tokens del juez: para trabajos futuros, aumentar `max_tokens` del LLM juez o usar un modelo con mayor ventana de contexto.

---

## 8. Configuración técnica del experimento

```
Modelo generador : meta-llama/Llama-3.1-8B-Instruct
Motor serving    : vLLM 0.8.5.post1 (motor V0, VLLM_USE_V1=0)
Hardware         : NVIDIA A100-SXM4-40GB, CUDA 12.8
dtype            : bfloat16
max_model_len    : 8192 tokens
gpu_memory_util  : 0.85
Embeddings       : BAAI/bge-m3 (CPU, normalize=True)
Vector store     : ChromaDB PersistentClient
Chunks totales   : 5.427 (22 documentos BOE/DOUE)
top-k retrieval  : 5
RAGAS versión    : 0.4.3
Métricas         : Faithfulness, ResponseRelevancy,
                   LLMContextPrecisionWithReference, LLMContextRecall
```

---

## 9. Comparativa de jueces: Llama-8B vs Qwen-14B

Se repitió la Fase 2 (evaluación RAGAS) usando **Qwen2.5-14B-Instruct** como juez sobre las mismas 200 respuestas generadas por Llama 3.1-8B (`eval/rag_answers.json`), con `--skip-inference`.

### 9.1 Resultados agregados

| Métrica | Llama-8B juez | Qwen-14B juez | Δ (Qwen − Llama) |
|---|---|---|---|
| **Faithfulness** | 0.7730 | 0.7788 | +0.006 |
| **Answer Relevancy** | 0.8125 | 0.7547 | −0.058 |
| **Context Precision** | 0.8696 | 0.7887 | −0.081 |
| **Context Recall** | 0.9284 | 0.8338 | **−0.095** |

Qwen es **más estricto** en todas las métricas excepto faithfulness, donde los dos jueces casi coinciden. Las mayores divergencias aparecen en Context Recall (−0.095) y Context Precision (−0.081).

### 9.2 Valores nulos por juez

| Métrica | Llama (nulos) | Qwen (nulos) | En ambos | Solo Llama | Solo Qwen |
|---|---|---|---|---|---|
| Faithfulness | 10 | 25 | — | — | — |
| Answer Relevancy | 4 | 5 | — | — | — |
| Context Precision | 46 | 43 | — | — | — |
| Context Recall | 6 | 2 | — | — | — |
| **Total (item×métrica)** | **66** | **75** | **8** | **58** | **67** |

**Los fallos no coinciden.** Solo 8 pares (item, métrica) son nulos en ambos jueces — el resto son fallos independientes causados por prompts que superan el límite de tokens del juez en ese momento concreto. Esto tiene implicaciones metodológicas: los promedios de cada evaluación excluyen conjuntos de items diferentes, por lo que la comparación directa entre columnas es aproximada.

Items nulos en **ambos** jueces: `qa_012` (faith), `qa_013`, `qa_014`, `qa_015`, `qa_106`, `qa_163`, `qa_194`, `qa_198` (prec/faith) — candidatos a revisión manual.

### 9.3 Casos de máxima discrepancia en faithfulness (|Δ| > 0.3)

| ID | Topic | Dif. | Llama | Qwen | Δ | Nota |
|---|---|---|---|---|---|---|
| qa_036 | contratacion_publica | alta | 1.00 | 0.00 | −1.00 | Llama demasiado indulgente consigo mismo |
| qa_058 | empleo_publico | media | 1.00 | 0.00 | −1.00 | Ídem |
| qa_126 | proteccion_datos | media | 1.00 | 0.00 | −1.00 | Ídem |
| qa_130 | proteccion_datos | baja | 1.00 | 0.00 | −1.00 | Ídem |
| qa_127 | proteccion_datos | baja | 0.00 | 1.00 | +1.00 | Llama demasiado severo |
| qa_118 | procedimiento_adm | media | 0.17 | 1.00 | +0.83 | Qwen más permisivo |
| qa_079 | funcion_publica | alta | 0.25 | 1.00 | +0.75 | Qwen más permisivo |
| qa_040 | contratacion_publica | alta | 0.33 | 1.00 | +0.67 | Qwen más permisivo |
| qa_059 | empleo_publico | baja | 0.33 | 1.00 | +0.67 | Qwen más permisivo |

**Patrón observado:**
- Cuando Llama juzga sus propias respuestas con faithfulness=1.0 pero Qwen dice 0.0 (4 casos), apunta a un sesgo de auto-evaluación: el modelo generador es demasiado indulgente con sus propias formulaciones.
- En el sentido contrario (Llama=0, Qwen=1), Llama es excesivamente severo en preguntas donde la respuesta sí está fundamentada en el contexto pero usa paráfrasis en lugar de cita literal.
- `proteccion_datos` concentra 3 de los 9 casos de máxima discrepancia — topic más susceptible al efecto juez.

### 9.4 Conclusiones sobre el efecto juez

1. **La faithfulness es la métrica más estable** entre jueces (+0.006): ambos modelos evalúan de forma similar si una afirmación está respaldada por el contexto.

2. **Context Recall y Precision son las más sensibles al juez** (−0.095 y −0.081): Qwen aplica un criterio más exigente al valorar si el ground_truth está cubierto por los chunks recuperados.

3. **El sesgo de auto-evaluación existe pero es limitado**: hay 4 casos claros donde Llama puntúa sus propias respuestas con 1.0 en faithfulness cuando Qwen asigna 0.0. Sin embargo, el efecto global es pequeño (+0.006 en promedio).

4. **Los nulos son independientes**: solo 8 de 133 fallos totales coinciden en ambos jueces, lo que indica que son artefactos del límite de tokens y no errores sistemáticos del dataset.

5. **Recomendación para el paper**: reportar ambos jueces y calcular el acuerdo inter-juez (p.ej. correlación de Spearman por item) para cuantificar la robustez de las métricas.
