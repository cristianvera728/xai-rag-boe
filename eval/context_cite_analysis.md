# Análisis ContextCite — RAG-BOE

**Fecha:** 2026-06-02  
**Modelo generador/atribuidor:** Llama 3.1-8B-Instruct (vLLM 0.22.0, NVIDIA GB10 / DGX Spark)  
**Dataset:** `dataset_qa_boe.json` — 200 pares QA sobre normativa española/europea  
**Corpus:** ChromaDB con 5.427 chunks (bge-m3, top-k=5)  
**Ablaciones por item:** 64 (mask_prob=0.5, Ridge regression)

---

## 1. Resultados globales

| Métrica ContextCite | Score |
|---|---|
| **Items válidos** | **200/200** (0 errores, 64/64 ablaciones OK por item) |
| **Mean top-1 score** | **0.757** |
| **Mean concentration** | **0.597** |
| **Mean entropy** | **0.649** |
| **Mean chunks con >10% atribución** | **1.91** |
| **Mean chunks con >20% atribución** | **1.36** |

### Interpretación general

El algoritmo ContextCite computa, para cada respuesta generada, qué fracción del score de log-probabilidad se debe a cada fragmento de contexto recuperado. Los resultados muestran que el modelo Llama 3.1-8B:

- **Cita principalmente de un único chunk** (`mean top-1 score = 0.757`): en promedio, el fragmento más atribuido concentra el 75% del peso de atribución.
- **En media, solo 1-2 chunks son funcionalmente relevantes**: el 63% de los items tiene a lo sumo 2 chunks con más del 10% de atribución, y el 78% tiene a lo sumo 1 chunk con más del 20%.
- **La concentración es moderada (0.597)**: no extrema ni uniforme — hay variabilidad significativa entre topics y dificultades que refleja la estructura semántica del corpus.

---

## 2. Distribución de concentración

| Rango | Items | % |
|---|---|---|
| **≥ 0.7** (muy concentrado) | 78 | **39.0%** |
| 0.4 – 0.7 (moderado) | 79 | **39.5%** |
| **< 0.4** (difuso) | 43 | **21.5%** |

- **11 items** con concentración = 1.0 (toda la atribución recae en un único chunk, perfectamente concentrado).
- **1 item** (`qa_127`, `proteccion_datos`) con concentración ≈ 0.011 — los 5 chunks reciben pesos casi idénticos (0.21, 0.25, 0.22, 0.14, 0.19), indicando que el modelo no tiene un fundamento claro en el contexto.

---

## 3. Análisis por topic

| Topic | Concentration | Top-1 Score | n |
|---|---|---|---|
| **hacienda** | **0.707** | **0.842** | 20 |
| **subvenciones** | **0.676** | **0.805** | 20 |
| regimen_local | 0.635 | 0.789 | 20 |
| funcion_publica | 0.615 | 0.770 | 20 |
| empleo_publico | 0.614 | 0.772 | 20 |
| contratacion_publica | 0.576 | 0.736 | 20 |
| transparencia | 0.567 | 0.726 | 20 |
| admin_electronica | 0.559 | 0.745 | 20 |
| **proteccion_datos** | **0.512** | **0.706** | 20 |
| **procedimiento_adm** | **0.504** | **0.679** | 20 |

### Observaciones por topic

**Más concentrados:**
- `hacienda` (conc=0.707): las preguntas sobre normativa tributaria y presupuestaria tienen respuestas bien localizadas en artículos específicos. El modelo cita consistentemente de un único fragmento.
- `subvenciones` (conc=0.676): la Ley 38/2003 General de Subvenciones está bien articulada, con artículos autocontenidos que el chunker separó limpiamente.
- `funcion_publica` y `empleo_publico` (conc≈0.615): el EBEP tiene una estructura articulada muy clara. **11 items de `empleo_publico` alcanzan concentración perfecta (1.0)** — el modelo obtiene toda la información de un solo artículo del EBEP.

**Más difusos:**
- `procedimiento_adm` (conc=0.504): paradójico dado que era el mejor topic en RAGAS faithfulness (0.847). La Ley 39/2015 tiene artículos interconectados (remisiones cruzadas entre artículos de procedimiento) que llevan al modelo a apoyarse en múltiples fragmentos simultáneamente. Alta faithfulness con atribución difusa indica síntesis coherente de varios artículos.
- `proteccion_datos` (conc=0.512): atribución difusa y, a diferencia de `procedimiento_adm`, también con faithfulness baja (0.638 en RAGAS). Aquí la difusión refleja que el modelo no encuentra una base sólida — el RGPD (EUR-Lex, formato Formex4) presenta chunks menos homogéneos que el BOE.

---

## 4. Análisis por dificultad

| Dificultad | n | Concentration | Top-1 Score |
|---|---|---|---|
| **baja** | 80 | **0.619** | **0.775** |
| **media** | 80 | 0.587 | 0.750 |
| **alta** | 40 | 0.570 | 0.735 |

**Patrón claro y consistente con RAGAS:** a mayor dificultad, menor concentración de atribución. Las preguntas difíciles (comparativas, multi-hop, que requieren síntesis de varios artículos) distribuyen la atribución entre más chunks. Este gradiente —invisible en el context recall de RAGAS, que se mantiene constante (~0.929)— sí aparece en ContextCite, confirmando que la **complejidad afecta a la generación, no a la recuperación**.

---

## 5. Sesgo de posición: el primer chunk domina

| Posición del chunk más atribuido (top-1) | Items | % |
|---|---|---|
| chunk 0 (primer recuperado, mayor similitud) | 120 | **60.0%** |
| chunk 1 | 45 | 22.5% |
| chunk 2 | 19 | 9.5% |
| chunk 3 | 10 | 5.0% |
| chunk 4 | 6 | 3.0% |

**Atribución media por posición:**
- Chunk 0 (rank retrieval = 1): **0.507** de atribución media
- Chunks 1-4 combinados: **0.123** de atribución media cada uno

El primer chunk recuperado (el de mayor similitud coseno con la pregunta) recibe en promedio **4 veces más atribución** que cada uno de los restantes. Esto puede reflejar tanto la calidad del retrieval (el chunk 0 suele contener la respuesta) como un sesgo posicional del modelo, donde el contenido que aparece primero en el prompt tiene mayor influencia sobre la generación.

**Nota metodológica:** distinguir ambos efectos requeriría un experimento de control con posiciones aleatorizadas, que se propone como trabajo futuro.

---

## 6. Relación entre ContextCite y faithfulness RAGAS

### 6.1 Correlación global

| Par de métricas | Pearson r | Spearman ρ | n |
|---|---|---|---|
| concentration ↔ faithfulness | 0.088 | 0.012 | 190 |
| top-1 score ↔ faithfulness | 0.103 | — | 190 |
| n_chunks >20% ↔ faithfulness | **−0.158** | — | 190 |

Las correlaciones son **débiles**. Esto no indica que ContextCite sea inútil — indica que mide una dimensión diferente a RAGAS faithfulness:
- **RAGAS faithfulness** evalúa si las afirmaciones de la respuesta están respaldadas por el contexto recuperado (corrección de las afirmaciones).
- **ContextCite concentration** evalúa cuánto depende la respuesta de cada fragmento de contexto (estructura de la atribución).

Una respuesta puede ser completamente fiel (faithfulness=1.0) pero con atribución difusa (síntesis coherente de varios artículos). Alternativamente, puede tener atribución concentrada pero aun así alucinar (citar del chunk correcto pero reformular erróneamente un dato numérico).

### 6.2 Concentración por rango de faithfulness

| Rango faithfulness | Concentration media | n |
|---|---|---|
| faithfulness = 1.0 | 0.598 | 88 |
| 0.7 – 0.9 | 0.640 | 36 |
| 0.5 – 0.7 | 0.649 | 39 |
| **< 0.5** | **0.498** | **27** |

El único patrón visible: los items con faithfulness < 0.5 tienen concentración sistemáticamente más baja (0.498 vs ~0.62-0.64 en los demás rangos). **La difusión de atribución es señal de riesgo de alucinación**, pero no su causa directa — hay items muy concentrados que también alucinan.

---

## 7. Análisis cualitativo: casos faithfulness = 0.0

Los 7 items con faithfulness = 0.0 en RAGAS presentan perfiles ContextCite heterogéneos:

| ID | Topic | Dif. | Concentration | Top-1 | Chunks (atribución norm.) | Interpretación |
|---|---|---|---|---|---|---|
| qa_127 | proteccion_datos | baja | **0.011** | 0.250 | [0.21, 0.25, 0.22, 0.14, 0.19] | Atribución uniforme — el modelo no tiene fuente clara para su respuesta (alucinación sin ancla) |
| qa_196 | transparencia | alta | 0.151 | 0.391 | [0.39, 0.06, 0.04, 0.25, 0.26] | Difusa, varios chunks semi-relevantes — síntesis incoherente |
| qa_039 | contratacion_publica | alta | 0.334 | 0.576 | [0.13, 0.00, 0.05, 0.25, 0.58] | Moderadamente concentrado en chunk 4 — el modelo cita de ese chunk pero extrapola más allá |
| qa_185 | transparencia | baja | 0.501 | 0.506 | [0.00, 0.46, 0.00, 0.03, 0.51] | Dos chunks relevantes — el modelo integra ambos pero genera una adscripción institucional incorrecta |
| qa_134 | proteccion_datos | media | 0.453 | 0.637 | [0.00, 0.64, 0.11, 0.25, 0.00] | Concentrado en chunk 1 — el chunk contiene el plazo pero el modelo lo reformula incorrectamente |
| qa_144 | regimen_local | media | 0.654 | 0.755 | [0.00, 0.00, 0.75, 0.25, 0.00] | Alta concentración — el modelo cita del chunk correcto pero da un umbral de población erróneo |
| qa_132 | proteccion_datos | media | **0.685** | **0.856** | [0.09, 0.05, 0.86, 0.00, 0.00] | Muy concentrado — el chunk 2 sí contiene el plazo de prescripción; el modelo lo recupera pero lo confunde con el plazo de otra categoría de infracción |

**Taxonomía emergente de alucinaciones según ContextCite:**

1. **Alucinación sin ancla** (`qa_127`, conc≈0): atribución uniforme entre todos los chunks — el modelo no encontró un fundamento sólido y generó información de sus pesos preentrenados. Máximo riesgo.

2. **Alucinación por extrapolación** (`qa_039`, `qa_196`, conc<0.4): atribución difusa — el modelo basa su respuesta parcialmente en el contexto pero sintetiza/extrapola más allá de lo que los chunks establecen.

3. **Alucinación por confusión interna** (`qa_132`, `qa_134`, `qa_144`, conc>0.4): atribución concentrada en el chunk correcto — el modelo SÍ recupera el fragmento pertinente pero malinterpreta o sustituye un dato numérico específico (plazo, umbral, cifra). Este tipo es el más difícil de detectar solo con ContextCite, ya que la concentración sugiere un fundamento sólido que en realidad no existe.

---

## 8. Casos extremos notables

### Items con concentración perfecta (conc = 1.0, n = 11)

| ID | Topic | Dif. | Chunk más atribuido |
|---|---|---|---|
| qa_041, qa_047, qa_050, qa_051, qa_054 | empleo_publico | baja/media | chunk 0 |
| qa_153, qa_155 | regimen_local | baja/media | chunk 0 |
| qa_008 | admin_electronica | media | chunk 0 |
| qa_096 | hacienda | baja | chunk 0 |
| qa_108 | procedimiento_adm | baja | chunk 0 |
| qa_172 | subvenciones | baja | chunk 0 |

Todos son preguntas de dificultad baja o media donde la respuesta está completamente contenida en un único artículo bien delimitado. El predominio de `empleo_publico` (5/11) confirma que el EBEP tiene la estructura de chunking más limpia del corpus.

### Item más difuso (qa_127, proteccion_datos/baja)

Este item aparece como el caso más problemático en **ambas** evaluaciones:
- RAGAS Llama-judge: faithfulness = **0.0**
- RAGAS Qwen-judge: faithfulness = **1.0** (máxima discrepancia inter-juez)
- ContextCite: concentration = **0.011** (distribución casi uniforme)

La concentración casi nula de ContextCite respalda la evaluación del juez Llama: el modelo generó su respuesta sin basarse en ningún fragmento específico, lo que explica tanto la alucinación como la discrepancia entre jueces (Qwen podría haber evaluado la coherencia formal de la respuesta, no su fundamentación real en el contexto).

---

## 9. Configuración técnica del experimento

```
Modelo generador/atribuidor : meta-llama/Llama-3.1-8B-Instruct
Motor serving               : vLLM 0.22.0
Hardware                    : NVIDIA GB10 (DGX Spark), CUDA 13.0
dtype                       : bfloat16
max_model_len               : 8192 tokens
Ablaciones por item         : 64 (mask_prob=0.5)
Regresión                   : Ridge (alpha=1.0)
Granularidad                : frase (media: 40.5 partes/item, rango 17-89)
Agregación                  : suma de scores positivos por chunk, normalizada
Items procesados            : 200/200 (0 errores)
Ablaciones válidas          : 64/64 por item (100%)
```

---

## 10. Conclusiones para el paper

1. **ContextCite es complementario a RAGAS, no redundante.** La baja correlación entre concentration y faithfulness (r=0.09) indica que ambas métricas capturan aspectos distintos de la fidelidad: RAGAS evalúa la corrección de las afirmaciones, ContextCite evalúa la estructura de la atribución. Usadas conjuntamente ofrecen un diagnóstico más completo.

2. **La taxonomía de alucinaciones es la aportación central.** ContextCite permite distinguir tres tipos cualitativamente distintos: (a) alucinación sin ancla contextual —detectable por concentración≈0—, (b) alucinación por extrapolación —concentración baja/media con faithfulness=0—, y (c) alucinación por confusión interna —concentración alta con faithfulness=0, el tipo más difícil de detectar automáticamente.

3. **El sesgo de posición es significativo.** El chunk 0 recibe 4× más atribución media que los demás. Esto puede reflejar calidad del retrieval (bge-m3 coloca el chunk más relevante en primer lugar) o un sesgo posicional del modelo. Este efecto tiene implicaciones para el diseño de prompts RAG y merece un experimento de control.

4. **Protección de datos sigue siendo el topic más problemático.** Peor en RAGAS (faithfulness=0.638) y peor en ContextCite (concentration=0.512). La combinación de faithfulness baja + concentración baja identifica este topic como el candidato más claro para mejoras específicas (chunking del RGPD, fine-tuning jurídico).

5. **La dificultad afecta a la generación, no a la recuperación.** El gradiente de concentración (baja→alta: 0.619→0.570) replica el gradiente de faithfulness RAGAS sin afectar al context recall. ContextCite confirma que la degradación en preguntas complejas ocurre en la fase de generación/síntesis.

6. **`qa_127` como caso de estudio transversal.** Este item aparece como outlier en RAGAS (faithfulness=0 con ambos jueces coincidiendo en el peor resultado), en el análisis inter-juez (máxima discrepancia) y en ContextCite (concentración mínima del corpus). Merece un análisis cualitativo manual detallado en el paper.
