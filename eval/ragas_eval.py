"""
eval/ragas_eval.py
------------------
Evaluación RAGAS del sistema RAG-BOE.

Fase 1 — RAG inference:
  Para cada pregunta del dataset, recupera top-K chunks de ChromaDB
  y genera una respuesta con Llama 3.1-8B-Instruct vía vLLM.
  Resultado: eval/rag_answers.json (se puede reusar con --skip-inference)

Fase 2 — RAGAS evaluation:
  Evalúa con 4 métricas usando el mismo LLM como juez:
    · faithfulness          → ¿la respuesta se atiene al contexto?
    · response_relevancy    → ¿la respuesta responde la pregunta?
    · context_precision     → ¿los chunks son relevantes para la pregunta?
    · context_recall        → ¿el contexto cubre la respuesta correcta?

Uso:
  python eval/ragas_eval.py \\
      --vllm-url http://localhost:8000 \\
      --dataset dataset_qa_boe.json \\
      --out eval/ragas_results.json
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# Retrieval usa bge-m3 en CPU para dejar la GPU libre a vLLM
os.environ.setdefault("SENTENCE_TRANSFORMERS_DEVICE", "cpu")

# ── Importaciones lazy (se cargan solo cuando se necesitan) ──

def _import_ragas():
    try:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.metrics import (
            Faithfulness,
            ResponseRelevancy,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
        )
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        return {
            "EvaluationDataset": EvaluationDataset,
            "SingleTurnSample": SingleTurnSample,
            "evaluate": evaluate,
            "Faithfulness": Faithfulness,
            "ResponseRelevancy": ResponseRelevancy,
            "LLMContextPrecisionWithReference": LLMContextPrecisionWithReference,
            "LLMContextRecall": LLMContextRecall,
            "LangchainLLMWrapper": LangchainLLMWrapper,
            "LangchainEmbeddingsWrapper": LangchainEmbeddingsWrapper,
        }
    except ImportError as e:
        log.error(f"Error importando RAGAS: {e}")
        log.error("Instala: pip install ragas langchain-openai langchain-community")
        sys.exit(1)


# ── Prompts ──────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "Eres un asistente jurídico especializado en normativa española y europea. "
    "Responde a la pregunta basándote ÚNICAMENTE en los fragmentos de normativa "
    "proporcionados. Sé preciso y cita el texto relevante cuando sea posible. "
    "Si la información necesaria no aparece en los fragmentos, indícalo explícitamente."
)

def build_user_prompt(question: str, contexts: list[str]) -> str:
    frags = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    return f"Fragmentos de normativa:\n{frags}\n\nPregunta: {question}"


# ── vLLM helpers ─────────────────────────────────────────────

def wait_for_vllm(base_url: str, timeout: int = 300) -> None:
    log.info(f"Esperando vLLM en {base_url} (timeout {timeout}s)...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/v1/models", timeout=5)
            if r.status_code == 200:
                models = r.json().get("data", [])
                log.info(f"vLLM listo. Modelos: {[m['id'] for m in models]}")
                return
        except requests.RequestException:
            pass
        time.sleep(5)
    log.error("vLLM no respondió en el tiempo límite")
    sys.exit(1)


def generate_answer(client, model: str, question: str, contexts: list[str]) -> str:
    from openai import OpenAI  # importación local para no fallar si no está al inicio

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(question, contexts)},
        ],
        temperature=0.0,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


# ── Fase 1: RAG inference ─────────────────────────────────────

def run_inference(args) -> list[dict]:
    answers_path = Path(args.answers)

    if answers_path.exists() and not args.force_inference:
        log.info(f"Cargando respuestas previas de {answers_path} (usa --force-inference para regenerar)")
        return json.loads(answers_path.read_text(encoding="utf-8"))

    # Cargar dataset
    qa_items: list[dict] = json.loads(
        Path(args.dataset).read_text(encoding="utf-8")
    )
    log.info(f"Dataset: {len(qa_items)} pares QA")

    # ChromaDB
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from boe_pipeline import get_chroma_collection
    collection = get_chroma_collection(args.chroma, args.collection)
    log.info(f"ChromaDB: {collection.count()} chunks")

    # Cliente vLLM
    from openai import OpenAI
    client = OpenAI(base_url=f"{args.vllm_url}/v1", api_key="token-local")

    results = []
    for i, item in enumerate(qa_items, 1):
        log.info(f"[{i}/{len(qa_items)}] {item['id']} — {item['question'][:70]}...")

        # Retrieval
        r = collection.query(
            query_texts=[item["question"]],
            n_results=args.top_k,
            include=["documents", "metadatas"],
        )
        contexts   = r["documents"][0]
        metadatas  = r["metadatas"][0]

        # Generación
        answer = generate_answer(client, args.model, item["question"], contexts)

        results.append({
            "id":           item["id"],
            "question":     item["question"],
            "answer":       answer,
            "contexts":     contexts,
            "context_meta": [
                {"doc_id": m["doc_id"], "art_num": m.get("art_num", ""), "url": m["url"]}
                for m in metadatas
            ],
            "ground_truth": item["ground_truth"],
            "doc_id":       item["doc_id"],
            "topic":        item.get("topic", ""),
            "difficulty":   item.get("difficulty", ""),
        })

    answers_path.parent.mkdir(parents=True, exist_ok=True)
    answers_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info(f"Respuestas guardadas en {answers_path}")
    return results


# ── Fase 2: Evaluación RAGAS ──────────────────────────────────

def run_ragas(answers_data: list[dict], args) -> dict:
    r = _import_ragas()

    # LLM juez (mismo Llama via vLLM)
    from langchain_openai import ChatOpenAI
    from langchain_community.embeddings import HuggingFaceEmbeddings

    langchain_llm = ChatOpenAI(
        model=args.model,
        base_url=f"{args.vllm_url}/v1",
        api_key="token-local",
        temperature=0,
        max_tokens=1024,
    )
    ragas_llm = r["LangchainLLMWrapper"](langchain_llm)

    langchain_emb = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    ragas_emb = r["LangchainEmbeddingsWrapper"](langchain_emb)

    metrics = [
        r["Faithfulness"](llm=ragas_llm),
        r["ResponseRelevancy"](llm=ragas_llm, embeddings=ragas_emb),
        r["LLMContextPrecisionWithReference"](llm=ragas_llm),
        r["LLMContextRecall"](llm=ragas_llm),
    ]

    samples = [
        r["SingleTurnSample"](
            user_input=d["question"],
            response=d["answer"],
            retrieved_contexts=d["contexts"],
            reference=d["ground_truth"],
        )
        for d in answers_data
    ]

    log.info(f"Evaluando {len(samples)} muestras con RAGAS ({len(metrics)} métricas)...")
    dataset = r["EvaluationDataset"](samples=samples)
    result  = r["evaluate"](dataset, metrics=metrics)

    # Convertir resultado a dict serializable
    scores_df = result.to_pandas()
    per_item  = scores_df.to_dict(orient="records")

    # Añadir metadatos del dataset a cada fila
    meta_keys = ["id", "doc_id", "topic", "difficulty"]
    for row, d in zip(per_item, answers_data):
        for k in meta_keys:
            row[k] = d.get(k, "")

    aggregate = {
        col: float(scores_df[col].mean())
        for col in scores_df.columns
        if col not in ("user_input", "response", "retrieved_contexts", "reference")
        and scores_df[col].dtype in ("float64", "float32")
    }

    return {"aggregate": aggregate, "per_item": per_item}


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluación RAGAS del RAG-BOE")
    parser.add_argument("--dataset",        default="dataset_qa_boe.json")
    parser.add_argument("--chroma",         default="./chroma_boe")
    parser.add_argument("--collection",     default="boe_normativa")
    parser.add_argument("--vllm-url",       default="http://localhost:8000")
    parser.add_argument("--model",          default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--top-k",          type=int, default=5)
    parser.add_argument("--answers",        default="eval/rag_answers.json",
                        help="JSON intermedio con respuestas RAG (se reutiliza en rerun)")
    parser.add_argument("--out",            default="eval/ragas_results.json")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Salta la fase 1 y usa --answers existente")
    parser.add_argument("--force-inference", action="store_true",
                        help="Regenera respuestas aunque --answers ya exista")
    parser.add_argument("--wait-vllm",      type=int, default=300,
                        help="Segundos a esperar por vLLM (0 = no esperar)")
    args = parser.parse_args()

    # Verificar vLLM
    if args.wait_vllm > 0:
        wait_for_vllm(args.vllm_url, args.wait_vllm)

    # Fase 1
    if args.skip_inference:
        if not Path(args.answers).exists():
            log.error(f"--skip-inference pero {args.answers} no existe")
            sys.exit(1)
        answers_data = json.loads(Path(args.answers).read_text(encoding="utf-8"))
        log.info(f"Usando {len(answers_data)} respuestas previas de {args.answers}")
    else:
        answers_data = run_inference(args)

    # Fase 2
    ragas_output = run_ragas(answers_data, args)

    # Guardar resultados
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(ragas_output, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log.info("=" * 60)
    log.info("RESULTADOS RAGAS (media)")
    for metric, score in ragas_output["aggregate"].items():
        log.info(f"  {metric:<40} {score:.4f}")
    log.info(f"Resultados guardados en {out_path}")


if __name__ == "__main__":
    main()
