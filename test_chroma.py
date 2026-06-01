"""
test_chroma.py
--------------
Verifica que ChromaDB tiene datos correctos tras ejecutar boe_pipeline.py.

Uso:
  python test_chroma.py
  python test_chroma.py --chroma ./chroma_boe --query "excedencia funcionario"
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

EMBED_MODEL = "BAAI/bge-m3"
CHROMA_PATH = "./chroma_boe"
COLLECTION  = "boe_normativa"

REQUIRED_META = ["doc_id", "titulo", "fecha", "url", "estrategia", "chunk_idx"]
SAMPLE_QUERIES = [
    "¿Qué límite económico tiene el contrato menor?",
    "procedimiento administrativo sancionador",
    "transparencia en la administración pública",
]


def sep(title: str = "") -> None:
    line = "=" * 60
    print(f"\n{line}")
    if title:
        print(f"  {title}")
        print(line)


def check_collection(path: str, embed_fn) -> chromadb.Collection:
    p = Path(path)
    if not p.exists():
        print(f"ERROR: ChromaDB no encontrada en {p.resolve()}")
        sys.exit(1)

    client = chromadb.PersistentClient(path=str(p))
    try:
        col = client.get_collection(name=COLLECTION, embedding_function=embed_fn)
    except Exception as e:
        print(f"ERROR: No se pudo abrir '{COLLECTION}': {e}")
        sys.exit(1)
    return col


def test_counts(col: chromadb.Collection) -> int:
    sep("1. Totales")
    total = col.count()
    print(f"  Chunks totales : {total}")
    if total == 0:
        print("  WARN: La colección está vacía — ejecuta boe_pipeline.py primero.")
        sys.exit(1)
    return total


def test_strategy_distribution(metas: list[dict]) -> None:
    sep("2. Distribución por estrategia de chunking")
    estrategias = Counter(m.get("estrategia", "unknown") for m in metas)
    for k, v in estrategias.most_common():
        pct = 100 * v / len(metas)
        print(f"  {k:22s}: {v:5d}  ({pct:.1f} %)")


def test_unique_docs(metas: list[dict]) -> int:
    sep("3. Documentos únicos")
    doc_ids = Counter(m.get("doc_id", "") for m in metas)
    n_docs = len(doc_ids)
    avg_chunks = len(metas) / max(n_docs, 1)
    print(f"  Documentos únicos : {n_docs}")
    print(f"  Chunks por doc    : {avg_chunks:.1f} (media en muestra)")
    return n_docs


def test_metadata_completeness(metas: list[dict]) -> bool:
    sep("4. Completitud de metadatos")
    ok = True
    for field in REQUIRED_META:
        missing = sum(1 for m in metas if not m.get(field) and m.get(field) != 0)
        status = "OK" if missing == 0 else f"WARN — {missing} vacíos"
        print(f"  {field:20s}: {status}")
        if missing > 0:
            ok = False

    # Campos extra del chunking mejorado
    has_art_num = sum(1 for m in metas if m.get("art_num"))
    pct = 100 * has_art_num / len(metas)
    print(f"  {'art_num':20s}: {has_art_num} chunks con número de artículo ({pct:.0f} %)")
    return ok


def test_no_duplicates(col: chromadb.Collection) -> None:
    sep("5. Duplicados")
    all_ids = col.get(include=[])["ids"]
    dupes = len(all_ids) - len(set(all_ids))
    status = "OK — sin duplicados" if dupes == 0 else f"ERROR — {dupes} IDs repetidos"
    print(f"  {status}")


def test_chunk_length(col: chromadb.Collection, sample_n: int = 200) -> None:
    sep("6. Longitud de chunks (muestra)")
    sample = col.get(limit=sample_n, include=["documents"])
    lengths = [len(d.split()) for d in sample["documents"]]
    if not lengths:
        return
    print(f"  Min palabras  : {min(lengths)}")
    print(f"  Max palabras  : {max(lengths)}")
    print(f"  Media palabras: {sum(lengths)/len(lengths):.0f}")
    short = sum(1 for l in lengths if l < 20)
    if short:
        print(f"  WARN: {short} chunks con menos de 20 palabras")


def test_sample_chunks(col: chromadb.Collection, n: int = 3) -> None:
    sep("7. Muestra de chunks")
    sample = col.get(limit=n, include=["documents", "metadatas"])
    for i, (doc_text, meta) in enumerate(zip(sample["documents"], sample["metadatas"])):
        art_info = ""
        if meta.get("art_num"):
            art_info = f"  Art. {meta['art_num']}"
            if meta.get("art_titulo"):
                art_info += f" — {meta['art_titulo'][:50]}"
        print(f"\n  [{i+1}] {sample['ids'][i]}")
        print(f"       doc_id    : {meta.get('doc_id','?')}")
        print(f"       título    : {meta.get('titulo','?')[:65]}")
        print(f"       fecha     : {meta.get('fecha','?')}")
        print(f"       estrategia: {meta.get('estrategia','?')}{art_info}")
        print(f"       texto     : {doc_text[:220]}...")


def test_retrieval(col: chromadb.Collection, queries: list[str]) -> None:
    sep("8. Test de retrieval")
    total = col.count()
    for query in queries:
        results = col.query(
            query_texts=[query],
            n_results=min(3, total),
            include=["documents", "metadatas", "distances"],
        )
        print(f"\n  Query: {query}")
        for doc_text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = 1 - dist
            art = f" | Art.{meta['art_num']}" if meta.get("art_num") else ""
            print(f"    Score {score:.3f}{art} | {meta.get('titulo','')[:55]}")
            print(f"    {doc_text[:160]}...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verifica ChromaDB del pipeline BOE")
    parser.add_argument("--chroma", default=CHROMA_PATH, help="Directorio ChromaDB")
    parser.add_argument("--query",  default=None,
                        help="Query adicional para el test de retrieval")
    args = parser.parse_args()

    embed_fn = SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL, device="cpu", normalize_embeddings=True
    )

    sep(f"ChromaDB Test — {args.chroma} / {COLLECTION}")

    col   = check_collection(args.chroma, embed_fn)
    total = test_counts(col)

    # Muestra representativa para estadísticas (máx. 1000 chunks)
    raw    = col.get(limit=min(total, 1_000), include=["metadatas"])
    metas  = raw["metadatas"]

    test_strategy_distribution(metas)
    n_docs = test_unique_docs(metas)
    test_metadata_completeness(metas)
    test_no_duplicates(col)
    test_chunk_length(col)
    test_sample_chunks(col)

    queries = SAMPLE_QUERIES.copy()
    if args.query:
        queries.insert(0, args.query)
    test_retrieval(col, queries[:3])

    sep("Resultado final")
    print(f"  Total chunks  : {total}")
    print(f"  Documentos    : {n_docs}")
    print("  ChromaDB OK — listo para evaluación RAG.\n")


if __name__ == "__main__":
    main()
