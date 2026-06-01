"""
boe_pipeline.py
---------------
Pipeline completo para el corpus XAI-RAG-AAPP:
  1. Descarga documentos del BOE via API oficial (api.boe.es)
  2. Extrae texto y chunking por artículo
  3. Genera embeddings con bge-m3
  4. Almacena en ChromaDB persistente

Uso:
  pip install -r requirements.txt
  python boe_pipeline.py --query "contrato menor" --n 50

Referencia API BOE: https://www.boe.es/datosabiertos/documentos/APIguia.pdf
"""

import argparse
import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import chromadb
import requests
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ── Config ────────────────────────────────────────────────
BOE_API_BASE   = "https://www.boe.es/diario_boe/xml.php"
BOE_SEARCH_URL = "https://www.boe.es/diario_boe/xml.php"
SUMARIO_URL    = "https://www.boe.es/diario_boe/xml.php?id=BOE-S-{date}"
DOC_URL        = "https://www.boe.es/diario_boe/xml.php?id={doc_id}"

EMBED_MODEL    = "BAAI/bge-m3"          # multilingüe, legal-friendly
CHROMA_PATH    = "./chroma_boe"         # directorio persistente local
COLLECTION     = "boe_normativa"

CHUNK_SIZE     = 400    # palabras por chunk (proxy de tokens)
CHUNK_OVERLAP  = 80

# Artículo N. / ARTÍCULO 12bis: / Art. 3 -
_ART_RE = re.compile(
    r"(?:Art[íi]culo|ARTÍCULO|Art\.)\s+"
    r"(\d+[\w\-]*)"           # número: 1, 12, 3bis, 2-A
    r"[ \t]*[\.:\-–]?[ \t]*"  # separador opcional
    r"([^\n\.;]{0,120})",     # título del artículo (hasta 120 chars)
    re.IGNORECASE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── 1. Scraping BOE ───────────────────────────────────────

def search_boe(query: str, n_results: int = 50) -> list[dict]:
    """
    Busca en el buscador del BOE y devuelve lista de {id, titulo, fecha}.
    Usa la API de búsqueda de texto libre del BOE.
    """
    results = []
    page = 1
    headers = {"Accept": "application/json", "User-Agent": "XAI-RAG-Research/1.0"}

    while len(results) < n_results:
        params = {
            "c[0][campo]": "titulo",
            "c[0][dato]": query,
            "c[0][oper]": "AND",
            "page_hits": 20,
            "actual_page": page,
        }
        try:
            resp = requests.get(
                "https://boe.es/buscar/legislacion.php",
                params=params,
                headers=headers,
                timeout=15,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            log.warning(f"Error en búsqueda BOE página {page}: {e}")
            break

        # Extraer IDs de documentos del HTML devuelto
        doc_ids = re.findall(r'id=(BOE-[A-Z]-\d{4}-\d+)', resp.text)
        if not doc_ids:
            break

        for doc_id in doc_ids:
            results.append({"id": doc_id})
            if len(results) >= n_results:
                break

        page += 1
        time.sleep(0.5)  # respetar rate limit BOE

    log.info(f"Encontrados {len(results)} documentos para query='{query}'")
    return results[:n_results]


def fetch_boe_document(doc_id: str) -> dict | None:
    """
    Descarga el XML de un documento BOE y extrae texto + metadatos.
    Preserva separación de párrafos (necesaria para detectar artículos).
    Devuelve dict con {id, titulo, fecha, texto, url}.
    """
    url = f"https://www.boe.es/diario_boe/xml.php?id={doc_id}"
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as e:
        log.warning(f"No se pudo descargar {doc_id}: {e}")
        return None

    titulo, fecha, texto = "", "", ""
    try:
        root = ET.fromstring(resp.content)
        titulo = _et_text(root, "metadatos/titulo") or doc_id
        fecha  = _et_text(root, "metadatos/fecha_publicacion") or ""
        # "texto" es hijo directo de <documento>; ".//texto" encontraría
        # el primer <texto> de las referencias cruzadas en <analisis>
        texto_elem = root.find("texto")
        if texto_elem is not None:
            # Un párrafo por línea → los encabezados de artículo quedan aislados
            parrafos = [_elem_full_text(p).strip() for p in texto_elem.iter("p")]
            texto = "\n".join(p for p in parrafos if p)
    except ET.ParseError:
        # Fallback con regex cuando el XML está malformado
        raw = resp.text
        titulo = _extract_xml_tag(raw, "titulo") or doc_id
        fecha  = _extract_xml_tag(raw, "fecha_publicacion") or ""
        texto  = re.sub(r"<[^>]+>", " ", _extract_xml_tag(raw, "texto") or "")

    texto = re.sub(r"\n{3,}", "\n\n", texto).strip()
    if not texto:
        return None

    return {
        "id":     doc_id,
        "titulo": titulo,
        "fecha":  fecha,
        "texto":  texto,
        "url":    f"https://www.boe.es/buscar/doc.php?id={doc_id}",
    }


def _et_text(root, xpath: str) -> str:
    elem = root.find(xpath)
    return _elem_full_text(elem).strip() if elem is not None else ""


def _elem_full_text(elem) -> str:
    """Concatena todo el texto de un elemento y sus descendientes."""
    parts = [elem.text or ""]
    for child in elem:
        parts.append(_elem_full_text(child))
        parts.append(child.tail or "")
    return "".join(parts)


def _extract_xml_tag(xml: str, tag: str) -> str:
    match = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", xml, re.DOTALL)
    return match.group(1).strip() if match else ""


# ── 2. Chunking por artículo ──────────────────────────────

def chunk_document(doc: dict) -> list[dict]:
    """
    Divide el texto del documento en chunks.
    1. Intenta dividir por artículos (regex _ART_RE).
       - Extrae número y título de cada artículo como metadato.
       - Artículos muy largos se sub-dividen con sliding window.
       - El preámbulo previo al primer artículo se guarda como chunk propio.
    2. Si no hay artículos (< 2 matches), usa sliding window.
    """
    texto = doc["texto"]
    chunks = _split_by_article(texto, doc)
    if not chunks:
        chunks = _sliding_window(texto, doc)
    log.debug(f"  {doc['id']}: {len(chunks)} chunks ({chunks[0]['estrategia'] if chunks else '-'})")
    return chunks


def _split_by_article(texto: str, doc: dict) -> list[dict]:
    matches = list(_ART_RE.finditer(texto))
    if len(matches) < 2:
        return []

    chunks = []

    # Preámbulo: texto anterior al primer artículo
    preamble = texto[:matches[0].start()].strip()
    if len(preamble) > 150:
        chunks.append(_make_chunk(doc, preamble, 0, "articulo",
                                  art_num="preambulo", art_titulo="Preámbulo"))

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(texto)
        art_text = texto[start:end].strip()
        art_num   = m.group(1)
        art_titulo = m.group(2).strip().rstrip(".")

        if len(art_text) < 80:
            continue

        # Artículos grandes: sub-chunking con sliding window
        if len(art_text.split()) > CHUNK_SIZE + CHUNK_OVERLAP:
            for frag in _window_words(art_text, CHUNK_SIZE, CHUNK_OVERLAP):
                chunks.append(_make_chunk(doc, frag, len(chunks), "articulo_split",
                                          art_num=art_num, art_titulo=art_titulo))
        else:
            chunks.append(_make_chunk(doc, art_text, len(chunks), "articulo",
                                      art_num=art_num, art_titulo=art_titulo))

    return chunks


def _sliding_window(texto: str, doc: dict) -> list[dict]:
    return [
        _make_chunk(doc, frag, j, "sliding")
        for j, frag in enumerate(_window_words(texto, CHUNK_SIZE, CHUNK_OVERLAP))
    ]


def _window_words(texto: str, size: int, overlap: int) -> list[str]:
    palabras = texto.split()
    step = max(1, size - overlap)
    return [
        " ".join(palabras[i: i + size])
        for i in range(0, len(palabras), step)
        if len(palabras[i: i + size]) > 10
    ]


def _make_chunk(doc: dict, texto: str, idx: int, estrategia: str,
                art_num: str = "", art_titulo: str = "") -> dict:
    return {
        "id":         f"{doc['id']}_chunk{idx:03d}",
        "texto":      texto,
        "doc_id":     doc["id"],
        "titulo":     doc["titulo"],
        "fecha":      doc["fecha"],
        "url":        doc["url"],
        "estrategia": estrategia,
        "chunk_idx":  idx,
        "art_num":    art_num,
        "art_titulo": art_titulo,
    }


# ── 3. ChromaDB ───────────────────────────────────────────

def get_chroma_collection(path: str = CHROMA_PATH, collection: str = COLLECTION):
    """
    Crea o abre la colección ChromaDB persistente.
    Usa bge-m3 como función de embedding.
    """
    log.info(f"Iniciando ChromaDB en {path}...")
    client = chromadb.PersistentClient(path=path)

    embed_fn = SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL,
        device="cpu",         # cambiar a "cuda" si hay GPU local
        normalize_embeddings=True,
    )

    collection_obj = client.get_or_create_collection(
        name=collection,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    log.info(f"Colección '{collection}': {collection_obj.count()} documentos existentes")
    return collection_obj


def ingest_chunks(collection, chunks: list[dict], batch_size: int = 50):
    """
    Inserta chunks en ChromaDB en batches.
    Salta los que ya existen (por ID).
    """
    existing_ids = set()
    if collection.count() > 0:
        existing = collection.get(include=[])
        existing_ids = set(existing["ids"])

    nuevos = [c for c in chunks if c["id"] not in existing_ids]
    if not nuevos:
        log.info("Todos los chunks ya estaban en la colección — nada que insertar")
        return 0

    for i in range(0, len(nuevos), batch_size):
        batch = nuevos[i : i + batch_size]
        collection.add(
            ids       = [c["id"] for c in batch],
            documents = [c["texto"] for c in batch],
            metadatas = [
                {
                    "doc_id":     c["doc_id"],
                    "titulo":     c["titulo"][:200],
                    "fecha":      c["fecha"],
                    "url":        c["url"],
                    "estrategia": c["estrategia"],
                    "chunk_idx":  c["chunk_idx"],
                    "art_num":    c.get("art_num", ""),
                    "art_titulo": c.get("art_titulo", "")[:150],
                }
                for c in batch
            ],
        )
        log.info(f"  Insertados {min(i + batch_size, len(nuevos))}/{len(nuevos)} chunks")

    return len(nuevos)


# ── 4. Retrieval de prueba ────────────────────────────────

def test_retrieval(collection, query: str, n: int = 5):
    """
    Prueba rápida: recupera los n chunks más relevantes para una query.
    """
    results = collection.query(
        query_texts=[query],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )

    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        print(f"\n[{i+1}] Score: {1-dist:.4f} | {meta['titulo'][:60]}...")
        print(f"     Fecha: {meta['fecha']} | URL: {meta['url']}")
        print(f"     Chunk: {doc[:200]}...")
    print()


# ── Main ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pipeline BOE → ChromaDB")
    parser.add_argument("--query",  default="contrato menor administración pública",
                        help="Término de búsqueda en el BOE")
    parser.add_argument("--n",      type=int, default=30,
                        help="Número de documentos a descargar")
    parser.add_argument("--test",   type=str, default=None,
                        help="Query de prueba al final (opcional)")
    parser.add_argument("--chroma", default=CHROMA_PATH,
                        help="Directorio ChromaDB")
    args = parser.parse_args()

    # Paso 1: Buscar documentos
    log.info(f"=== PASO 1: Búsqueda BOE — query='{args.query}' n={args.n} ===")
    doc_refs = search_boe(args.query, args.n)

    # Paso 2: Descargar y parsear
    log.info("=== PASO 2: Descarga y parseo de documentos ===")
    docs = []
    for ref in doc_refs:
        doc = fetch_boe_document(ref["id"])
        if doc:
            docs.append(doc)
            log.info(f"  ✓ {doc['id']} — {doc['titulo'][:60]}")
        time.sleep(0.3)

    log.info(f"Documentos descargados con texto: {len(docs)}/{len(doc_refs)}")

    # Paso 3: Chunking
    log.info("=== PASO 3: Chunking por artículo ===")
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
    log.info(f"Total chunks generados: {len(all_chunks)}")

    # Paso 4: Ingest en ChromaDB
    log.info("=== PASO 4: Embeddings + ChromaDB ===")
    collection = get_chroma_collection(args.chroma)
    inserted = ingest_chunks(collection, all_chunks)
    log.info(f"Chunks nuevos insertados: {inserted}")
    log.info(f"Total en colección: {collection.count()}")

    # Paso 5: Test opcional
    if args.test:
        log.info("=== PASO 5: Test de retrieval ===")
        test_retrieval(collection, args.test)

    # Guardar resumen
    summary = {
        "query":      args.query,
        "docs_found": len(doc_refs),
        "docs_ok":    len(docs),
        "chunks":     len(all_chunks),
        "inserted":   inserted,
        "total_db":   collection.count(),
    }
    Path("boe_pipeline_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log.info("Resumen guardado en boe_pipeline_summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
