"""
ingest_from_kimi.py
-------------------
Ingesta en ChromaDB los documentos referenciados en dataset_qa_boe.json.

Soporta dos tipos de doc_id:
  - BOE-*              → https://www.boe.es/diario_boe/xml.php?id={doc_id}
  - DOUE-L-2016-80807  → RGPD vía EUR-Lex, CELEX:32016R0679

Añade el campo "qa_ids" (string CSV) en los metadatos de cada chunk.
Usa upsert para no duplicar si ya existe un chunk del mismo documento.

Uso:
  python ingest_from_kimi.py --dataset dataset_qa_boe.json
  python ingest_from_kimi.py --dataset dataset_qa_boe.json --chroma ./chroma_boe --dry-run
"""

import argparse
import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests

# Reutilizar chunking y ChromaDB de boe_pipeline
from boe_pipeline import (
    fetch_boe_document,
    chunk_document,
    get_chroma_collection,
    _elem_full_text,
    CHROMA_PATH,
    COLLECTION,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

HEADERS = {"User-Agent": "XAI-RAG-Research/1.0 (UA-ENIA-Catedra)"}

# Mapa doc_id DOUE → metadatos fijos + URL de descarga XML
EURLEX_DOCS: dict[str, dict] = {
    "DOUE-L-2016-80807": {
        "celex":   "32016R0679",
        "titulo":  "Reglamento (UE) 2016/679 del Parlamento Europeo y del Consejo, "
                   "de 27 de abril de 2016 (RGPD)",
        "fecha":   "20160504",
        # URL del chunk metadata: coincide con doc_url del dataset
        "url":     "https://www.boe.es/buscar/doc.php?id=DOUE-L-2016-80807",
        "xml_url": "https://eur-lex.europa.eu/legal-content/ES/TXT/XML/?uri=CELEX:32016R0679",
    },
}


# ── EUR-Lex fetcher ───────────────────────────────────────

def fetch_eurlex_document(doc_id: str) -> dict | None:
    """
    Descarga el reglamento desde EUR-Lex en formato XML Formex4.
    Devuelve dict compatible con chunk_document(): {id, titulo, fecha, texto, url}.
    """
    if doc_id not in EURLEX_DOCS:
        log.warning(f"  No hay mapping CELEX para {doc_id}")
        return None

    meta = EURLEX_DOCS[doc_id]
    log.info(f"  Descargando EUR-Lex ({meta['celex']}) …")
    try:
        resp = requests.get(meta["xml_url"], timeout=90, headers=HEADERS)
        resp.raise_for_status()
    except requests.RequestException as e:
        log.warning(f"  ERROR descargando {doc_id} desde EUR-Lex: {e}")
        return None

    texto = _parse_eurlex_xml(resp.content)
    if not texto:
        log.warning(f"  Sin texto útil en {doc_id}")
        return None

    words = len(texto.split())
    arts  = len(re.findall(r"Artículo\s+\d+", texto, re.IGNORECASE))
    log.info(f"  EUR-Lex OK: {words} palabras, ~{arts} artículos detectados")

    return {
        "id":     doc_id,
        "titulo": meta["titulo"],
        "fecha":  meta["fecha"],
        "texto":  texto,
        "url":    meta["url"],
    }


def _strip_ns(tag: str) -> str:
    """Elimina el prefijo de namespace de un tag XML ({uri}LocalName → LocalName)."""
    return tag.split("}")[-1] if "}" in tag else tag


def _parse_eurlex_xml(content: bytes) -> str:
    """
    Extrae texto del XML Formex4 de EUR-Lex conservando la estructura de artículos.

    Estrategia 1: elementos <ARTICLE> del Formex4 → reconstruye
                  "Artículo N. Título\ncuerpo" por artículo.
    Estrategia 2: cualquier nodo con texto leaf (ALINEA, P, TXT, …).
    Fallback:     stripping de tags con regex.
    """
    try:
        root = ET.fromstring(content)
    except ET.ParseError as e:
        log.warning(f"  XML parse error EUR-Lex: {e} — fallback regex")
        raw = content.decode("utf-8", errors="replace")
        return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", raw)).strip()

    # ── Estrategia 1: ARTICLE elements ───────────────────
    articles = [n for n in root.iter() if _strip_ns(n.tag) == "ARTICLE"]
    if articles:
        parts = []
        for art in articles:
            ti_node = next(
                (n for n in art if _strip_ns(n.tag) == "TI.ART"), None
            )
            sti_node = next(
                (n for n in art if _strip_ns(n.tag) in ("STI.ART", "SUBTITLE")), None
            )

            art_num_raw = _elem_full_text(ti_node).strip() if ti_node is not None else ""
            art_subtitle = _elem_full_text(sti_node).strip() if sti_node is not None else ""

            # Normalizar a "Artículo N." para que _ART_RE lo detecte con certeza
            m = re.search(r"\d+[\w\-]*", art_num_raw)
            num = m.group() if m else art_num_raw
            if art_subtitle:
                header = f"Artículo {num}. {art_subtitle}"
            else:
                header = f"Artículo {num}."

            # Cuerpo: todo el texto del artículo excepto TI.ART y STI.ART
            SKIP_TAGS = {"TI.ART", "STI.ART", "SUBTITLE"}
            body_lines = []
            for node in art.iter():
                tag = _strip_ns(node.tag)
                if tag in SKIP_TAGS:
                    continue
                # Solo nodos hoja con texto propio (no repetir texto de ancestros)
                own_text = (node.text or "").strip()
                if own_text and not any(own_text in bl for bl in body_lines):
                    body_lines.append(own_text)
                tail = (node.tail or "").strip()
                if tail and not any(tail in bl for bl in body_lines):
                    body_lines.append(tail)

            if header or body_lines:
                parts.append(header + "\n" + "\n".join(body_lines))

        if parts:
            texto = "\n\n".join(parts)
            return re.sub(r"\n{3,}", "\n\n", texto).strip()

    # ── Estrategia 2: párrafos genéricos ─────────────────
    PARA_TAGS = {"P", "ALINEA", "PARAG", "TXT", "STI", "TITLE", "TI"}
    seen: set[str] = set()
    lines: list[str] = []
    for node in root.iter():
        if _strip_ns(node.tag).upper() in PARA_TAGS:
            t = _elem_full_text(node).strip()
            if t and t not in seen:
                seen.add(t)
                lines.append(t)

    texto = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", texto).strip()


# ── Dispatcher ────────────────────────────────────────────

def fetch_document(doc_id: str) -> dict | None:
    # La API del BOE sirve tanto documentos BOE-* como DOUE-* con el mismo endpoint
    if doc_id.startswith("BOE-") or doc_id.startswith("DOUE-"):
        return fetch_boe_document(doc_id)
    # Fallback a EUR-Lex para doc_ids DOUE sin soporte en la API del BOE
    if doc_id in EURLEX_DOCS:
        return fetch_eurlex_document(doc_id)
    log.warning(f"  Tipo de doc_id no reconocido: {doc_id}")
    return None


# ── ChromaDB upsert con qa_ids ────────────────────────────

def upsert_chunks(collection, chunks: list[dict], batch_size: int = 50) -> int:
    """
    Upsert de chunks en ChromaDB (inserta o actualiza según ID).
    Incluye qa_ids (string CSV) en los metadatos de cada chunk.
    """
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i: i + batch_size]
        collection.upsert(
            ids=      [c["id"] for c in batch],
            documents=[c["texto"] for c in batch],
            metadatas=[
                {
                    "doc_id":     c["doc_id"],
                    "titulo":     c["titulo"][:200],
                    "fecha":      c["fecha"],
                    "url":        c["url"],
                    "estrategia": c["estrategia"],
                    "chunk_idx":  c["chunk_idx"],
                    "art_num":    c.get("art_num", ""),
                    "art_titulo": c.get("art_titulo", "")[:150],
                    "qa_ids":     c.get("qa_ids", ""),
                }
                for c in batch
            ],
        )
        log.info(f"    upsert {min(i + batch_size, len(chunks))}/{len(chunks)}")
    return len(chunks)


# ── Main ──────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest de dataset_qa_boe.json → ChromaDB"
    )
    parser.add_argument("--dataset",    default="dataset_qa_boe.json")
    parser.add_argument("--chroma",     default=CHROMA_PATH)
    parser.add_argument("--collection", default=COLLECTION)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Descarga y chunkea sin escribir en ChromaDB"
    )
    args = parser.parse_args()

    # ── 1. Cargar dataset ─────────────────────────────────
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        log.error(f"Dataset no encontrado: {dataset_path.resolve()}")
        raise SystemExit(1)

    qa_items: list[dict] = json.loads(dataset_path.read_text(encoding="utf-8"))
    log.info(f"Dataset: {len(qa_items)} pares QA en {dataset_path}")

    # Mapeo doc_id → lista ordenada de qa_ids
    doc_to_qas: dict[str, list[str]] = defaultdict(list)
    for item in qa_items:
        doc_to_qas[item["doc_id"]].append(item["id"])

    log.info(f"Documentos únicos referenciados: {len(doc_to_qas)}")

    # ── 2. Inicializar ChromaDB ───────────────────────────
    collection = None
    if not args.dry_run:
        collection = get_chroma_collection(args.chroma, args.collection)
    else:
        log.info("DRY RUN — no se escribirá en ChromaDB")

    # ── 3. Determinar docs ya en ChromaDB (resume fiable) ─
    docs_in_db: set[str] = set()
    if collection is not None and collection.count() > 0:
        log.info("Consultando ChromaDB para resume...")
        existing = collection.get(include=["metadatas"])
        docs_in_db = {m["doc_id"] for m in existing["metadatas"]}
        if docs_in_db:
            log.info(f"Resume: {len(docs_in_db)} doc_ids ya en ChromaDB — se saltan")

    # ── 4. Procesar cada documento ────────────────────────
    per_doc: list[dict] = []
    failed:  list[str]  = []
    total_chunks   = 0
    total_upserted = 0

    for doc_id, qa_ids in doc_to_qas.items():
        if doc_id in docs_in_db:
            log.info(f"→ {doc_id}  SKIP (ya en ChromaDB)")
            per_doc.append({
                "doc_id": doc_id, "status": "ok_existing",
                "qa_ids": qa_ids, "chunks": 0, "upserted": 0,
            })
            continue

        log.info(f"→ {doc_id}  qa=[{', '.join(qa_ids)}]")

        doc = fetch_document(doc_id)
        if doc is None:
            failed.append(doc_id)
            per_doc.append({
                "doc_id": doc_id, "status": "error_download",
                "qa_ids": qa_ids, "chunks": 0, "upserted": 0,
            })
            time.sleep(0.4)
            continue

        chunks = chunk_document(doc)
        if not chunks:
            log.warning(f"  Sin chunks para {doc_id}")
            failed.append(doc_id)
            per_doc.append({
                "doc_id": doc_id, "status": "error_no_chunks",
                "qa_ids": qa_ids, "chunks": 0, "upserted": 0,
            })
            time.sleep(0.4)
            continue

        # Anotar qa_ids en cada chunk como CSV (ChromaDB no admite listas)
        qa_ids_csv = ",".join(qa_ids)
        for chunk in chunks:
            chunk["qa_ids"] = qa_ids_csv

        strategy = chunks[0]["estrategia"]
        log.info(
            f"  {len(chunks)} chunks  estrategia={strategy}"
            f"  palabras={len(doc['texto'].split())}"
        )

        upserted = 0
        if collection is not None:
            upserted = upsert_chunks(collection, chunks)

        total_chunks   += len(chunks)
        total_upserted += upserted

        per_doc.append({
            "doc_id":     doc_id,
            "status":     "ok",
            "titulo":     doc["titulo"][:120],
            "fecha":      doc["fecha"],
            "qa_ids":     qa_ids,
            "n_qa":       len(qa_ids),
            "chunks":     len(chunks),
            "upserted":   upserted,
            "estrategia": strategy,
            "palabras":   len(doc["texto"].split()),
        })

        time.sleep(0.4)

    # ── 5. Informe ────────────────────────────────────────
    total_in_db = collection.count() if collection else -1

    report = {
        "dataset":           args.dataset,
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        "dry_run":           args.dry_run,
        "docs_total":        len(doc_to_qas),
        "docs_ok":           len(per_doc) - len(failed),
        "docs_failed":       failed,
        "chunks_generated":  total_chunks,
        "chunks_upserted":   total_upserted,
        "total_in_chroma":   total_in_db,
        "per_doc":           per_doc,
    }

    report_path = Path("ingest_report.json")
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log.info("=" * 60)
    log.info(f"Docs OK    : {report['docs_ok']}/{report['docs_total']}")
    if failed:
        log.warning(f"Docs FAIL  : {failed}")
    log.info(f"Chunks gen : {total_chunks}   upserted: {total_upserted}")
    log.info(f"Total DB   : {total_in_db}")
    log.info(f"Informe    : {report_path.resolve()}")


if __name__ == "__main__":
    main()
