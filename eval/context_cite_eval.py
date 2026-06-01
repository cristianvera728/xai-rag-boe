#!/usr/bin/env python3
"""
context_cite_eval.py — ContextCite-style attribution for RAG-BOE.

Implements the ContextCite ablation algorithm (Anthropic/MIT 2024) using the
running vLLM server.  For each QA pair in rag_answers.json:

  1. Split the 5 context chunks into sentences (context parts).
  2. Sample N_ABLATIONS random binary masks over the context parts.
  3. For each mask, compute log P(answer | ablated_context) via the vLLM
     completions endpoint (echo=True, logprobs=1 → teacher-forcing).
  4. Fit a Ridge regression: log_prob ~ mask @ w + bias.
     Attribution score of part i = w[i] (effect on answer log prob).
  5. Aggregate part-level scores to chunk level (sum of positive scores).
  6. Compute per-item metrics and global aggregates.

Output: eval/context_cite_results.json
"""

import argparse
import json
import logging
import math
import pathlib
import re
import sys
import time

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "Eres un asistente jurídico especializado en normativa española y europea. "
    "Responde de forma precisa y fundamentada exclusivamente en los fragmentos "
    "de normativa proporcionados. Si la información no aparece en los fragmentos, "
    "indícalo explícitamente."
)


def build_user_content(context_parts: list[str], active_mask: list[bool], question: str) -> str:
    """Reconstruct user message keeping only active context parts."""
    active_parts = [p for p, keep in zip(context_parts, active_mask) if keep]
    context_str = "\n".join(active_parts) if active_parts else "[Sin contexto]"
    return f"Contexto normativo:\n{context_str}\n\nPregunta: {question}"


def split_context_into_parts(contexts: list[str], context_meta: list[dict]) -> tuple[list[str], list[int]]:
    """
    Split the 5 retrieved chunks into individual sentences (context parts).
    Returns:
        parts: list of sentence strings
        part_to_chunk: list of chunk index (0-4) for each part
    """
    parts: list[str] = []
    part_to_chunk: list[int] = []

    for chunk_idx, (ctx, meta) in enumerate(zip(contexts, context_meta)):
        doc_id = meta.get("doc_id", f"doc_{chunk_idx}")
        art = meta.get("art_num", "")
        header = f"[Fragmento {chunk_idx + 1} | {doc_id}" + (f" art.{art}" if art else "") + "]"

        # Add header as its own part so it's always active when its chunk is active
        parts.append(header)
        part_to_chunk.append(chunk_idx)

        # Split chunk text into sentences
        sentences = _split_sentences(ctx.strip())
        for sent in sentences:
            parts.append(sent)
            part_to_chunk.append(chunk_idx)

    return parts, part_to_chunk


def _split_sentences(text: str) -> list[str]:
    """Heuristic sentence splitter for Spanish legal text."""
    # Split on '. ', '.\n', '! ', '? ' — keep the delimiter with the sentence
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    result = []
    for s in raw:
        s = s.strip()
        if len(s) > 10:  # skip very short fragments
            result.append(s)
    return result if result else [text.strip()]


# ──────────────────────────────────────────────────────────────────────────────
# vLLM log-prob computation
# ──────────────────────────────────────────────────────────────────────────────

def wait_for_vllm(base_url: str, max_wait: int = 300) -> None:
    import urllib.request
    elapsed = 0
    while True:
        try:
            urllib.request.urlopen(f"{base_url}/v1/models", timeout=3)
            log.info(f"vLLM listo en {base_url} (esperado {elapsed}s)")
            return
        except Exception:
            if elapsed >= max_wait:
                raise RuntimeError(f"vLLM no respondió en {max_wait}s")
            time.sleep(5)
            elapsed += 5


def _sanitize_text(text: str) -> str:
    """Remove null bytes and Llama special-token patterns from user-provided text."""
    # Null bytes break HTTP/JSON
    text = text.replace('\x00', ' ')
    # Raw <|...|> sequences in user text get tokenized as Llama special tokens,
    # which can cause vLLM to reject the completions request.
    text = re.sub(r'<\|([^|]*)\|>', r'[\1]', text)
    return text


def get_log_prob(
    client,
    model_id: str,
    tokenizer,
    system: str,
    user_content: str,
    answer: str,
) -> float | None:
    """
    Compute log P(answer | system, user_content) via vLLM completions endpoint.

    Uses the model's chat template to format the full sequence, then requests
    echo=True + logprobs=1 to get per-token log probs without generating new tokens.
    """
    # Sanitize user-provided text before inserting into chat template
    user_content_safe = _sanitize_text(user_content)
    answer_safe = _sanitize_text(answer)

    messages_with_answer = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content_safe},
        {"role": "assistant", "content": answer_safe},
    ]
    messages_prefix = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content_safe},
    ]

    try:
        full_text = tokenizer.apply_chat_template(
            messages_with_answer, tokenize=False, add_generation_prompt=False
        )
        prefix_text = tokenizer.apply_chat_template(
            messages_prefix, tokenize=False, add_generation_prompt=True
        )
    except Exception as e:
        log.warning(f"apply_chat_template failed: {e}")
        return None

    prefix_len = len(prefix_text)

    try:
        resp = client.completions.create(
            model=model_id,
            prompt=full_text,
            max_tokens=1,       # minimum required by some backends
            echo=True,
            logprobs=1,
            temperature=0.0,
        )
    except Exception as e:
        # Log the actual HTTP error body so we can diagnose vLLM 500s
        body = getattr(getattr(e, "response", None), "text", None) or str(e)
        log.warning(f"completions.create failed: {body[:300]}")
        return None

    lp = resp.choices[0].logprobs
    if lp is None:
        return None

    offsets = lp.text_offset or []
    token_lps = lp.token_logprobs or []

    answer_lps = [
        lp_val
        for off, lp_val in zip(offsets, token_lps)
        if off >= prefix_len and lp_val is not None and not math.isnan(lp_val)
    ]

    return float(sum(answer_lps)) if answer_lps else None


# ──────────────────────────────────────────────────────────────────────────────
# ContextCite ablation + attribution
# ──────────────────────────────────────────────────────────────────────────────

def run_context_cite(
    client,
    model_id: str,
    tokenizer,
    item: dict,
    n_ablations: int,
    mask_prob: float = 0.5,
) -> dict | None:
    """
    Run ContextCite for one QA item.
    Returns attribution dict or None on failure.
    """
    context_parts, part_to_chunk = split_context_into_parts(
        item["contexts"], item["context_meta"]
    )
    n_parts = len(context_parts)
    n_chunks = len(item["contexts"])

    if n_parts == 0:
        return None

    rng = np.random.default_rng(seed=42)

    # Sample ablation masks: shape (n_ablations, n_parts), dtype bool
    masks = rng.random((n_ablations, n_parts)) < mask_prob  # True = part is kept

    log_probs: list[float] = []
    valid_masks: list[np.ndarray] = []

    # Full context log prob (all parts active) as reference
    full_user = build_user_content(context_parts, [True] * n_parts, item["question"])
    lp_full = get_log_prob(client, model_id, tokenizer, SYSTEM_PROMPT, full_user, item["answer"])
    if lp_full is None:
        log.warning(f"    full-context log prob failed for {item['id']}")
        return None

    log.debug(f"    log P(full) = {lp_full:.3f}")

    for j, mask in enumerate(masks):
        user_content = build_user_content(context_parts, mask.tolist(), item["question"])
        lp = get_log_prob(client, model_id, tokenizer, SYSTEM_PROMPT, user_content, item["answer"])
        if lp is not None:
            log_probs.append(lp)
            valid_masks.append(mask.astype(float))

    if len(valid_masks) < 10:
        log.warning(f"    Too few valid ablations ({len(valid_masks)}) for {item['id']}")
        return None

    # Fit Ridge regression: log_prob ~ mask @ w + bias
    from sklearn.linear_model import Ridge
    X = np.stack(valid_masks)           # (n_valid, n_parts)
    y = np.array(log_probs)             # (n_valid,)
    reg = Ridge(alpha=1.0)
    reg.fit(X, y)
    part_scores = reg.coef_             # (n_parts,) — raw attribution per part

    # Aggregate to chunk level (sum of raw scores per chunk)
    chunk_scores_raw = np.zeros(n_chunks)
    for part_idx, chunk_idx in enumerate(part_to_chunk):
        chunk_scores_raw[chunk_idx] += part_scores[part_idx]

    # Normalize to [0,1] using positive scores only
    positive = np.clip(chunk_scores_raw, 0, None)
    total_pos = positive.sum()
    chunk_scores_norm = (positive / total_pos).tolist() if total_pos > 0 else [1 / n_chunks] * n_chunks

    return {
        "part_scores": part_scores.tolist(),
        "part_to_chunk": part_to_chunk,
        "context_parts_preview": [p[:80] for p in context_parts],
        "chunk_scores_raw": chunk_scores_raw.tolist(),
        "chunk_scores_norm": chunk_scores_norm,
        "n_parts": n_parts,
        "n_valid_ablations": len(valid_masks),
        "lp_full": lp_full,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Per-item metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(chunk_scores_norm: list[float]) -> dict:
    scores = np.array(chunk_scores_norm)
    n = len(scores)
    top1_idx = int(scores.argmax())
    top1 = float(scores[top1_idx])

    # Shannon entropy (base e)
    nonzero = scores[scores > 1e-10]
    entropy = float(-np.sum(nonzero * np.log(nonzero))) if len(nonzero) else 0.0
    max_entropy = math.log(n) if n > 1 else 1.0
    concentration = float(1.0 - entropy / max_entropy)

    return {
        "top1_chunk": top1_idx,
        "top1_score": top1,
        "entropy": entropy,
        "concentration": concentration,
        "n_chunks_above_10pct": int((scores > 0.10).sum()),
        "n_chunks_above_20pct": int((scores > 0.20).sum()),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="ContextCite attribution evaluation for RAG-BOE")
    p.add_argument("--vllm-url",     default="http://localhost:8000")
    p.add_argument("--model",        default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--answers",      default="eval/rag_answers.json")
    p.add_argument("--out",          default="eval/context_cite_results.json")
    p.add_argument("--num-ablations",type=int, default=64)
    p.add_argument("--mask-prob",    type=float, default=0.5)
    p.add_argument("--max-items",    type=int, default=None,
                   help="Limit to first N items (useful for smoke tests)")
    p.add_argument("--wait-vllm",    type=int, default=300,
                   help="Seconds to wait for vLLM to be ready (0 = skip wait)")
    p.add_argument("--cache-dir",    default=None,
                   help="HF cache dir (None = use HF_HOME env var, recommended)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.wait_vllm > 0:
        wait_for_vllm(args.vllm_url, args.wait_vllm)

    from openai import OpenAI
    from transformers import AutoTokenizer

    client = OpenAI(base_url=f"{args.vllm_url}/v1", api_key="dummy")

    log.info(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)

    answers = json.loads(pathlib.Path(args.answers).read_text())
    if args.max_items:
        answers = answers[: args.max_items]

    log.info(f"Items to process : {len(answers)}")
    log.info(f"Ablations/item   : {args.num_ablations}")
    log.info(f"Mask probability : {args.mask_prob}")

    per_item = []
    t0 = time.time()

    for i, item in enumerate(answers):
        item_id = item["id"]
        t_item = time.time()
        log.info(
            f"[{i + 1}/{len(answers)}] {item_id} | {item.get('topic')} | {item.get('difficulty')}"
        )

        try:
            cc_result = run_context_cite(
                client=client,
                model_id=args.model,
                tokenizer=tokenizer,
                item=item,
                n_ablations=args.num_ablations,
                mask_prob=args.mask_prob,
            )

            if cc_result is None:
                per_item.append({
                    "id": item_id,
                    "topic": item.get("topic"),
                    "difficulty": item.get("difficulty"),
                    "error": "run_context_cite returned None",
                })
                continue

            metrics = compute_metrics(cc_result["chunk_scores_norm"])
            elapsed = time.time() - t_item
            total_elapsed = time.time() - t0
            avg = total_elapsed / (i + 1)
            eta_min = avg * (len(answers) - i - 1) / 60

            log.info(
                f"  n_parts={cc_result['n_parts']} "
                f"valid_abl={cc_result['n_valid_ablations']} "
                f"top1={metrics['top1_score']:.3f} "
                f"conc={metrics['concentration']:.3f} "
                f"ETA {eta_min:.1f}min  [{elapsed:.1f}s]"
            )

            per_item.append({
                "id": item_id,
                "topic": item.get("topic"),
                "difficulty": item.get("difficulty"),
                **metrics,
                "chunk_scores_norm": cc_result["chunk_scores_norm"],
                "chunk_scores_raw": cc_result["chunk_scores_raw"],
                "n_parts": cc_result["n_parts"],
                "n_valid_ablations": cc_result["n_valid_ablations"],
                "lp_full": cc_result["lp_full"],
                "context_parts_preview": cc_result["context_parts_preview"],
                "part_scores": cc_result["part_scores"],
                "part_to_chunk": cc_result["part_to_chunk"],
            })

        except Exception as e:
            log.error(f"  UNEXPECTED ERROR: {e}", exc_info=True)
            per_item.append({
                "id": item_id,
                "topic": item.get("topic"),
                "difficulty": item.get("difficulty"),
                "error": str(e),
            })

    # ── Aggregate ────────────────────────────────────────────────────────────
    valid = [x for x in per_item if "error" not in x]

    def _mean(key):
        vals = [x[key] for x in valid if key in x and x[key] is not None]
        return float(np.mean(vals)) if vals else None

    aggregate = {
        "n_items": len(per_item),
        "n_valid": len(valid),
        "n_errors": len(per_item) - len(valid),
        "mean_top1_score": _mean("top1_score"),
        "mean_entropy": _mean("entropy"),
        "mean_concentration": _mean("concentration"),
        "mean_n_chunks_above_10pct": _mean("n_chunks_above_10pct"),
        "mean_n_chunks_above_20pct": _mean("n_chunks_above_20pct"),
        "mean_valid_ablations": _mean("n_valid_ablations"),
        "mean_n_parts": _mean("n_parts"),
    }

    output = {
        "config": {
            "model": args.model,
            "num_ablations": args.num_ablations,
            "mask_prob": args.mask_prob,
            "n_items_total": len(answers),
        },
        "aggregate": aggregate,
        "per_item": per_item,
    }

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))

    log.info("\n=== RESUMEN ContextCite ===")
    log.info(f"Items válidos:         {aggregate['n_valid']}/{aggregate['n_items']}")
    if aggregate["mean_top1_score"] is not None:
        log.info(f"Mean top-1 score:      {aggregate['mean_top1_score']:.4f}")
        log.info(f"Mean entropy:          {aggregate['mean_entropy']:.4f}")
        log.info(f"Mean concentration:    {aggregate['mean_concentration']:.4f}")
        log.info(f"Mean chunks >10%:      {aggregate['mean_n_chunks_above_10pct']:.2f}")
        log.info(f"Mean chunks >20%:      {aggregate['mean_n_chunks_above_20pct']:.2f}")
    log.info(f"Saved to: {args.out}")


if __name__ == "__main__":
    main()
