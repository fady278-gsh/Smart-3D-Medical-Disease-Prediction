from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

from datasets import load_dataset

from . import config as cfg

DatasetRef = Union[str, Tuple[str, str]]

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: Optional[str]) -> str:
    """Normalize whitespace and strip HTML tags from a raw text field."""
    if not text:
        return ""
    text = text.replace("\r", " ").replace("\n", " ").strip()
    text = _HTML_TAG_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text


def hf_stream_text(
    dataset_ref: DatasetRef,
    split: str = "train",
    field_candidates: Optional[List[str]] = None,
    fraction: Optional[float] = None,
    max_samples: Optional[int] = None,
    trust_remote_code: bool = False,
    seed: int = cfg.SEED,
) -> Iterator[str]:
    """Stream cleaned text records from a Hugging Face dataset.

    dataset_ref is either a dataset name or a (name, config) tuple.
    field_candidates is the ordered list of column names to look for text
    in; the first non-empty match is used. max_samples takes precedence
    over fraction when both are given.
    """
    name = dataset_ref[0] if isinstance(dataset_ref, tuple) else dataset_ref
    subset = dataset_ref[1] if isinstance(dataset_ref, tuple) else None
    label = f"{name} ({subset})" if subset else name

    try:
        ds = load_dataset(
            name,
            subset,
            split=split,
            streaming=True,
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:
        print(f"[WARN] could not load '{label}': {exc}")
        return

    if max_samples is not None:
        ds = ds.shuffle(seed=seed, buffer_size=10_000).take(max_samples)
    elif fraction is not None:
        approx_take = max(1, int(fraction * 100_000))
        ds = ds.shuffle(seed=seed, buffer_size=10_000).take(approx_take)

    candidates = field_candidates or ["text", "article", "abstract"]
    n_yielded = 0

    for item in ds:
        text = ""
        for field_name in candidates:
            value = item.get(field_name)
            if value:
                text = value
                break

        text = clean_text(text)
        if len(text) < 20:
            continue

        n_yielded += 1
        yield text

    print(f"[DATA] {label}: {n_yielded} records yielded")


def csv_stream_qas(
    csv_path: str,
    q_col: str = "question",
    a_col: str = "answer",
    max_rows: Optional[int] = None,
) -> Iterator[str]:
    """Stream 'Q: ... A: ...' strings from a doctor/patient QA CSV."""
    path = Path(csv_path)
    if not path.exists():
        print(f"[WARN] CSV not found: {csv_path}")
        return

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if q_col not in (reader.fieldnames or []) or a_col not in (reader.fieldnames or []):
            print(f"[WARN] columns '{q_col}'/'{a_col}' not found. Available: {reader.fieldnames}")
            return

        n_yielded = 0
        for row in reader:
            if max_rows is not None and n_yielded >= max_rows:
                break
            question = clean_text(row.get(q_col, ""))
            answer = clean_text(row.get(a_col, ""))
            if not question or not answer:
                continue
            n_yielded += 1
            yield f"Q: {question} A: {answer}"

    print(f"[DATA] {csv_path}: {n_yielded} Q&A pairs yielded")


def build_phase1_jsonl(
    out_path: str = cfg.PHASE1_JSONL,
    dev_mode: bool = cfg.DEV_MODE,
) -> str:
    """Build the merged Phase 1 corpus (RedPajama + PubMed + guidelines)."""
    out_path_p = Path(out_path)
    if out_path_p.exists():
        print(f"[DATA] Phase 1 corpus already exists: {out_path_p}")
        return str(out_path_p)

    out_path_p.parent.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {"redpajama": 0, "pubmed": 0, "guidelines": 0}

    with open(out_path_p, "w", encoding="utf-8") as fo:

        max_samples = cfg.PHASE1_DEV_SAMPLES["redpajama"] if dev_mode else None
        fraction = None if dev_mode else cfg.PHASE1_FULL_FRACTIONS["redpajama"]
        for txt in hf_stream_text(
            cfg.PHASE1_DATASETS["redpajama"],
            field_candidates=["text", "article", "content"],
            max_samples=max_samples,
            fraction=fraction,
        ):
            fo.write(json.dumps({"text": txt, "source": "redpajama"}, ensure_ascii=False) + "\n")
            counts["redpajama"] += 1

        max_samples = cfg.PHASE1_DEV_SAMPLES["pubmed"] if dev_mode else None
        fraction = None if dev_mode else cfg.PHASE1_FULL_FRACTIONS["pubmed"]
        for txt in hf_stream_text(
            cfg.PHASE1_DATASETS["pubmed"],
            field_candidates=["abstract", "AbstractText", "text"],
            max_samples=max_samples,
            fraction=fraction,
            trust_remote_code=True,
        ):
            fo.write(json.dumps({"text": txt, "source": "pubmed"}, ensure_ascii=False) + "\n")
            counts["pubmed"] += 1

        max_samples = cfg.PHASE1_DEV_SAMPLES["guidelines"] if dev_mode else None
        fraction = None if dev_mode else cfg.PHASE1_FULL_FRACTIONS["guidelines"]
        for txt in hf_stream_text(
            cfg.PHASE1_DATASETS["guidelines"],
            field_candidates=["text", "clean_text", "guideline"],
            max_samples=max_samples,
            fraction=fraction,
        ):
            fo.write(json.dumps({"text": txt, "source": "guidelines"}, ensure_ascii=False) + "\n")
            counts["guidelines"] += 1

    total = sum(counts.values())
    print(f"[DATA] Phase 1 corpus built: {out_path_p} ({total} records)")
    for src, n in counts.items():
        print(f"  - {src}: {n}")

    return str(out_path_p)


def build_phase2_jsonl(
    out_path: str = cfg.PHASE2_JSONL,
    csv_path: str = cfg.PHASE2_QA_CSV,
    dev_mode: bool = cfg.DEV_MODE,
) -> str:
    """Build the Phase 2 corpus (doctor-patient Q&A pairs) from a local CSV."""
    out_path_p = Path(out_path)
    if out_path_p.exists():
        print(f"[DATA] Phase 2 corpus already exists: {out_path_p}")
        return str(out_path_p)

    out_path_p.parent.mkdir(parents=True, exist_ok=True)
    max_rows = cfg.PHASE2_DEV_SAMPLES if dev_mode else None
    n_written = 0

    with open(out_path_p, "w", encoding="utf-8") as fo:
        for qa in csv_stream_qas(
            csv_path,
            q_col=cfg.PHASE2_QA_COLUMNS["question"],
            a_col=cfg.PHASE2_QA_COLUMNS["answer"],
            max_rows=max_rows,
        ):
            fo.write(json.dumps({"text": qa, "source": "medical_qa"}, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"[DATA] Phase 2 corpus built: {out_path_p} ({n_written} records)")
    return str(out_path_p)
