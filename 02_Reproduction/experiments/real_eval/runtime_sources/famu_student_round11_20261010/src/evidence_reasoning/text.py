"""Offset-preserving evidence windows, adaptive token budgets and numeric facts."""
from __future__ import annotations

import re
from datetime import date
from decimal import Decimal, InvalidOperation


def windows(text: str, width: int = 180, overlap: int = 32) -> list[dict]:
    if not 0 <= overlap < width:
        raise ValueError("Window overlap must be smaller than width")
    if not text:
        return []
    result, start = [], 0
    while start < len(text):
        end = min(len(text), start + width)
        result.append({"text": text[start:end], "start": start, "end": end})
        if end == len(text):
            break
        start = end - overlap
    return result


def adaptive_tokenize(fields: list[tuple[str, str]], tokenizer, max_length: int = 512,
                      priority: list[float] | None = None) -> tuple[dict, dict]:
    """Allocate unused field capacity before truncating; tokens are never decoded/re-encoded."""
    if not fields:
        raise ValueError("At least one input field is required")
    prefixes = [tokenizer.encode(name + "：", add_special_tokens=False) for name, _ in fields]
    values = [tokenizer.encode(text, add_special_tokens=False) for _, text in fields]
    separators = tokenizer.encode("\n", add_special_tokens=False)
    overhead = sum(map(len, prefixes)) + len(separators) * (len(fields) - 1) + tokenizer.num_special_tokens_to_add(False)
    available = max_length - overhead
    if available < 0:
        raise ValueError("Token budget cannot fit field headers")
    weights = priority or [1.] * len(fields)
    if len(weights) != len(fields) or any(w <= 0 for w in weights):
        raise ValueError("All field priorities must be positive")
    lengths = [0] * len(fields)
    for _ in range(min(available, sum(map(len, values)))):
        eligible = [i for i in range(len(fields)) if lengths[i] < len(values[i])]
        i = min(eligible, key=lambda j: (lengths[j] / weights[j], j))
        lengths[i] += 1
    ids = []
    for i, (header, tokens) in enumerate(zip(prefixes, values)):
        if i:
            ids.extend(separators)
        ids.extend(header)
        ids.extend(tokens[:lengths[i]])
    builder = getattr(tokenizer, "build_inputs_with_special_tokens", None)
    if callable(builder):
        ids = builder(ids)
    elif (tokenizer.num_special_tokens_to_add(False) == 2 and
          getattr(tokenizer, "cls_token_id", None) is not None and getattr(tokenizer, "sep_token_id", None) is not None):
        # Transformers 5 removed this method from the local BertTokenizer.
        # BGE/BERT's single-sequence layout is exactly [CLS] ids [SEP].
        ids = [tokenizer.cls_token_id, *ids, tokenizer.sep_token_id]
    else:
        raise ValueError("Unsupported tokenizer special-token layout")
    return {"input_ids": ids, "attention_mask": [1] * len(ids)}, {
        "tokens": len(ids), "capacity": max_length,
        "fields": [{"name": field[0], "before": len(value), "kept": kept,
                    "truncated": kept < len(value)} for field, value, kept in zip(fields, values, lengths)]}


def numeric_facts(text: str) -> dict:
    amounts = []
    for match in re.finditer(r"(?<![\d.])(\d+(?:,\d{3})*(?:\.\d+)?)\s*(万元|元)", text):
        try:
            value = Decimal(match[1].replace(",", "")) * (10000 if match[2] == "万元" else 1)
        except InvalidOperation:
            continue
        amounts.append({"raw": match[0], "value_yuan": str(value), "start": match.start(), "end": match.end()})
    dates, errors = [], []
    for match in re.finditer(r"(\d{4})[年/-](\d{1,2})[月/-](\d{1,2})日?", text):
        try:
            value = date(*map(int, match.groups())).isoformat()
        except ValueError:
            value = None
            errors.append({"type": "invalid_date", "raw": match[0]})
        dates.append({"raw": match[0], "value": value, "start": match.start(), "end": match.end()})
    return {"amounts": amounts, "dates": dates, "errors": errors}


def verify_quotes(quotes, text: str) -> dict:
    if not isinstance(quotes, list):
        return {"present": False, "verbatim": False, "checks": []}
    checks = [{"quote": q, "valid": isinstance(q, str) and bool(q.strip()) and q in text}
              for q in quotes]
    return {"present": bool(checks), "verbatim": bool(checks) and all(c["valid"] for c in checks), "checks": checks}
