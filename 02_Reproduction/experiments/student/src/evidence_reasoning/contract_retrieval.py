"""Question-conditioned retrieval over supplied text, with exact source offsets.

Retrieval scores express lexical relevance, never legal support or truth.
No reviewer labels, selected gold spans, or source identifiers enter ranking.
"""
from __future__ import annotations
import copy
import hashlib
import math
import re
from collections import Counter
from src.risk_grading import PROPOSITION_TYPES

INPUT_FIELDS = {"question", "proposition_type", "domain", "review_perspective", "sources",
                "withdrawn_sources", "required_source_kinds", "technical_status", "severity_basis"}
SOURCE_KINDS = {"contract", "performance_record", "law", "other"}
ARTICLE = re.compile(r"第([零〇一二三四五六七八九十百千万\d]+)条")
STOP_PHRASES = ("是否", "写清楚", "合同", "协议", "明确", "现有材料", "仅凭", "一起看", "本案", "足以", "条款")
# Retrieval paraphrases only; these do not encode legal conclusions.
ALIASES = (("社保", "社会保险", "参保", "缴费"), ("退费", "退款", "返还", "退还"),
           ("加班", "延长工作时间", "超时工资"), ("师资", "老师", "讲师", "教师"),
           ("停服", "停止服务", "终止服务"), ("调岗", "调动", "工作岗位"),
           ("签署", "签订", "签字"), ("解雇", "解除劳动合同", "辞退"))


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize(text):
    return re.sub(r"\s+", "", text).lower()


def terms(text):
    value = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text).lower()
    return Counter(value[i:i+n] for n in (2, 3) for i in range(max(0, len(value)-n+1)))


def validate_request(payload):
    if not isinstance(payload, dict) or set(payload) - INPUT_FIELDS:
        raise ValueError("Only observable question/source fields are accepted")
    x = copy.deepcopy(payload)
    if not isinstance(x.get("question"), str) or not x["question"].strip() or len(x["question"]) > 4000:
        raise ValueError("A nonempty question of at most 4000 characters is required")
    if x.get("proposition_type") not in PROPOSITION_TYPES:
        raise ValueError("Caller must specify the question type")
    for field in ("domain", "review_perspective"):
        if not isinstance(x.get(field), str) or len(x[field]) > 100:
            raise ValueError("Invalid observable context field")
    sources = x.get("sources")
    if not isinstance(sources, list) or len(sources) > 30:
        raise ValueError("Expected at most 30 supplied sources")
    ids = set()
    for source in sources:
        if not isinstance(source, dict) or set(source) - {"id", "text", "kind", "available", "sha256", "completeness"}:
            raise ValueError("Unsupported source field")
        if not isinstance(source.get("id"), str) or not source["id"] or source["id"] in ids:
            raise ValueError("Unique source identifiers are required")
        ids.add(source["id"])
        if not isinstance(source.get("text"), str) or source.get("kind") not in SOURCE_KINDS:
            raise ValueError("Source needs text and kind")
        if type(source.get("available", True)) is not bool:
            raise ValueError("Source availability must be boolean")
        if source.get("completeness", "unknown") not in {"full_supplied_text", "excerpt", "unknown"}:
            raise ValueError("Invalid supplied-text scope")
        if source.get("sha256") and source["sha256"] != sha(source["text"]):
            raise ValueError("Supplied text hash mismatch")
    if sum(len(s["text"]) for s in sources) > 500_000:
        raise ValueError("Supplied text exceeds the local retrieval budget")
    removed = x.setdefault("withdrawn_sources", [])
    if not isinstance(removed, list) or any(s not in ids for s in removed):
        raise ValueError("Unknown withdrawn source")
    required = x.setdefault("required_source_kinds", [])
    if not isinstance(required, list) or any(k not in SOURCE_KINDS for k in required):
        raise ValueError("Source requirements must be explicitly supplied kinds")
    if x.setdefault("technical_status", "ok") not in {"ok", "failed"}:
        raise ValueError("Invalid technical status")
    return x


def article_number(value):
    if value.isdigit():
        return int(value)
    digits = dict(zip("零〇一二三四五六七八九", [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    total = current = 0
    for char in value:
        if char in digits:
            current = digits[char]
        elif char in {"十", "百", "千", "万"}:
            total += (current or 1) * {"十": 10, "百": 100, "千": 1000, "万": 10000}[char]
            current = 0
    return total + current


def source_units(source, width=420):
    units = []
    article = None
    for line in re.finditer(r"[^\r\n]+", source["text"]):
        text = line[0]
        if not text.strip():
            continue
        heading = ARTICLE.match(text.lstrip())
        if heading:
            article = article_number(heading[1])
        # Preserve each short paragraph, including exception/negation words.
        # Long paragraph chunks remain marked incomplete unless all are retrieved.
        for offset in range(0, len(text), width):
            start, end = line.start()+offset, min(line.start()+offset+width, line.end())
            units.append({"source_id": source["id"], "source_kind": source["kind"],
                "start": start, "end": end, "text": source["text"][start:end],
                "paragraph_start": line.start(), "paragraph_end": line.end(),
                "article": article, "is_heading": bool(heading and offset == 0),
                "text_sha256": sha(source["text"][start:end]), "source_text_sha256": sha(source["text"]),
                "unit_id": source["id"] + ":" + str(start) + ":" + str(end)})
    return units


class ContractRetriever:
    def __init__(self, top_k=5, max_context_chars=7000, ranker=None):
        if top_k < 1 or max_context_chars < 100:
            raise ValueError("Invalid retrieval budget")
        self.top_k, self.max_context_chars = top_k, max_context_chars
        self.ranker = ranker

    def retrieve(self, payload):
        x = validate_request(payload)
        sources = [s for s in x["sources"] if s.get("available", True) and s["id"] not in x["withdrawn_sources"]]
        units, per_source = [], {}
        for source in sources:
            per_source[source["id"]] = source_units(source)
            units.extend(per_source[source["id"]])
        # Duplicate paragraphs are one ranking unit; source multiplicity is not evidence strength.
        unique, duplicates = {}, {}
        for unit in units:
            key = normalize(unit["text"])
            if key not in unique:
                unique[key] = unit
                duplicates[key] = []
            duplicates[key].append({k: unit[k] for k in ("source_id", "start", "end")})
        corpus = list(unique.values())
        query = x["question"]
        for phrase in STOP_PHRASES:
            query = query.replace(phrase, "")
        for aliases in ALIASES:
            if any(alias in x["question"] for alias in aliases):
                query += " " + " ".join(aliases)
        q = terms(query)
        counts = [terms(u["text"]) for u in corpus]
        frequency = Counter(token for c in counts for token in c)
        avg_len = sum(sum(c.values()) for c in counts) / max(1, len(counts))
        scored = []
        for unit, c in zip(corpus, counts):
            length = sum(c.values())
            score = 0.0
            for token in q.keys() & c.keys():
                idf = math.log(1 + (len(corpus)-frequency[token]+0.5)/(frequency[token]+0.5))
                score += idf*c[token]*2.2/(c[token]+1.2*(0.25+0.75*length/max(avg_len, 1)))
            if score > 0:
                scored.append((score, unit))
        scored.sort(key=lambda item: (-item[0], item[1]["source_id"], item[1]["start"]))
        if self.ranker is not None and corpus:
            lexical_scores = {u["unit_id"]: value for value, u in scored}
            values = self.ranker.rank(x["question"], corpus, lexical_scores)
            if len(values) != len(corpus) or any(not math.isfinite(v) for v in values):
                raise ValueError("Invalid semantic retrieval scores")
            scored = [(value, unit) for value, unit in zip(values, corpus) if value > 0]
            scored.sort(key=lambda item: (-item[0], item[1]["source_id"], item[1]["start"]))
        anchors = scored[:self.top_k]
        selected, reasons = {}, {}
        def add(unit, why):
            selected[unit["unit_id"]] = unit
            reasons.setdefault(unit["unit_id"], set()).add(why)
        for _, anchor in anchors:
            add(anchor, "query_match")
            group = per_source[anchor["source_id"]]
            position = next(i for i, u in enumerate(group) if u["unit_id"] == anchor["unit_id"])
            for unit in group[max(0, position-1):position+2]:
                add(unit, "adjacent_context")
            for unit in group:
                if unit["paragraph_start"] == anchor["paragraph_start"]:
                    add(unit, "whole_paragraph")
                if anchor["article"] is not None and unit["article"] == anchor["article"]:
                    add(unit, "same_article_scope")
        unresolved = []
        # Follow article references recursively, retaining all ambiguous targets.
        visited = set()
        for _ in range(8):
            pending = []
            for unit in list(selected.values()):
                for match in ARTICLE.finditer(unit["text"]):
                    number = article_number(match[1])
                    key = (unit["source_id"], number)
                    if key not in visited and number != unit["article"]:
                        visited.add(key)
                        pending.append(key)
            if not pending:
                break
            for source_id, number in pending:
                targets = [u for u in per_source[source_id] if u["article"] == number]
                heads = [u for u in targets if u["is_heading"]]
                if len(heads) != 1:
                    unresolved.append({"source_id": source_id, "article": number,
                                       "reason": "missing" if not heads else "ambiguous"})
                for unit in targets:
                    add(unit, "article_reference")
        else:
            unresolved.append({"reason": "reference_depth_budget"})
        # Budget rejection is explicit. It cannot be interpreted as absent evidence.
        ordered = sorted(selected.values(), key=lambda u: (u["source_id"], u["start"]))
        budget_exceeded = sum(len(u["text"]) for u in ordered) > self.max_context_chars
        anchor_ids = {u["unit_id"] for _, u in anchors}
        if budget_exceeded:
            priority = sorted(ordered, key=lambda u: (u["unit_id"] not in anchor_ids, u["source_id"], u["start"]))
            kept, size = [], 0
            for unit in priority:
                if size+len(unit["text"]) <= self.max_context_chars:
                    kept.append(unit)
                    size += len(unit["text"])
            ordered = sorted(kept, key=lambda u: (u["source_id"], u["start"]))
        score_map = {u["unit_id"]: score for score, u in anchors}
        evidence = [{**u, "selection_reasons": sorted(reasons[u["unit_id"]]),
                     "relevance_score": score_map.get(u["unit_id"], 0),
                     "duplicate_occurrences": duplicates[normalize(u["text"])],
                     "relation": "unclassified_candidate"} for u in ordered]
        return {"evidence": evidence, "anchor_ids": sorted(anchor_ids), "unresolved_references": unresolved,
                "context_budget_exceeded": budget_exceeded, "all_supplied_visible_text_scanned": True,
                "original_document_completeness_verified": False,
                "visible_source_count": len(sources), "visible_source_kinds": sorted({s["kind"] for s in sources}),
                "unique_paragraph_candidates": len(corpus), "ranking": "document_local_bm25_char_2_3" if self.ranker is None else self.ranker.name,
                "status": "candidate_evidence" if anchors else "no_retrieval_match",
                "missing_required_source_kinds": sorted(set(x["required_source_kinds"])-{s["kind"] for s in sources}),
                "full_text_scope_declared": bool(sources) and all(s.get("completeness") == "full_supplied_text" for s in sources),
                "retrieval_is_legal_support": False}


def prepare_features(payload, retrieval):
    """Only observable input; candidate annotations and audit fields are excluded."""
    x = validate_request(payload)
    texts = dict.fromkeys(e["text"] for e in retrieval["evidence"])
    sources = [s for s in x["sources"] if s.get("available", True) and s["id"] not in x["withdrawn_sources"]]
    return {"question": x["question"], "evidence": "\n".join(texts),
            "fulltext": "\n".join(dict.fromkeys(s["text"] for s in sources)),
            "domain": x["domain"], "review_perspective": x["review_perspective"],
            "proposition_type": x["proposition_type"],
            "has_performance_source": any(s["kind"] == "performance_record" for s in sources),
            "retrieved": bool(retrieval["anchor_ids"]), "context_incomplete": bool(retrieval["unresolved_references"] or retrieval["context_budget_exceeded"])}
