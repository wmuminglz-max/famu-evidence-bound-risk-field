"""Local evidence IDs with exact, independently checkable source offsets.

Never deduplicate equal text across different source occurrences. Hashes, global
IDs and annotation-selected offsets stay in the sidecar, not in model prompts.
"""
from __future__ import annotations
from .contract_retrieval import validate_request
from .review_ir import digest


def build_evidence_pack(request, spans=None, width=1400):
    request = validate_request(request)
    if width < 1:
        raise ValueError('Evidence width must be positive')
    sources = [s for s in request['sources'] if s.get('available', True)
               and s['id'] not in request.get('withdrawn_sources', [])]
    available = {s['id']: s for s in sources}
    local = {s['id']: f'S{i + 1}' for i, s in enumerate(sources)}
    selected = []
    if spans is None:
        for s in sources:
            # Exact contiguous slices preserve every supplied character.
            for start in range(0, len(s['text']), width):
                selected.append((s['id'], start, min(start + width, len(s['text']))))
    else:
        for span in spans:
            s = available.get(span['source_id'])
            start, end = span['start'], span['end']
            if s is None or type(start) is not int or type(end) is not int or not 0 <= start < end <= len(s['text']):
                raise ValueError('Evidence span is outside the available material')
            if 'quote' in span and span['quote'] != s['text'][start:end]:
                raise ValueError('Evidence quotation does not match source bytes')
            selected.append((s['id'], start, end))
    pieces, mapping = [], {}
    seen = set()
    for source_id, start, end in selected:
        if (source_id, start, end) in seen:
            continue  # Same occurrence, not identical text in another scope.
        seen.add((source_id, start, end))
        s = available[source_id]
        eid = f'E{len(pieces) + 1}'
        text = s['text'][start:end]
        pieces.append({'id': eid, 'source': local[source_id], 'kind': s['kind'], 'text': text})
        mapping[eid] = {'source_id': source_id, 'start': start, 'end': end,
                        'source_sha256': digest(s['text']), 'text_sha256': digest(text)}
    return {'pieces': pieces, 'mapping': mapping,
            'selection': 'full_supplied_text' if spans is None else 'reviewed_candidate_group',
            'material_complete': None, 'citation_sufficient': None}


def validate_citations(pack, request, citations):
    sources = {s['id']: s for s in validate_request(request)['sources']}
    pieces = {s['id']: s for s in pack['pieces']}
    errors = []
    for eid in citations:
        span = pack['mapping'].get(eid)
        if span is None or eid not in pieces:
            errors.append({'id': eid, 'reason': 'unknown_local_id'})
            continue
        s = sources.get(span['source_id'])
        if s is None or not s.get('available', True) or s['id'] in request.get('withdrawn_sources', []):
            errors.append({'id': eid, 'reason': 'source_unavailable'})
        elif (digest(s['text']) != span['source_sha256'] or
              s['text'][span['start']:span['end']] != pieces[eid]['text'] or
              digest(pieces[eid]['text']) != span['text_sha256']):
            errors.append({'id': eid, 'reason': 'source_or_span_changed'})
    return {'passed': not errors, 'errors': errors,
            'scope': 'reference_integrity_only', 'legal_support_verified': False}
