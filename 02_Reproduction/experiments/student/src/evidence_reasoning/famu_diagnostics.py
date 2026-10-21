"""Completion CE attribution and predeclared full-answer likelihood scoring."""
import copy
import json
from .famu_llm import encode_example, prompt_text


def attributed_encoding(tokenizer, row, candidate=None, max_length=16384):
    x = copy.deepcopy(row)
    if candidate is not None:
        x['target']['value'] = candidate
    encoded = encode_example(tokenizer, x, max_length)
    if encoded['status'] != 'ready':
        raise ValueError('Diagnostic input exceeds the fixed context budget')
    prompt = prompt_text(tokenizer, x)
    response = json.dumps({'value': x['target']['value']}, separators=(',', ':'))
    text = prompt + response + tokenizer.eos_token
    mapped = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if mapped['input_ids'] != encoded['input_ids']:
        raise ValueError('Offset tokenizer differs from training tokenizer')
    start = len(prompt) + len('{"value":"')
    end = start + len(x['target']['value'])
    eos_start = len(prompt) + len(response)
    parts = []
    for index in range(encoded['prompt_tokens'], encoded['tokens']):
        a, b = mapped['offset_mapping'][index]
        if encoded['input_ids'][index] == tokenizer.eos_token_id and index == encoded['tokens'] - 1:
            kind = 'eos'
        elif a < end and b > start:
            kind = 'label_value'
        else:
            kind = 'json_format'
        parts.append({'kind': kind, 'target_position': index, 'logit_position': index - 1,
                      'token_id': encoded['input_ids'][index], 'offset': [a - len(prompt), b - len(prompt)]})
    if not any(p['kind'] == 'label_value' for p in parts) or parts[-1]['kind'] != 'eos':
        raise ValueError('Label or EOS attribution failed')
    return encoded, parts


def score_answer(model, encoded, parts, device):
    import torch
    ids = torch.tensor([encoded['input_ids']], device=device)
    positions = torch.tensor([p['target_position'] for p in parts], device=device)
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                       logits_to_keep=positions - 1).logits[0]
        ce = torch.nn.functional.cross_entropy(logits.float(), ids[0, positions], reduction='none').cpu().tolist()
    grouped = {}
    for kind in ('label_value', 'json_format', 'eos'):
        losses = [v for p, v in zip(parts, ce) if p['kind'] == kind]
        grouped[kind] = {'n': len(losses), 'sum_ce': sum(losses), 'mean_ce': sum(losses) / len(losses) if losses else None}
    return {'sum_log_probability': -sum(ce), 'mean_log_probability': -sum(ce) / len(ce),
            'completion_tokens': len(ce), 'ce_by_part': grouped,
            'causal_alignment': [{'target_position': p['target_position'], 'logit_position': p['logit_position'],
                                  'token_id': p['token_id'], 'part': p['kind'], 'ce': v} for p, v in zip(parts, ce)]}
