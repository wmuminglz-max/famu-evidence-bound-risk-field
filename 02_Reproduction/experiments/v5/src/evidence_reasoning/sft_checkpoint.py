"""Hash-verified adapter/optimizer/RNG checkpoints at committed updates."""
import hashlib
import json
from pathlib import Path
import torch
from .review_ir import write_json


def save_checkpoint(model, optimizer, folder, progress, protocol, device):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(folder / 'adapter', safe_serialization=True)
    def cpu(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {k: cpu(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(cpu(v) for v in value)
        return value
    state = {'optimizer': cpu(optimizer.state_dict()), 'cpu_rng': torch.get_rng_state(),
             'device_rng': getattr(torch, device).get_rng_state_all() if device != 'cpu' else [],
             'progress': progress}
    torch.save(state, folder / 'state.pt')
    hashes = {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in folder.rglob('*') if p.is_file()}
    # Written last: incomplete directories cannot be resumed.
    write_json(folder / 'checkpoint.json', {'protocol': protocol, 'files': hashes,
                                           'optimizer_steps': progress['updates']})


def load_checkpoint(model, optimizer, folder, protocol, device):
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    folder = Path(folder).resolve()
    manifest = json.loads((folder / 'checkpoint.json').read_text('utf-8'))
    if manifest['protocol'] != protocol:
        raise ValueError('Resume protocol or code differs from the committed checkpoint')
    for name, expected in manifest['files'].items():
        file = (folder / name).resolve()
        if not file.is_relative_to(folder) or hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise ValueError('Checkpoint integrity failure')
    weights = load_file(str(folder / 'adapter/adapter_model.safetensors'))
    if set(weights) != set(get_peft_model_state_dict(model)):
        raise ValueError('Checkpoint adapter structure differs')
    set_peft_model_state_dict(model, weights)
    state = torch.load(folder / 'state.pt', map_location='cpu', weights_only=True)
    optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['cpu_rng'])
    if device != 'cpu':
        getattr(torch, device).set_rng_state_all(state['device_rng'])
    return state['progress']
