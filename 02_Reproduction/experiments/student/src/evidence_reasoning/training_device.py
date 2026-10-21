"""Explicit training device selection; unavailable accelerators never fall back."""
from __future__ import annotations

import torch


def resolve_device(name='cpu'):
    device = torch.device(name)
    if device.type == 'cpu':
        return device
    if device.type not in {'xpu', 'cuda'}:
        raise ValueError('Supported training devices: cpu, xpu, cuda')
    backend = getattr(torch, device.type)
    if not backend.is_available():
        raise RuntimeError(f'Requested {name}, but torch {torch.__version__} cannot access it')
    index = device.index if device.index is not None else backend.current_device()
    if not 0 <= index < backend.device_count():
        raise ValueError(f'Invalid {device.type} device index: {index}')
    return torch.device(device.type, index)


def synchronize(device):
    device = torch.device(device)
    if device.type in {'xpu', 'cuda'}:
        getattr(torch, device.type).synchronize(device)


def runtime_metadata(device):
    device = torch.device(device)
    result = {'device': str(device), 'torch': torch.__version__, 'precision': 'float32',
              'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()}
    if device.type in {'xpu', 'cuda'}:
        result['device_name'] = getattr(torch, device.type).get_device_name(device)
    return result
