"""Source-aware contract inference. Neural dependencies are loaded on demand."""

VERSION = "2026.09.20-er1"
LABELS = ("T", "F", "U")

from .logic import solve, verify_certificate
from .provenance import withdraw, visible_evidence

__all__ = ["VERSION", "LABELS", "solve", "verify_certificate", "withdraw", "visible_evidence"]
