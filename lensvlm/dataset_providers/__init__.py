# Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""Dataset providers for LensVLM evaluation data construction.

These are the exact providers used to build the paper's evaluation set: they load each source dataset, build a long-text
context with distractor augmentation, and locate evidence character spans.
prepare_data.py renders their output into compressed page images.
"""
from .base import BaseProvider, EvalSample
from .hotpotqa import HotpotQAProvider
from .nq import NQProvider
from .musique import MuSiQueProvider

PROVIDERS = {
    "hotpotqa": HotpotQAProvider,
    "nq": NQProvider,
    "musique": MuSiQueProvider,
}
