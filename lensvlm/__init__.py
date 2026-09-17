# Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""LensVLM: Selective Context Expansion for Compressed Visual Representation of Text."""

from .vision_config import load_model
from .rendering import render_pages, render_from_page_texts
from .evaluate import run_multi_turn_inference
