#!/usr/bin/env python3
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""
MuSiQue dataset provider.

Source: dgslibisey/MuSiQue.
Multi-hop: num_hops = len(question_decomposition), typically 2-4.
Context: concatenate all paragraphs[i].paragraph_text with [Title] prefix.
Evidence: paragraphs[i].is_supporting == True → char offsets.
Filter: answerable == True only.

Augmentation: MuSiQue paragraphs are often short (~1000-2000 tokens).
To reach the 3k-12k token range, we pad with distractor paragraphs from
other samples (unrelated to the query).
"""

import logging
import os
import sys
import random
from typing import Dict, List, Optional, Tuple

from .base import BaseProvider, EvalSample


logger = logging.getLogger(__name__)


class MuSiQueProvider(BaseProvider):
    """MuSiQue dataset provider.

    Uses dgslibisey/MuSiQue for multi-hop QA (2-4 hops).
    Each sample has multiple paragraphs, some marked as supporting.
    Augments short contexts with distractor paragraphs from other samples.
    """

    def __init__(self, split: str = "train"):
        self._dataset = None
        self._split = split

    @property
    def name(self) -> str:
        return "musique"

    def _load_dataset(self):
        if self._dataset is not None:
            return self._dataset
        from datasets import load_dataset
        logger.info(f"[MuSiQue] Loading dgslibisey/MuSiQue ({self._split})...")
        try:
            self._dataset = load_dataset("dgslibisey/MuSiQue", split=self._split)
        except ValueError as e:
            if "Feature type" in str(e) and "not found" in str(e):
                logger.warning(f"[MuSiQue] Stale HF cache detected ({e}), re-downloading...")
                import tempfile
                cache_dir = tempfile.mkdtemp(prefix="hf_musique_")
                self._dataset = load_dataset(
                    "dgslibisey/MuSiQue", split=self._split,
                    cache_dir=cache_dir, trust_remote_code=True,
                )
            else:
                raise
        logger.info(f"[MuSiQue] Loaded {len(self._dataset)} samples")
        return self._dataset

    @staticmethod
    def _build_context_with_distractors(
        sample: Dict,
        distractor_pool: List[str],
        min_tokens: int,
        max_tokens: int,
        rng: random.Random,
        target_tokens: int = 0,
    ) -> Tuple[str, List[str], List[Tuple[int, int]], List[str], Dict[str, List[Tuple[int, int]]]]:
        """Build context from all paragraphs with [Title] prefix, augmenting if short.

        Distractors are placed only before or after the original context block,
        never interleaved, to preserve evidence paragraph adjacency.

        Returns:
            (context_text, all_paragraph_texts, evidence_spans,
             supporting_paragraph_texts, title_spans)
            where title_spans maps each supporting title to its evidence spans.
        """
        paragraphs = sample.get("paragraphs", [])

        original_parts = []
        evidence_titles = []
        supporting_texts: List[str] = []

        for doc_idx, para in enumerate(paragraphs):
            title = para.get("title", f"Document {doc_idx + 1}")
            text = para.get("paragraph_text", "").strip()
            if not text:
                continue
            is_supporting = para.get("is_supporting", False)

            full_para = f"[{title}] {text}"
            original_parts.append(full_para)

            if is_supporting:
                evidence_titles.append(title)
                supporting_texts.append(text)

        # Check if we need augmentation
        context = "\n".join(original_parts)
        approx_tok = BaseProvider._approx_tokens(context)

        all_parts = list(original_parts)
        n_prepend = 0
        if target_tokens > 0:
            upper_bound = min(2 * target_tokens - min_tokens, max_tokens)
            upper_bound = max(upper_bound, min_tokens)
            pad_target = rng.randint(min_tokens, upper_bound)
        else:
            pad_target = min_tokens

        if approx_tok < pad_target:
            available = list(distractor_pool)
            rng.shuffle(available)

            added = []
            current_tok = approx_tok
            answer_str = sample.get("answer", "")
            for para in available:
                if current_tok >= pad_target:
                    break
                if BaseProvider._text_contains_answer(para, answer_str):
                    continue
                added.append(para)
                current_tok += BaseProvider._approx_tokens(para) + 1

            # Place distractors only before or after the original context
            rng.shuffle(added)
            n_prepend = rng.randint(0, len(added))
            all_parts = added[:n_prepend] + original_parts + added[n_prepend:]

            context = "\n".join(all_parts)

        # Compute evidence spans — only match parts from the original context
        # (indices [n_prepend, n_prepend + len(original_parts))), not distractors
        # which may share the same [Title] prefix.
        orig_start_idx = n_prepend
        orig_end_idx = n_prepend + len(original_parts)
        evidence_spans: List[Tuple[int, int]] = []
        title_spans: Dict[str, List[Tuple[int, int]]] = {}
        offset = 0
        for part_idx, part in enumerate(all_parts):
            start = offset
            end = start + len(part)

            if orig_start_idx <= part_idx < orig_end_idx:
                for title in evidence_titles:
                    prefix = f"[{title}] "
                    if part.startswith(prefix):
                        text_start = start + len(prefix)
                        span = (text_start, end)
                        evidence_spans.append(span)
                        if title not in title_spans:
                            title_spans[title] = []
                        title_spans[title].append(span)
                        break

            offset = end + 1  # +1 for "\n"

        return context, all_parts, evidence_spans, supporting_texts, title_spans

    def _build_distractor_pool(
        self,
        dataset,
        indices: List[int],
        pool_size: int = 80000,
    ) -> List[str]:
        """Build a pool of distractor paragraphs from random samples."""
        logger.info(f"[MuSiQue] Building distractor pool (up to {pool_size} paragraphs)...")
        pool: List[str] = []
        for idx in indices[:pool_size // 3]:
            sample = dataset[idx]
            paragraphs = sample.get("paragraphs", [])
            for para in paragraphs:
                if para.get("is_supporting", False):
                    continue
                title = para.get("title", "")
                text = para.get("paragraph_text", "").strip()
                if text and len(text.split()) >= 20:
                    pool.append(f"[{title}] {text}")
                if len(pool) >= pool_size:
                    break
            if len(pool) >= pool_size:
                break
        logger.info(f"[MuSiQue] Distractor pool: {len(pool)} paragraphs")
        return pool

    def load_samples(
        self,
        num_samples: int,
        min_tokens: int = 3000,
        max_tokens: int = 12000,
        target_tokens: int = 0,
    ) -> List[EvalSample]:
        dataset = self._load_dataset()

        rng = random.Random(42)
        indices = list(range(len(dataset)))
        rng.shuffle(indices)

        # Build distractor pool
        distractor_pool = self._build_distractor_pool(dataset, indices)

        samples: List[EvalSample] = []
        skipped_long = 0
        skipped_not_answerable = 0
        skipped_no_evidence = 0
        augmented = 0

        for idx in indices:
            if len(samples) >= num_samples:
                break

            sample = dataset[idx]

            if not sample.get("answerable", True):
                skipped_not_answerable += 1
                continue

            question = sample.get("question", "")
            answer = sample.get("answer", "")
            if not question or not answer:
                continue

            context, all_texts, evidence_spans, supporting_texts, title_spans = (
                self._build_context_with_distractors(
                    sample, distractor_pool, min_tokens, max_tokens,
                    rng=random.Random(idx),
                    target_tokens=target_tokens,
                )
            )

            approx_tok = self._approx_tokens(context)
            if approx_tok > max_tokens:
                skipped_long += 1
                continue

            if not evidence_spans:
                skipped_no_evidence += 1
                continue

            # Track augmentation
            orig_para_count = len([
                p for p in sample.get("paragraphs", [])
                if p.get("paragraph_text", "").strip()
            ])
            if len(all_texts) > orig_para_count:
                augmented += 1

            decomposition = sample.get("question_decomposition", [])
            num_hops = max(len(decomposition), 2)

            # Build question_decomposition list of sub-question strings
            question_decomposition = [
                d.get("question", "") for d in decomposition
            ]

            # Build hop_evidence: map each supporting paragraph to its hop
            # via paragraph_support_idx in the decomposition entries.
            # Each decomposition entry has a paragraph_support_idx pointing
            # to the paragraph index that supports that hop.
            paragraphs_raw = sample.get("paragraphs", [])
            hop_evidence: List[List[Tuple[int, int]]] = [[] for _ in range(num_hops)]
            for hop_idx, decomp_entry in enumerate(decomposition):
                para_support_idx = decomp_entry.get("paragraph_support_idx")
                if para_support_idx is None or para_support_idx == "none":
                    continue
                try:
                    para_support_idx = int(para_support_idx)
                except (ValueError, TypeError):
                    continue
                if para_support_idx < 0 or para_support_idx >= len(paragraphs_raw):
                    continue
                para = paragraphs_raw[para_support_idx]
                title = para.get("title", "")
                if title in title_spans:
                    hop_evidence[hop_idx] = title_spans[title]

            samples.append(EvalSample(
                sample_id=f"musique_{self._split}_{idx}",
                question=self._format_question(question),
                answer=answer,
                answers=[answer],
                context=context,
                evidence_spans=evidence_spans,
                supporting_paragraphs=supporting_texts,
                hop_evidence=hop_evidence,
                question_decomposition=question_decomposition,
                dataset="musique",
                num_hops=num_hops,
                task_type="multi_hop_qa",
            ))

        logger.info(
            f"[MuSiQue] Selected {len(samples)} samples "
            f"(skipped: {skipped_long} long, "
            f"{skipped_not_answerable} not answerable, "
            f"{skipped_no_evidence} no evidence, "
            f"{augmented} augmented with distractors)"
        )
        return samples
