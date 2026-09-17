#!/usr/bin/env python3
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""
HotpotQA dataset provider.

Source: hotpotqa/hotpot_qa (distractor config, train split, ~90.4k samples).
Multi-hop: always num_hops=2.
Context: concatenate all context paragraphs (10 paragraphs with titles).
Evidence: supporting_facts.title + supporting_facts.sent_id → char offsets.

Augmentation: HotpotQA paragraphs are short (~1000-2000 tokens per sample).
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


class HotpotQAProvider(BaseProvider):
    """HotpotQA dataset provider.

    Uses hotpotqa/hotpot_qa (distractor setting) with multi-hop questions.
    Each sample has 10 paragraphs (2 supporting + 8 distractors).
    Augments short contexts with distractor paragraphs from other samples.
    """

    def __init__(self, split: str = "train"):
        self._dataset = None
        self._split = split

    @property
    def name(self) -> str:
        return "hotpotqa"

    @staticmethod
    def _find_title_ranges(context: str) -> Dict[str, List[Tuple[int, int]]]:
        """Scan context for [Title] patterns and return char ranges per title.

        A title may appear multiple times (original paragraph + distractor with
        the same title). Returns ALL occurrences so evidence search can check each.

        Returns:
            Dict mapping title string to list of (start, end) character offsets.
        """
        import re
        title_ranges: Dict[str, List[Tuple[int, int]]] = {}
        pattern = re.compile(r'(?:^|\n)\[([^\]]+)\] ')
        matches = list(pattern.finditer(context))
        for i, m in enumerate(matches):
            title = m.group(1)
            block_start = m.start() if m.start() == 0 else m.start() + 1  # skip \n
            block_end = matches[i + 1].start() if i + 1 < len(matches) else len(context)
            if title not in title_ranges:
                title_ranges[title] = []
            title_ranges[title].append((block_start, block_end))
        return title_ranges

    def _load_dataset(self):
        if self._dataset is not None:
            return self._dataset
        from datasets import load_dataset
        logger.info(f"[HotpotQA] Loading hotpotqa/hotpot_qa (distractor, {self._split})...")
        try:
            self._dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split=self._split)
        except ValueError as e:
            if "Feature type" in str(e) and "not found" in str(e):
                logger.warning(f"[HotpotQA] Stale HF cache detected ({e}), re-downloading...")
                import tempfile
                cache_dir = tempfile.mkdtemp(prefix="hf_hotpotqa_")
                self._dataset = load_dataset(
                    "hotpotqa/hotpot_qa", "distractor", split=self._split,
                    cache_dir=cache_dir, trust_remote_code=True,
                )
            else:
                raise
        logger.info(f"[HotpotQA] Loaded {len(self._dataset)} samples")
        return self._dataset

    @staticmethod
    def _extract_paragraphs(sample: Dict) -> List[str]:
        """Extract all titled paragraphs from a sample as '[Title] text' strings."""
        titles = sample["context"]["title"]
        sentences_list = sample["context"]["sentences"]
        paragraphs = []
        for title, sentences in zip(titles, sentences_list):
            para_text = " ".join(sentences)
            if para_text.strip():
                paragraphs.append(f"[{title}] {para_text}")
        return paragraphs

    @staticmethod
    def _build_context_with_distractors(
        sample: Dict,
        distractor_pool: List[str],
        min_tokens: int,
        max_tokens: int,
        rng: random.Random,
        answer: str = "",
        target_tokens: int = 0,
    ) -> Tuple[str, List[str], Dict[str, List[str]]]:
        """Build context from sample paragraphs, augmenting with distractors if too short.

        Distractors are placed only before or after the original context block,
        never interleaved, to preserve evidence paragraph adjacency.

        Returns:
            (context_text, paragraph_texts, title_to_sentences)
        """
        titles = sample["context"]["title"]
        sentences_list = sample["context"]["sentences"]

        # Build original paragraphs
        original_parts = []
        title_to_sentences: Dict[str, List[str]] = {}

        for title, sentences in zip(titles, sentences_list):
            title_to_sentences[title] = sentences
            para_text = " ".join(sentences)
            if para_text.strip():
                original_parts.append(f"[{title}] {para_text}")

        # Check if we need augmentation
        context = "\n".join(original_parts)
        approx_tok = BaseProvider._approx_tokens(context)

        if approx_tok >= min_tokens and target_tokens <= 0:
            return context, original_parts, title_to_sentences

        # Need more text — add distractors
        if target_tokens > 0:
            upper_bound = min(2 * target_tokens - min_tokens, max_tokens)
            upper_bound = max(upper_bound, min_tokens)
            pad_target = rng.randint(min_tokens, upper_bound)
        else:
            pad_target = rng.randint(min_tokens, min(min_tokens + 3000, max_tokens))

        if approx_tok >= pad_target:
            return context, original_parts, title_to_sentences

        available_distractors = list(distractor_pool)
        rng.shuffle(available_distractors)

        added = []
        current_tok = approx_tok
        for para in available_distractors:
            if current_tok >= pad_target:
                break
            if BaseProvider._text_contains_answer(para, answer):
                continue
            added.append(para)
            current_tok += BaseProvider._approx_tokens(para) + 1  # +1 for \n

        # Insert distractors only before or after the entire original context
        # to avoid splitting evidence paragraph adjacency.
        rng.shuffle(added)
        split = rng.randint(0, len(added))
        all_parts = added[:split] + original_parts + added[split:]

        context = "\n".join(all_parts)
        return context, all_parts, title_to_sentences

    @staticmethod
    def _map_supporting_facts(
        sample: Dict,
        context: str,
        title_to_sentences: Dict[str, List[str]],
    ) -> Tuple[List[Tuple[int, int]], List[str], Dict[str, List[Tuple[int, int]]]]:
        """Map supporting_facts to character offsets in the context.

        Uses title-anchored search: each sentence is searched only within the
        character range of its parent title block, avoiding false matches in
        distractor paragraphs that may contain identical text.

        Returns:
            (evidence_spans, supporting_paragraph_texts, title_spans)
            where title_spans maps each supporting title to its evidence spans.
        """
        sup_titles = sample["supporting_facts"]["title"]
        sup_sent_ids = sample["supporting_facts"]["sent_id"]

        evidence_spans: List[Tuple[int, int]] = []
        supporting_texts: List[str] = []
        title_spans: Dict[str, List[Tuple[int, int]]] = {}
        context_lower = context.lower()
        title_ranges = HotpotQAProvider._find_title_ranges(context)

        seen_titles = set()
        for title, sent_id in zip(sup_titles, sup_sent_ids):
            sentences = title_to_sentences.get(title, [])
            if sent_id >= len(sentences):
                continue
            sentence = sentences[sent_id].strip()
            if not sentence:
                continue

            # Search within all blocks for this title (there may be multiple
            # if a distractor has the same [Title] prefix).
            sentence_lower = sentence.lower()
            title_block_ranges = title_ranges.get(title)
            if title_block_ranges is None:
                logger.warning(
                    "[HotpotQA] Title %r not found in context title ranges, skipping sentence", title
                )
                continue

            found = False
            for t_start, t_end in title_block_ranges:
                idx = context_lower.find(sentence_lower, t_start, t_end)
                if idx >= 0:
                    span = (idx, idx + len(sentence))
                    evidence_spans.append(span)
                    if title not in title_spans:
                        title_spans[title] = []
                    title_spans[title].append(span)
                    found = True
                    break

            if not found:
                logger.warning(
                    "[HotpotQA] Sentence from title %r (sent_id=%d) not found in "
                    "any of its %d title block(s), skipping",
                    title, sent_id, len(title_block_ranges),
                )

            if title not in seen_titles:
                seen_titles.add(title)
                para = " ".join(sentences)
                supporting_texts.append(para)

        return evidence_spans, supporting_texts, title_spans

    def _build_distractor_pool(
        self,
        dataset,
        indices: List[int],
        pool_size: int = 100000,
    ) -> List[str]:
        """Build a pool of distractor paragraphs from random samples.

        Collects non-supporting paragraphs from many samples to use as
        fillers for short contexts.
        """
        logger.info(f"[HotpotQA] Building distractor pool (up to {pool_size} paragraphs)...")
        pool: List[str] = []
        for idx in indices[:pool_size // 5]:  # ~10 paragraphs per sample, need pool_size/10
            sample = dataset[idx]
            supporting_titles = set(sample["supporting_facts"]["title"])
            titles = sample["context"]["title"]
            sentences_list = sample["context"]["sentences"]
            for title, sentences in zip(titles, sentences_list):
                if title in supporting_titles:
                    continue  # Only use non-supporting paragraphs
                para_text = " ".join(sentences).strip()
                if len(para_text.split()) >= 20:
                    pool.append(f"[{title}] {para_text}")
                if len(pool) >= pool_size:
                    break
            if len(pool) >= pool_size:
                break
        logger.info(f"[HotpotQA] Distractor pool: {len(pool)} paragraphs")
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

        # Build distractor pool from the dataset
        distractor_pool = self._build_distractor_pool(dataset, indices)

        samples: List[EvalSample] = []
        skipped_long = 0
        skipped_no_evidence = 0
        augmented = 0

        for idx in indices:
            if len(samples) >= num_samples:
                break

            sample = dataset[idx]
            question = sample.get("question", "")
            answer = sample.get("answer", "")
            if not question or not answer:
                continue

            context, paragraph_texts, title_to_sentences = self._build_context_with_distractors(
                sample, distractor_pool, min_tokens, max_tokens,
                rng=random.Random(idx),
                answer=answer,
                target_tokens=target_tokens,
            )

            approx_tok = self._approx_tokens(context)
            if approx_tok > max_tokens:
                skipped_long += 1
                continue

            evidence_spans, supporting_texts, title_spans = self._map_supporting_facts(
                sample, context, title_to_sentences
            )
            if not evidence_spans:
                skipped_no_evidence += 1
                continue

            # Build hop_evidence: HotpotQA always has 2 hops.
            # Group titles by order of first appearance in supporting_facts.
            sup_titles_raw = sample["supporting_facts"]["title"]
            unique_titles = list(dict.fromkeys(sup_titles_raw))  # preserves order
            hop_evidence: List[List[Tuple[int, int]]] = []
            for title in unique_titles:
                hop_evidence.append(title_spans.get(title, []))

            was_augmented = len(paragraph_texts) > len(sample["context"]["title"])
            if was_augmented:
                augmented += 1

            samples.append(EvalSample(
                sample_id=f"hotpotqa_{self._split}_{idx}",
                question=self._format_question(question),
                answer=answer,
                answers=[answer],
                context=context,
                evidence_spans=evidence_spans,
                supporting_paragraphs=supporting_texts,
                hop_evidence=hop_evidence,
                dataset="hotpotqa",
                num_hops=2,
                task_type="multi_hop_qa",
            ))

        logger.info(
            f"[HotpotQA] Selected {len(samples)} samples "
            f"(skipped: {skipped_long} long, "
            f"{skipped_no_evidence} no evidence, "
            f"{augmented} augmented with distractors)"
        )
        return samples
