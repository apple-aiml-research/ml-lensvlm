#!/usr/bin/env python3
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""
Natural Questions dataset provider.

Source: lighteval/natural_questions_clean.

Each sample has:
  - document:     Full Wikipedia page text (the real source document).
  - long_answers: Gold paragraph(s) within the page.
  - short_answers: Exact answer string(s).

Approach:
    - Use the FULL Wikipedia page (document field) as context — realistic.
    - Find the long_answer paragraph within the document for evidence_spans.
    - If the page is too short (< min_tokens), prepend distractor text from
      OTHER NQ pages to reach the threshold (same half-and-half mix as
      triviaqa: 50% natural layout, 50% prepend distractors).
    - Single-hop QA (num_hops=1).
"""

import logging
import random
import re
from typing import List, Optional, Tuple

from .base import BaseProvider, EvalSample

logger = logging.getLogger(__name__)


def _normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace (including wiki formatting artifacts)."""
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _clean_long_answer(long_answer: str) -> str:
    """Clean a long_answer by removing table/list formatting artifacts.

    Many long_answers from NQ contain table-like structures with lots of
    newlines and short tab-separated fields. This cleans them into readable
    paragraph text.
    """
    if not long_answer:
        return long_answer

    lines = long_answer.split('\n')

    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if cleaned_lines and cleaned_lines[-1] != '':
                cleaned_lines.append('')
            continue
        # Short lines (<20 chars) not ending with sentence punct are likely
        # table cells/headers — skip them
        if len(stripped) < 20 and stripped[-1] not in '.!?)"\u201d':
            continue
        cleaned_lines.append(stripped)

    text = '\n'.join(cleaned_lines)
    while '\n\n\n' in text:
        text = text.replace('\n\n\n', '\n\n')

    return text.strip()


def _find_all_answer_occurrences(
    doc: str, answer: str, max_occurrences: int = 20,
) -> List[Tuple[int, int, str]]:
    """Find all occurrences of answer in doc, returning paragraph spans.

    For short answers (<=3 chars), requires word-boundary match to avoid
    false positives (e.g., "7" inside "1976"). Caps at max_occurrences
    to avoid pathological cases.
    """
    answer_lower = answer.lower()
    doc_lower = doc.lower()
    occurrences = []

    if len(answer) <= 3:
        for m in re.finditer(r'\b' + re.escape(answer_lower) + r'\b', doc_lower):
            if len(occurrences) >= max_occurrences:
                break
            pos = m.start()
            para_start, para_end, para_text = _expand_to_paragraph(doc, pos, len(answer))
            occurrences.append((para_start, para_end, para_text))
    else:
        search_start = 0
        while len(occurrences) < max_occurrences:
            pos = doc_lower.find(answer_lower, search_start)
            if pos < 0:
                break
            para_start, para_end, para_text = _expand_to_paragraph(doc, pos, len(answer))
            occurrences.append((para_start, para_end, para_text))
            search_start = pos + 1

    return occurrences


def _expand_to_paragraph(doc: str, pos: int, answer_len: int) -> Tuple[int, int, str]:
    """Expand a character position to surrounding paragraph boundaries."""
    para_start = doc.rfind('\n\n', 0, pos)
    para_start = para_start + 2 if para_start >= 0 else 0
    para_end = doc.find('\n\n', pos + answer_len)
    para_end = para_end if para_end >= 0 else len(doc)

    while para_start < para_end and doc[para_start] in ' \t\n':
        para_start += 1
    while para_end > para_start and doc[para_end - 1] in ' \t\n':
        para_end -= 1

    para_text = doc[para_start:para_end]

    if len(para_text.split()) < 10:
        para_start2 = doc.rfind('\n', 0, pos)
        para_start2 = para_start2 + 1 if para_start2 >= 0 else 0
        para_end2 = doc.find('\n', pos + answer_len)
        para_end2 = para_end2 if para_end2 >= 0 else len(doc)
        if (para_end2 - para_start2) > (para_end - para_start):
            para_start, para_end = para_start2, para_end2
            para_text = doc[para_start:para_end]

    return para_start, para_end, para_text


def _pick_best_occurrence(
    occurrences: List[Tuple[int, int, str]],
    question: str,
) -> Tuple[int, int]:
    """Pick the occurrence whose surrounding paragraph best answers the question.

    Scores each candidate by question-term overlap: the paragraph that shares
    the most content words with the question is most likely the gold evidence.
    """
    if len(occurrences) == 1:
        return occurrences[0][0], occurrences[0][1]

    q_words = set(re.sub(r'[^\w\s]', '', question.lower()).split())
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on',
                  'at', 'to', 'for', 'of', 'and', 'or', 'what', 'when',
                  'where', 'who', 'how', 'which', 'that', 'this', 'it', 'do',
                  'does', 'did', 'has', 'have', 'had', 'be', 'been', 'with'}
    q_content = q_words - stop_words

    best_idx = 0
    best_score = -1
    for i, (start, end, text) in enumerate(occurrences):
        p_words = set(re.sub(r'[^\w\s]', '', text.lower()).split())
        overlap = len(q_content & p_words)
        # Prefer longer paragraphs (more context) as tiebreaker
        score = overlap * 1000 + min(len(text.split()), 200)
        if score > best_score:
            best_score = score
            best_idx = i

    return occurrences[best_idx][0], occurrences[best_idx][1]


def _find_answer_paragraph(
    doc: str, answer: str, question: str = "",
) -> Optional[Tuple[int, int]]:
    """Find the best evidence paragraph in doc for the given answer + question.

    Three-tier approach:
    1. Single occurrence of answer → auto-use that paragraph
    2. Multiple occurrences → pick the one whose context best matches the question
    3. No occurrence → return None (sample will be skipped)

    Returns (start, end) character offsets, or None if answer not in doc.
    """
    occurrences = _find_all_answer_occurrences(doc, answer)
    if not occurrences:
        return None
    return _pick_best_occurrence(occurrences, question)


class NQProvider(BaseProvider):
    """NQ provider using the full Wikipedia document as context."""

    def __init__(self, split: str = "train"):
        self._dataset = None
        self._split = split

    @property
    def name(self) -> str:
        return "nq"

    def _load_dataset(self):
        if self._dataset is not None:
            return self._dataset
        from datasets import load_dataset
        logger.info(f"[NQ] Loading lighteval/natural_questions_clean ({self._split})...")
        self._dataset = load_dataset("lighteval/natural_questions_clean", split=self._split)
        logger.info(f"[NQ] Loaded {len(self._dataset)} samples")
        return self._dataset

    def _build_distractor_pool(
        self,
        dataset,
        indices: List[int],
        min_doc_words: int = 200,
        max_pool_size: int = 10000,
    ) -> List[Tuple[str, int]]:
        """Collect (doc_text, source_idx) from random samples for padding."""
        pool: List[Tuple[str, int]] = []
        for idx in indices:
            if len(pool) >= max_pool_size:
                break
            doc = dataset[idx].get("document", "")
            if doc and len(doc.split()) >= min_doc_words:
                pool.append((_normalize_whitespace(doc), idx))
        logger.info(f"[NQ] Built distractor pool: {len(pool)} docs")
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

        # Build distractor pool from random OTHER documents
        pool_indices = indices[:min(10000, len(indices))]
        distractor_pool = self._build_distractor_pool(dataset, pool_indices)

        samples: List[EvalSample] = []
        skipped_short = 0
        skipped_long = 0
        skipped_no_evidence = 0
        skipped_answer_not_in_doc = 0
        augmented_count = 0

        for idx in indices:
            if len(samples) >= num_samples:
                break

            raw = dataset[idx]
            question = raw.get("question", "").strip()
            short_answers = raw.get("short_answers", [])
            document = raw.get("document", "")

            if not question or not short_answers or not document:
                continue

            answer = short_answers[0].strip()
            if not answer:
                continue

            # Normalize the full document
            doc_clean = _normalize_whitespace(document)

            # Find the paragraph containing the short answer
            span = _find_answer_paragraph(doc_clean, answer, question)
            if span is None:
                skipped_answer_not_in_doc += 1
                continue
            gold_start, gold_end = span

            # Per-sample rng
            sample_rng = random.Random(idx + 54321)

            doc_tokens = self._approx_tokens(doc_clean)

            if doc_tokens > max_tokens:
                skipped_long += 1
                continue

            # Compute padding target for this sample.
            # When target_tokens > 0, pad aggressively toward the target average.
            # Sample uniformly in [min_tokens, upper_bound] so the mean ≈ target_tokens.
            if target_tokens > 0:
                upper_bound = min(2 * target_tokens - min_tokens, max_tokens)
                upper_bound = max(upper_bound, min_tokens)
                sample_target = sample_rng.randint(min_tokens, upper_bound)
            else:
                sample_target = min_tokens

            needs_padding = doc_tokens < sample_target
            if doc_tokens >= sample_target:
                # Document already meets or exceeds target — use directly
                context = doc_clean
                evidence_spans = [(gold_start, gold_end)]
            else:
                # Need padding
                shortfall = sample_target - doc_tokens
                use_natural = sample_rng.random() < 0.5

                # Pull distractors (excluding this sample)
                pool = list(distractor_pool)
                sample_rng.shuffle(pool)
                pad_parts: List[str] = []
                cur = 0
                for dtxt, didx in pool:
                    if cur >= shortfall:
                        break
                    if didx == idx:
                        continue
                    words = dtxt.split()
                    slice_text = " ".join(words[:10000])
                    if self._text_contains_answer(slice_text, answer, short_answers):
                        continue
                    pad_parts.append(slice_text)
                    cur += self._approx_tokens(slice_text)

                if not pad_parts:
                    skipped_short += 1
                    continue

                pad_text = "\n".join(pad_parts)
                augmented_count += 1

                if use_natural:
                    # Doc first, padding after
                    context = doc_clean + "\n" + pad_text
                    evidence_spans = [(gold_start, gold_end)]
                else:
                    # Padding first, then doc
                    pad_len = len(pad_text) + 1  # +1 for "\n"
                    context = pad_text + "\n" + doc_clean
                    evidence_spans = [(gold_start + pad_len, gold_end + pad_len)]

                total = self._approx_tokens(context)
                if total < min_tokens:
                    skipped_short += 1
                    continue
                    continue
                if total > max_tokens:
                    skipped_long += 1
                    continue

            gold_para = doc_clean[gold_start:gold_end]
            samples.append(EvalSample(
                sample_id=f"nq_{self._split}_{idx}",
                question=question,
                answer=answer,
                answers=short_answers,
                context=context,
                evidence_spans=evidence_spans,
                supporting_paragraphs=[gold_para],
                dataset="nq",
                num_hops=1,
                task_type="single_hop_qa",
            ))

        logger.info(
            f"[NQ] Selected {len(samples)} samples "
            f"(skipped: {skipped_short} short, {skipped_long} long, "
            f"{skipped_answer_not_in_doc} answer not in doc, "
            f"{skipped_no_evidence} no evidence, "
            f"{augmented_count} augmented with padding)"
        )
        return samples
