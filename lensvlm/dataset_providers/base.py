#!/usr/bin/env python3
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""
Base classes for dataset providers.

EvalSample is the universal output format — each provider loads
its dataset and outputs samples with full text context, evidence locations,
and hop information.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class EvalSample:
    """A dataset sample for evaluation.

    Fields:
        sample_id: Unique identifier (e.g. "hotpotqa_1234").
        question: The question to answer.
        answer: Primary answer text.
        answers: All acceptable answer variants.
        context: Full text context (3k-12k tokens, avg 8k).
        evidence_spans: Character offsets of answer-bearing text within context.
        supporting_paragraphs: Evidence paragraph texts (for multi-hop).
        hop_evidence: Per-hop evidence spans. hop_evidence[i] is a list of
            (start, end) char offsets for reasoning hop i.
        question_decomposition: Sub-questions for each hop (from MuSiQue;
            empty for other datasets).
        dataset: Source dataset name ("nq", "hotpotqa", "musique", "qasper", "helmet").
        num_hops: Number of reasoning hops (1=single-hop, 2+=multi-hop).
        task_type: Task category ("single_hop_qa", "multi_hop_qa").
    """
    sample_id: str
    question: str
    answer: str
    answers: List[str]
    context: str
    evidence_spans: List[Tuple[int, int]]
    supporting_paragraphs: List[str] = field(default_factory=list)
    hop_evidence: List[List[Tuple[int, int]]] = field(default_factory=list)
    question_decomposition: List[str] = field(default_factory=list)
    dataset: str = ""
    num_hops: int = 1
    task_type: str = "single_hop_qa"


class BaseProvider(ABC):
    """Abstract base class for dataset providers.

    Each provider loads samples from a HuggingFace dataset and outputs
    EvalSample objects with full text context in the 3k-12k token range.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier for logging."""

    @abstractmethod
    def load_samples(
        self,
        num_samples: int,
        min_tokens: int = 3000,
        max_tokens: int = 12000,
        target_tokens: int = 0,
    ) -> List[EvalSample]:
        """Load and return samples with full context.

        Args:
            num_samples: Maximum number of samples to return.
            min_tokens: Minimum token count for context.
            max_tokens: Maximum token count for context.
            target_tokens: If >0, pad each sample's context toward this target
                (sampled uniformly in [min_tokens, min(2*target - min_tokens, max_tokens)])
                so the overall mean is approximately target_tokens.

        Returns:
            List of EvalSample objects.
        """

    @staticmethod
    def _approx_tokens(text: str) -> int:
        """Approximate token count from word count (words * 1.3)."""
        return int(len(text.split()) * 1.3)

    @staticmethod
    def _find_all_spans(context: str, targets: List[str]) -> List[Tuple[int, int]]:
        """Find all character-level spans for target strings in context.

        Case-insensitive, finds every occurrence of each target.
        """
        spans: List[Tuple[int, int]] = []
        context_lower = context.lower()
        for target in targets:
            if not target or not target.strip():
                continue
            target_clean = target.strip()
            target_lower = target_clean.lower()
            search_start = 0
            while True:
                idx = context_lower.find(target_lower, search_start)
                if idx < 0:
                    break
                spans.append((idx, idx + len(target_clean)))
                search_start = idx + 1
        return spans

    @staticmethod
    def _text_contains_answer(
        text: str,
        answer: str,
        answers: Optional[List[str]] = None,
    ) -> bool:
        """Check if text contains any form of the answer. Used to filter distractors.

        For short answers (<=3 chars like "yes", "no", "42"), uses word-boundary
        matching to avoid false positives (e.g., "yes" inside "yesterday").
        For longer answers, uses simple substring matching.

        Args:
            text: The text to check for answer presence.
            answer: Primary answer string.
            answers: All acceptable answer variants.

        Returns:
            True if any answer variant is found in text.
        """
        import re as _re

        all_answers = list(answers) if answers else []
        if answer and answer not in all_answers:
            all_answers.append(answer)

        text_lower = text.lower()

        for ans in all_answers:
            if not ans or not ans.strip():
                continue
            ans_lower = ans.strip().lower()

            if len(ans_lower) <= 3:
                # Short answer: require word boundary match to avoid
                # false positives (e.g., "no" in "north", "yes" in "yesterday")
                if _re.search(r'\b' + _re.escape(ans_lower) + r'\b', text_lower):
                    return True
            else:
                if ans_lower in text_lower:
                    return True

        return False

    @staticmethod
    def _format_question(question: str) -> str:
        """Format question with proper capitalization and punctuation."""
        if not question:
            return question
        question = question.strip()
        if question:
            question = question[0].upper() + question[1:]
        if question and question[-1] not in '.?!':
            question = question + '?'
        return question
