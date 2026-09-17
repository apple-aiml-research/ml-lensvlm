# Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""
LLM-as-Judge Evaluator for QA tasks.

Uses a local VLM (via vLLM) or OpenAI API to evaluate answer correctness.
For retrieval/NIAH tasks (e.g., RULER), uses exact match heuristics.
"""
import re
import string
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Union
from collections import Counter
from abc import ABC, abstractmethod

from .vision_config import QWEN35_MM_PROCESSOR_KWARGS


# ===== Evaluation Prompts =====

JUDGE_PROMPT = '''You are an expert evaluator. Determine if the model's answer correctly answers the question based on the gold answers.

[QUESTION]
{question}
[/QUESTION]

[GOLD ANSWERS]
{gold_answers}
[/GOLD ANSWERS]

[MODEL ANSWER]
{model_answer}
[/MODEL ANSWER]

Evaluation criteria:
- The answer must convey the same core meaning as the gold answers
- Partial matches should be marked incorrect
- Additional correct information beyond gold answers is acceptable
- Empty or off-topic responses are incorrect
- Minor formatting differences (e.g., "10:30 pm" vs "10:30 p.m.") should be accepted

Respond with ONLY '[[YES]]' if the model answer is correct, or '[[NO]]' if incorrect.'''


@dataclass
class EvalResult:
    """Result of evaluating a single prediction."""
    sample_id: str
    question: str
    gold_answer: str
    prediction: str
    correct: bool
    score: float  # 1.0 for correct, 0.0 for incorrect
    eval_method: str  # "llm_judge", "exact_match", "f1"
    judge_response: Optional[str] = None  # Raw response from judge


# ===== Heuristic Evaluators =====

def normalize_text(s: str) -> str:
    """Normalize text for comparison."""
    s = s.lower()
    s = ''.join(c for c in s if c not in string.punctuation)
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return ' '.join(s.split())


def exact_match(pred: str, gold: str) -> bool:
    """Check if prediction exactly matches gold (normalized)."""
    return normalize_text(pred) == normalize_text(gold)


def substring_match(pred: str, gold: str) -> bool:
    """Check if gold answer is contained in prediction."""
    return normalize_text(gold) in normalize_text(pred)


def f1_score(pred: str, gold: str) -> float:
    """Compute token-level F1 score."""
    pred_toks = normalize_text(pred).split()
    gold_toks = normalize_text(gold).split()
    
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    
    common = Counter(pred_toks) & Counter(gold_toks)
    overlap = sum(common.values())
    
    if overlap == 0:
        return 0.0
    
    prec = overlap / len(pred_toks)
    rec = overlap / len(gold_toks)
    return 2 * prec * rec / (prec + rec)


def multiple_choice_match(pred: str, gold: str) -> bool:
    """Check if prediction matches gold for multiple choice (A/B/C/D)."""
    pred_clean = pred.strip().upper()
    gold_clean = gold.strip().upper()
    
    # Extract first letter option from prediction
    if len(gold_clean) == 1 and gold_clean in "ABCD":
        for char in pred_clean:
            if char in "ABCD":
                return char == gold_clean
        return False
    
    return exact_match(pred, gold)


# ===== Evaluator Classes =====

class BaseEvaluator(ABC):
    """Base class for evaluators."""
    
    @abstractmethod
    def evaluate(
        self,
        sample_id: str,
        question: str,
        gold_answer: Union[str, List[str]],
        prediction: str,
    ) -> EvalResult:
        """Evaluate a single prediction."""
        pass
    
    def evaluate_batch(
        self,
        samples: List[Dict],
        prediction_key: str = "prediction",
    ) -> Tuple[List[EvalResult], Dict]:
        """
        Evaluate a batch of samples.
        
        Args:
            samples: List of dicts with 'sample_id', 'question', 'answer', and prediction_key
            prediction_key: Key for model prediction in each sample
            
        Returns:
            (list of EvalResults, summary metrics dict)
        """
        results = []
        for sample in samples:
            result = self.evaluate(
                sample_id=sample.get("sample_id", ""),
                question=sample.get("question", ""),
                gold_answer=sample.get("answer", ""),
                prediction=sample.get(prediction_key, ""),
            )
            results.append(result)
        
        # Compute summary metrics
        correct = sum(1 for r in results if r.correct)
        total = len(results)
        metrics = {
            "accuracy": correct / total if total > 0 else 0.0,
            "correct": correct,
            "total": total,
        }
        
        return results, metrics


class HeuristicEvaluator(BaseEvaluator):
    """Evaluator using heuristic methods (EM, F1, substring)."""
    
    def __init__(self, method: str = "exact_match", f1_threshold: float = 0.5):
        """
        Args:
            method: One of "exact_match", "substring", "f1", "multiple_choice"
            f1_threshold: F1 threshold to consider correct (for method="f1")
        """
        self.method = method
        self.f1_threshold = f1_threshold
    
    def evaluate(
        self,
        sample_id: str,
        question: str,
        gold_answer: Union[str, List[str]],
        prediction: str,
    ) -> EvalResult:
        # Handle list of gold answers
        gold_answers = [gold_answer] if isinstance(gold_answer, str) else gold_answer
        
        # Try each gold answer
        correct = False
        best_score = 0.0
        
        for gold in gold_answers:
            if self.method == "exact_match":
                if exact_match(prediction, gold):
                    correct = True
                    best_score = 1.0
                    break
            elif self.method == "substring":
                if substring_match(prediction, gold):
                    correct = True
                    best_score = 1.0
                    break
            elif self.method == "f1":
                score = f1_score(prediction, gold)
                best_score = max(best_score, score)
                if score >= self.f1_threshold:
                    correct = True
            elif self.method == "multiple_choice":
                if multiple_choice_match(prediction, gold):
                    correct = True
                    best_score = 1.0
                    break
        
        return EvalResult(
            sample_id=sample_id,
            question=question,
            gold_answer=gold_answers[0] if len(gold_answers) == 1 else str(gold_answers),
            prediction=prediction,
            correct=correct,
            score=best_score if self.method == "f1" else (1.0 if correct else 0.0),
            eval_method=self.method,
        )


class LLMJudgeEvaluator(BaseEvaluator):
    """Evaluator using LLM as judge."""
    
    def __init__(
        self,
        model_name: str = "",
        tensor_parallel_size: int = 1,
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: Optional[float] = None,
        use_api: bool = False,
        api_model: str = "gpt-4o-mini",
        api_base_url: Optional[str] = None,
        max_judge_tokens: int = 2048,
    ):
        """
        Args:
            model_name: HuggingFace model name for local judge
            tensor_parallel_size: Number of GPUs
            max_model_len: Max context length
            use_api: Use OpenAI API instead of local model
            api_model: OpenAI model name (if use_api=True)
            api_base_url: Base URL for remote vLLM server (e.g. http://host:8000/v1).
                         When set, uses the remote server instead of loading a local model.
        """
        self.api_base_url = api_base_url
        self.max_judge_tokens = max_judge_tokens
        if api_base_url:
            self.use_api = True
            # When using remote vLLM, don't use the default OpenAI model name
            if api_model == "gpt-4o-mini":
                self.api_model = ""  # auto-detect from server
            else:
                self.api_model = api_model
        else:
            self.use_api = use_api
            self.api_model = api_model
        self._llm = None
        self._client = None

        self.gpu_memory_utilization = gpu_memory_utilization
        if not self.use_api:
            self.model_name = model_name
            self.tensor_parallel_size = tensor_parallel_size
            self.max_model_len = max_model_len

    def _init_local_model(self):
        """Lazy initialization of local vLLM model."""
        if self._llm is None:
            from vllm import LLM
            print(f"Loading judge model: {self.model_name}")
            target_util = self.gpu_memory_utilization
            if target_util is None:
                import os
                target_util = float(os.environ.get("OCR_GPU_MEMORY_UTILIZATION", "0.5"))
            try:
                import torch
                if torch.cuda.is_available():
                    free, total = torch.cuda.mem_get_info()
                    free_frac = (free / total) * 0.9
                    target_util = max(0.1, min(target_util, free_frac))
            except Exception:
                target_util = max(0.1, target_util)
            kwargs = {
                "model": self.model_name,
                "tensor_parallel_size": self.tensor_parallel_size,
                "dtype": "bfloat16",
                "trust_remote_code": True,
                "mm_processor_kwargs": QWEN35_MM_PROCESSOR_KWARGS,
            }
            if self.max_model_len:
                kwargs["max_model_len"] = self.max_model_len
            if getattr(self, "gpu_memory_utilization", None) is not None:
                kwargs["gpu_memory_utilization"] = target_util
            self._llm = LLM(**kwargs)
    
    def _init_api_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            import os
            from openai import OpenAI
            if self.api_base_url:
                self._client = OpenAI(base_url=self.api_base_url, api_key="dummy", timeout=300.0)
            else:
                self._client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), timeout=300.0)
    
    def _call_api(self, prompt: str) -> str:
        """Call API (OpenAI or remote vLLM server)."""
        self._init_api_client()
        model = self.api_model
        if self.api_base_url and not model:
            # Auto-detect model from remote vLLM server
            models = self._client.models.list()
            model = models.data[0].id
            self.api_model = model
        response = self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=self.max_judge_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return response.choices[0].message.content.strip()

    def _call_api_batch(self, prompts: List[str]) -> List[str]:
        """Batch call to remote vLLM server (concurrent HTTP requests)."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        self._init_api_client()
        model = self.api_model
        if self.api_base_url and not model:
            models = self._client.models.list()
            model = models.data[0].id
            self.api_model = model

        results = [None] * len(prompts)
        timeout_count = 0

        def _call_one(idx, prompt, retries=3):
            nonlocal timeout_count
            for attempt in range(retries):
                try:
                    response = self._client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=self.max_judge_tokens,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    )
                    return idx, response.choices[0].message.content.strip()
                except Exception as e:
                    if attempt < retries - 1:
                        import time
                        time.sleep(2 ** attempt)
                        continue
                    timeout_count += 1
                    print(f"  [WARN] Judge call failed after {retries} retries (idx={idx}): {e}")
                    return idx, ""

        with ThreadPoolExecutor(max_workers=min(64, len(prompts))) as executor:
            futures = [executor.submit(_call_one, i, p) for i, p in enumerate(prompts)]
            for future in as_completed(futures):
                idx, text = future.result()
                results[idx] = text

        if timeout_count > 0:
            print(f"  [WARN] {timeout_count}/{len(prompts)} judge calls timed out")

        return results
    
    def _call_local(self, prompts: List[str]) -> List[str]:
        """Call local vLLM model."""
        self._init_local_model()
        from vllm import SamplingParams
        
        messages_batch = [[{"role": "user", "content": p}] for p in prompts]
        sampling_params = SamplingParams(max_tokens=self.max_judge_tokens, temperature=0.0)
        outputs = self._llm.chat(messages_batch, sampling_params=sampling_params)
        
        return [out.outputs[0].text.strip() for out in outputs]
    
    def _parse_response(self, response: str) -> bool:
        """Parse judge response to boolean.

        For reasoning models that output long responses, only look at the
        final verdict. Check [[YES]]/[[NO]] tags first (preferred). For short
        responses (< 50 chars), fall back to plain yes/no matching.
        """
        response_lower = response.lower()
        has_yes_tag = "[[yes]]" in response_lower
        has_no_tag = "[[no]]" in response_lower
        # If explicit tags found, use them (last one wins for reasoning models)
        if has_yes_tag or has_no_tag:
            last_yes = response_lower.rfind("[[yes]]")
            last_no = response_lower.rfind("[[no]]")
            return last_yes > last_no
        # Short response (non-reasoning model): plain yes/no matching
        if len(response.strip()) < 50:
            return "yes" in response_lower
        # Long response without tags: check if it ends with yes/no
        last_line = response.strip().split("\n")[-1].lower()
        return "yes" in last_line and "no" not in last_line
    
    def evaluate(
        self,
        sample_id: str,
        question: str,
        gold_answer: Union[str, List[str]],
        prediction: str,
    ) -> EvalResult:
        # Format gold answers
        gold_answers = [gold_answer] if isinstance(gold_answer, str) else gold_answer
        gold_str = '\n'.join(f"- {ans}" for ans in gold_answers)
        
        prompt = JUDGE_PROMPT.format(
            question=question,
            gold_answers=gold_str,
            model_answer=prediction,
        )
        
        if self.use_api:
            response = self._call_api(prompt)
        else:
            response = self._call_local([prompt])[0]
        
        correct = self._parse_response(response)
        
        return EvalResult(
            sample_id=sample_id,
            question=question,
            gold_answer=gold_answers[0] if len(gold_answers) == 1 else str(gold_answers),
            prediction=prediction,
            correct=correct,
            score=1.0 if correct else 0.0,
            eval_method="llm_judge",
            judge_response=response,
        )
    
    def evaluate_batch(
        self,
        samples: List[Dict],
        prediction_key: str = "prediction",
    ) -> Tuple[List[EvalResult], Dict]:
        """Batch evaluation with vLLM for efficiency."""
        if self.use_api and not self.api_base_url:
            # Fall back to sequential for OpenAI API (rate limits)
            return super().evaluate_batch(samples, prediction_key)

        # Prepare all prompts
        prompts = []
        for sample in samples:
            gold_answer = sample.get("answer", "")
            gold_answers = [gold_answer] if isinstance(gold_answer, str) else gold_answer
            gold_str = '\n'.join(f"- {ans}" for ans in gold_answers)

            prompt = JUDGE_PROMPT.format(
                question=sample.get("question", ""),
                gold_answers=gold_str,
                model_answer=sample.get(prediction_key, ""),
            )
            prompts.append(prompt)

        # Batch inference
        if self.api_base_url:
            responses = self._call_api_batch(prompts)
        else:
            responses = self._call_local(prompts)
        
        # Parse results
        results = []
        for sample, response in zip(samples, responses):
            gold_answer = sample.get("answer", "")
            gold_answers = [gold_answer] if isinstance(gold_answer, str) else gold_answer
            correct = self._parse_response(response)
            
            results.append(EvalResult(
                sample_id=sample.get("sample_id", ""),
                question=sample.get("question", ""),
                gold_answer=gold_answers[0] if len(gold_answers) == 1 else str(gold_answers),
                prediction=sample.get(prediction_key, ""),
                correct=correct,
                score=1.0 if correct else 0.0,
                eval_method="llm_judge",
                judge_response=response,
            ))
        
        # Compute summary
        correct_count = sum(1 for r in results if r.correct)
        total = len(results)
        metrics = {
            "accuracy": correct_count / total if total > 0 else 0.0,
            "correct": correct_count,
            "total": total,
        }
        
        return results, metrics


class CompositeEvaluator(BaseEvaluator):
    """
    Evaluator that selects method based on task type.
    
    - RULER/NIAH tasks: exact match or substring match
    - Multiple choice: multiple choice match
    - QA tasks: LLM judge
    """
    
    def __init__(
        self,
        llm_judge_config: Optional[Dict] = None,
        task_to_method: Optional[Dict[str, str]] = None,
    ):
        """
        Args:
            llm_judge_config: Config dict for LLMJudgeEvaluator
            task_to_method: Mapping from task name to evaluation method
                          Methods: "exact_match", "substring", "f1", "multiple_choice", "llm_judge"
        """
        # Default task to method mapping
        self.task_to_method = task_to_method or {
            # RULER/NIAH - use exact match
            "ruler": "substring",
            "niah": "substring",
            "multihop_kv": "substring",
            # Code tasks - use LLM judge
            "code": "llm_judge",
            # QA tasks - use LLM judge
            "singlehop_qa": "llm_judge",
            "multihop_qa": "llm_judge",
            "qa": "llm_judge",
            # Multiple choice
            "multiple_choice": "multiple_choice",
        }
        
        # Initialize evaluators
        self.heuristic_evaluator = HeuristicEvaluator()
        
        llm_config = llm_judge_config or {}
        self._llm_judge = None
        self._llm_config = llm_config
    
    @property
    def llm_judge(self):
        """Lazy initialization of LLM judge."""
        if self._llm_judge is None:
            self._llm_judge = LLMJudgeEvaluator(**self._llm_config)
        return self._llm_judge
    
    def get_method_for_task(self, task: str) -> str:
        """Determine evaluation method based on task name."""
        task_lower = task.lower()
        
        # Check direct matches first
        if task_lower in self.task_to_method:
            return self.task_to_method[task_lower]
        
        # Check partial matches
        for key, method in self.task_to_method.items():
            if key in task_lower:
                return method
        
        # Default to LLM judge for unknown tasks
        return "llm_judge"
    
    def evaluate(
        self,
        sample_id: str,
        question: str,
        gold_answer: Union[str, List[str]],
        prediction: str,
        task: str = "",
    ) -> EvalResult:
        method = self.get_method_for_task(task)
        
        if method == "llm_judge":
            return self.llm_judge.evaluate(sample_id, question, gold_answer, prediction)
        else:
            self.heuristic_evaluator.method = method
            return self.heuristic_evaluator.evaluate(sample_id, question, gold_answer, prediction)
    
    def evaluate_batch(
        self,
        samples: List[Dict],
        prediction_key: str = "prediction",
    ) -> Tuple[List[EvalResult], Dict]:
        """
        Batch evaluation with task-aware method selection.
        Groups samples by method for efficient batch processing.
        """
        # Group samples by evaluation method
        method_groups: Dict[str, List[Tuple[int, Dict]]] = {}
        
        for idx, sample in enumerate(samples):
            task = sample.get("task", "")
            method = self.get_method_for_task(task)
            
            if method not in method_groups:
                method_groups[method] = []
            method_groups[method].append((idx, sample))
        
        # Evaluate each group
        all_results = [None] * len(samples)
        
        for method, indexed_samples in method_groups.items():
            indices, group_samples = zip(*indexed_samples)
            
            if method == "llm_judge":
                group_results, _ = self.llm_judge.evaluate_batch(list(group_samples), prediction_key)
            else:
                self.heuristic_evaluator.method = method
                group_results, _ = self.heuristic_evaluator.evaluate_batch(list(group_samples), prediction_key)
            
            for idx, result in zip(indices, group_results):
                all_results[idx] = result
        
        # Compute summary
        correct = sum(1 for r in all_results if r.correct)
        total = len(all_results)
        metrics = {
            "accuracy": correct / total if total > 0 else 0.0,
            "correct": correct,
            "total": total,
        }
        
        # Per-task metrics
        task_metrics = {}
        for sample, result in zip(samples, all_results):
            task = sample.get("task", "unknown")
            if task not in task_metrics:
                task_metrics[task] = {"correct": 0, "total": 0}
            task_metrics[task]["total"] += 1
            if result.correct:
                task_metrics[task]["correct"] += 1
        
        for task in task_metrics:
            t = task_metrics[task]
            t["accuracy"] = t["correct"] / t["total"] if t["total"] > 0 else 0.0
        
        metrics["per_task"] = task_metrics
        
        return all_results, metrics


def create_evaluator(
    eval_type: str = "composite",
    **kwargs
) -> BaseEvaluator:
    """
    Factory function to create evaluators.
    
    Args:
        eval_type: "heuristic", "llm_judge", or "composite"
        **kwargs: Arguments passed to the evaluator constructor
        
    Returns:
        BaseEvaluator instance
    """
    if eval_type == "heuristic":
        return HeuristicEvaluator(**kwargs)
    elif eval_type == "llm_judge":
        return LLMJudgeEvaluator(**kwargs)
    elif eval_type == "composite":
        return CompositeEvaluator(**kwargs)
    else:
        raise ValueError(f"Unknown evaluator type: {eval_type}")


# ===== Localization Evaluation =====

def evaluate_localization(results: List[Dict]) -> Dict:
    """
    Evaluate localization (Setting D) results.
    
    Computes accuracy (IoU >= 0.5) and average IoU for bbox localization.
    
    Args:
        results: List of dicts with 'gt_bbox' and 'predicted_bbox' keys
    
    Returns:
        Dict with n, accuracy, avg_iou
    """
    def _clamp(box):
        if not box or len(box) < 4:
            return None
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        x1 = max(0.0, min(1000.0, x1))
        y1 = max(0.0, min(1000.0, y1))
        x2 = max(0.0, min(1000.0, x2))
        y2 = max(0.0, min(1000.0, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _iou(a, b):
        a = _clamp(a)
        b = _clamp(b)
        if not a or not b:
            return 0.0
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter_area
        return inter_area / union if union > 0 else 0.0

    ious = []
    hits = 0
    for r in results:
        iou = _iou(r.get("gt_bbox"), r.get("predicted_bbox"))
        ious.append(iou)
        if iou >= 0.5:
            hits += 1

    n = len(results)
    return {
        "n": n,
        "accuracy": hits / n if n else 0,
        "avg_iou": sum(ious) / len(ious) if ious else 0,
    }


# ===== Results Printing =====

def print_metrics(metrics: Dict, render_config: Dict = None):
    """
    Print evaluation metrics in a clear table format.
    
    Args:
        metrics: Dict mapping setting (A/B/C/D) to metrics dict containing:
                 - accuracy, em, f1_score, correct, n (for A/B/C)
                 - precision, recall, f1 (for D - localization)
                 - avg_text_tokens, avg_visual_tokens, avg_oracle_tokens, avg_instruction_tokens
        render_config: Optional dict with width, height, dpi, margin_px info
    """    
    if render_config:
        print("\n--- Image Configuration ---")
        print(f"  Width x Height:  {render_config.get('width', '?')} x {render_config.get('height', '?')} pixels")
        if render_config.get('font_size'):
            print(f"  Font Size:       {render_config.get('font_size')} pt")
    
    setting_info = {
        "A": ("Text-only (baseline)", "Text"),
        "B": ("Image-only", "Images"),
        "C": ("Image + Oracle", "Images + Oracle text"),
        "D": ("Localization", "Images"),
        "E": ("Image + Predicted Oracle", "Images + Predicted text"),
    }
    
    active_settings = [s for s in ["A", "B", "C", "D", "E"] if s in metrics]
    if not active_settings:
        print("\n(No results to display)")
        return
    
    # Get baseline text tokens from Setting A for compression calculation
    baseline_tokens = metrics.get("A", {}).get('avg_text_tokens', 0) or \
                     metrics.get("A", {}).get('avg_context_tokens', 0)
    
    # ===== Token Consumption Table =====
    print("\n--- Token Consumption ---")
    print(f"{'Setting':<10} {'Context':<20} {'Oracle':<15} {'Instr':<12} {'Compress':<12}")
    print("-" * 75)
    
    for s in active_settings:
        m = metrics[s]
        text_tokens = m.get('avg_text_tokens', 0) or m.get('avg_context_tokens', 0)
        visual_tokens = m.get('avg_visual_tokens', 0)
        oracle_tokens = m.get('avg_oracle_tokens', 0)
        instr_tokens = m.get('avg_instruction_tokens', 0)
        
        # Context string (text for A, visual for B/C/D)
        if s == "A":
            ctx_str = f"{text_tokens:.0f} text" if text_tokens else "-"
        else:
            ctx_str = f"{visual_tokens:.0f} visual" if visual_tokens else "-"
        
        # Oracle tokens (only meaningful for C and E)
        oracle_str = f"+{oracle_tokens:.0f} text" if oracle_tokens > 0 else "-"
        
        # Instruction tokens
        instr_str = f"{instr_tokens:.0f}" if instr_tokens > 0 else "-"
        
        # Compression rate
        if s == "A":
            compress_str = "(baseline)"
        else:
            total_input = visual_tokens + oracle_tokens
            if total_input > 0 and baseline_tokens > 0:
                compress_str = f"{baseline_tokens / total_input:.2f}x"
            else:
                compress_str = "-"
        
        print(f"{s:<10} {ctx_str:<20} {oracle_str:<15} {instr_str:<12} {compress_str:<12}")
    
    # ===== Performance Table =====
    print("\n--- Performance ---")
    
    # Settings A/B/C/E use Judge Acc, EM, F1; Setting D uses P/R/F1
    # Check if any A/B/C/E settings have em/f1_score
    has_abce = any(s in metrics for s in ["A", "B", "C", "E"])
    has_d = "D" in metrics
    
    if has_abce:
        print(f"{'Setting':<10} {'Correct':<12} {'Judge Acc':<12} {'EM':<10} {'F1':<10}")
        print("-" * 60)
        
        for s in ["A", "B", "C", "E"]:
            if s not in metrics:
                continue
            m = metrics[s]
            n = m.get('n', m.get('total', 0))
            correct = m.get('correct', 0)
            accuracy = m.get('accuracy', 0)
            em = m.get('em', m.get('exact_match', 0))
            f1 = m.get('f1_score', m.get('f1', 0))
            
            correct_str = f"{correct}/{n}" if n else "-"
            print(f"{s:<10} {correct_str:<12} {accuracy:.3f}        {em:.3f}      {f1:.3f}")
    
    if has_d:
        if has_abce:
            print()  # Separator
        print(f"{'Setting':<10} {'Accuracy':<12} {'Avg IoU':<12}")
        print("-" * 50)
        
        m = metrics["D"]
        acc = m.get('accuracy', 0)
        iou = m.get('avg_iou', m.get('f1', 0))
        print(f"{'D':<10} {acc:.3f}        {iou:.3f}")
