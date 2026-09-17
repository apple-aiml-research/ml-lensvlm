#!/usr/bin/env python3
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""
Standalone QA evaluation.

Evaluates a model on QA test data using multi-turn
tool-use inference (read_page), then runs an LLM judge for correctness.

Saves predictions.json separately so you can re-judge with different
models later using eval/judge.py (without re-running model inference).

See ``eval/evaluate.py`` (inference + judge in one pass) and ``eval/judge.py``
(re-judge saved predictions with a different judge) for the command-line entry
points.
"""

import argparse
import gc
import json
import os
import re
import sys
import time
from collections import defaultdict
from math import comb
from typing import Dict, List, Optional, Tuple

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# Set to False to silence the [QA_EVAL] progress logs (e.g. for the demo).
QA_EVAL_VERBOSE = True


def _qa_log(*args, **kwargs):
    if QA_EVAL_VERBOSE:
        print(*args, **kwargs)

# Patch tokenizers 0.22+ compatibility with vLLM (all_special_tokens_extended removed)
import transformers.tokenization_utils_base as _tub
_orig_getattr = _tub.PreTrainedTokenizerBase.__getattr__
def _patched_getattr(self, key):
    if key == "all_special_tokens_extended":
        return self.all_special_tokens
    return _orig_getattr(self, key)
_tub.PreTrainedTokenizerBase.__getattr__ = _patched_getattr

# Add project root and src/ to path

from .evaluator import (
    LLMJudgeEvaluator,
    exact_match,
    f1_score,
)
from .vision_config import QWEN35_MM_PROCESSOR_KWARGS


# ===== vLLM lifecycle =====

def _unload_vllm(llm):
    """Properly unload a vLLM model and free GPU memory."""
    import torch

    free_before = None
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            free_before, total = torch.cuda.mem_get_info()
            print(f"  GPU memory BEFORE unload: {(total - free_before) / 1e9:.2f} GB used")
        except RuntimeError as e:
            print(f"  [WARN] CUDA error: {e}")

    try:
        if hasattr(llm, "llm_engine"):
            engine = llm.llm_engine
            if hasattr(engine, "shutdown"):
                engine.shutdown()
            if hasattr(engine, "engine_core") and hasattr(engine.engine_core, "shutdown"):
                engine.engine_core.shutdown()
    except Exception as e:
        print(f"  [WARN] Engine shutdown error (non-fatal): {e}")

    del llm
    gc.collect()

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            time.sleep(2)
            gc.collect()
            torch.cuda.empty_cache()
            free_after, total = torch.cuda.mem_get_info()
            print(f"  GPU memory AFTER unload: {(total - free_after) / 1e9:.2f} GB used")
            if free_before is not None:
                print(f"  Memory freed: {(free_after - free_before) / 1e9:.2f} GB")
        except RuntimeError:
            pass


# ===== Model loading =====

# Set by main() before model loading so _load_vllm can compute max images
_current_test_samples = []

def _merge_lora_and_load(
    base_model: str,
    adapter_path: str,
    tp: int = 8,
    # If prompts exceed max_model_len, increase this (e.g. 65536, 131072, 262144).
    # Long-context evals with 60+ page documents need 131072+.
    max_model_len: int = 262144,    gpu_memory_utilization: float = 0.85,
) -> "LLM":
    """Merge LoRA adapter into base model and load with vLLM."""
    from peft import PeftModel
    from transformers import AutoModelForVision2Seq, AutoModelForImageTextToText, AutoProcessor
    import tempfile

    _qa_log(f"[QA_EVAL] Merging LoRA adapter: {adapter_path}")
    _qa_log(f"[QA_EVAL] Base model: {base_model}")

    # Load base model in bfloat16 for merging
    # GLM-4.1V registers as AutoModelForImageTextToText; Qwen3-VL uses AutoModelForVision2Seq
    base_model_lower = base_model.lower()
    if "glm" in base_model_lower or "glyph" in base_model_lower:
        auto_cls = AutoModelForImageTextToText
    else:
        auto_cls = AutoModelForVision2Seq

    base = auto_cls.from_pretrained(
        base_model,
        torch_dtype="auto",
        trust_remote_code=True,
        device_map="cpu",
    )
    peft_model = PeftModel.from_pretrained(base, adapter_path)
    merged = peft_model.merge_and_unload()

    # Save merged model to temp directory
    merge_dir = tempfile.mkdtemp(prefix="merged_")
    _qa_log(f"[QA_EVAL] Saving merged model to: {merge_dir}")
    merged.save_pretrained(merge_dir)

    # Also copy tokenizer/processor files from base model
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    processor.save_pretrained(merge_dir)

    del base, peft_model, merged
    gc.collect()

    # Load merged model in vLLM
    return _load_vllm(
        merge_dir, tp=tp, max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    ), merge_dir


def _load_vllm(
    model_path: str,
    tp: int = 8,
    max_model_len: int = 262144,
    gpu_memory_utilization: float = 0.85,
) -> "LLM":
    """Load a model with vLLM for inference."""
    from vllm import LLM

    _qa_log(f"[QA_EVAL] Loading model in vLLM: {model_path}")
    _qa_log(f"[QA_EVAL]   max_model_len={max_model_len}, "
          f"gpu_memory_utilization={gpu_memory_utilization}")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        allowed_local_media_path="/",
        mm_processor_kwargs=QWEN35_MM_PROCESSOR_KWARGS,
    )
    _qa_log("[QA_EVAL] vLLM model loaded successfully")
    return llm


# ===== Tool call parsing =====

def parse_tool_call(response: str) -> Optional[int]:
    """Parse read_page tool call from model response.

    Expected format: <tool_call>{"name": "read_page", "arguments": {"page": N}}</tool_call>

    Returns page number (1-indexed) or None if parsing fails.
    """
    pattern = r'<tool_call>\s*\{.*?"page"\s*:\s*(\d+).*?\}\s*</tool_call>'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return int(match.group(1))
    # Fallback: look for read_page with page number anywhere
    pattern2 = r'"page"\s*:\s*(\d+)'
    match2 = re.search(pattern2, response)
    if match2:
        return int(match2.group(1))
    return None


def extract_answer(response: str) -> str:
    """Extract the model's final answer from a multi-turn agentic rollout.

    Handles multi-turn tool-use responses correctly:
    1. If <answer>...</answer> tags exist, use the LAST one.
    2. Otherwise, isolate the last assistant segment (after the last
       </tool_response>) and strip <think> blocks from it. This covers both
       paired <think>...</think> and a dangling </think>, which is what the
       model actually emits when the chat template opens the block in the
       prompt.

    """
    import re

    # Strategy 1: explicit <answer> tags (last match wins)
    answer_matches = list(re.finditer(r'<answer>(.*?)</answer>', response, re.DOTALL))
    if answer_matches:
        return answer_matches[-1].group(1).strip() or ""

    # Strategy 2: find the model's last turn
    last_segment = response

    # Split on </tool_response> to get text after the last tool response
    if "</tool_response>" in response:
        last_segment = response.rsplit("</tool_response>", 1)[-1]

    # If "assistant" marker appears, take text after it
    parts = re.split(r'assistant\s*\n?', last_segment)
    if len(parts) > 1:
        last_segment = parts[-1]

    # Strip <think>...</think> blocks
    last_segment = re.sub(r'<think>.*?</think>', '', last_segment, flags=re.DOTALL).strip()

    # The chat template opens <think> in the *prompt*, so generation begins
    # inside the reasoning block and the model emits only the closing tag. That
    # leaves a dangling </think> the paired pattern above cannot match, and the
    # whole chain of thought would otherwise be returned as the answer.
    # Everything up to and including the last </think> is reasoning.
    if '</think>' in last_segment:
        last_segment = last_segment.rsplit('</think>', 1)[-1].strip()

    # Strip any remaining <tool_call> blocks
    last_segment = re.sub(r'<tool_call>.*?</tool_call>', '', last_segment, flags=re.DOTALL).strip()

    # Strip trailing incomplete tags
    last_segment = re.sub(r'<tool_call>.*$', '', last_segment, flags=re.DOTALL).strip()
    last_segment = re.sub(r'<think>.*$', '', last_segment, flags=re.DOTALL).strip()

    return last_segment if last_segment else ""


# ===== Multimodal message building =====

_MIN_IMAGE_DIM = 28  # GLM-4.1V patch factor requires both dims >= 28


def _ensure_min_image_size(img_path: str) -> str:
    """Pad image to minimum dimensions if needed (GLM-4.1V requires >= 28px).

    Returns the original path if OK, or a temp file path with the padded image.
    """
    from PIL import Image

    with Image.open(img_path) as img:
        w, h = img.size
        if w >= _MIN_IMAGE_DIM and h >= _MIN_IMAGE_DIM:
            return img_path

        new_w = max(w, _MIN_IMAGE_DIM)
        new_h = max(h, _MIN_IMAGE_DIM)
        padded = Image.new(img.mode, (new_w, new_h), (255, 255, 255))
        padded.paste(img, (0, 0))

        import tempfile
        fd, tmp_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        padded.save(tmp_path)
        return tmp_path


def _build_multimodal_content(text_content: str, image_paths: List[str]) -> list:
    """Convert text with <image> placeholders + image paths to OpenAI multimodal format.

    Replaces each <image> tag with an image_url content part using file:// URIs.
    Any remaining text is kept as text content parts.
    """
    import os

    parts = []
    img_idx = 0
    remaining = text_content

    while "<image>" in remaining and img_idx < len(image_paths):
        before, _, remaining = remaining.partition("<image>")
        if before.strip():
            parts.append({"type": "text", "text": before.strip()})
        img_path = image_paths[img_idx]
        # Pad small images to meet GLM-4.1V minimum dimension requirement
        img_path = _ensure_min_image_size(img_path)
        # Use absolute path with file:// URI
        abs_path = os.path.abspath(img_path)
        parts.append({
            "type": "image_url",
            "image_url": {"url": f"file://{abs_path}"},
        })
        img_idx += 1

    # Add any remaining text after last <image>
    if remaining.strip():
        parts.append({"type": "text", "text": remaining.strip()})

    # If no images or no <image> tags, return as plain text
    if not parts:
        return text_content

    return parts


def _build_tool_response(page_num: int, sample: dict, zoom_returns_image: bool):
    """Build tool response content for a read_page call.

    Returns a string (text mode) or list of content parts (image mode).
    """
    if zoom_returns_image:
        hires_images = sample.get("images_hires", [])
        if 1 <= page_num <= len(hires_images):
            abs_path = os.path.abspath(hires_images[page_num - 1])
            return [
                {"type": "text", "text": f"<tool_response>\nHigh-resolution image of Page {page_num}:\n"},
                {"type": "image_url", "image_url": {"url": f"file://{abs_path}"}},
                {"type": "text", "text": "\n</tool_response>"},
            ]
    # Default: text-based tool response
    page_texts = sample.get("page_texts", [])
    if 1 <= page_num <= len(page_texts):
        return (
            f"<tool_response>\nText content of Page {page_num}:\n"
            f"{page_texts[page_num - 1]}\n</tool_response>"
        )
    return None


# ===== Multi-turn inference =====

def _clean_response(response: str) -> str:
    """Clean model response: strip special tokens and fix tool call tags."""
    response = response.strip()
    # Strip Qwen special tokens that may appear
    for tok in ["<|im_end|>", "<|im_start|>", "<|endoftext|>"]:
        response = response.replace(tok, "")
    response = response.strip()
    # Re-append </tool_call> if missing (backward compat)
    if "<tool_call>" in response and "</tool_call>" not in response:
        response += "</tool_call>"
    return response


def _has_tool_call(response: str) -> bool:
    """Check if the response contains a tool_call."""
    return "<tool_call>" in response


def run_multi_turn_inference(
    llm,
    test_samples: List[dict],
    pass_k: int = 1,
    temperature: float = 0.0,
    max_turns: int = 6,
    zoom_returns_image: bool = False,
    return_trajectories: bool = False,
) -> List[dict]:
    """Run multi-turn inference with iterative tool execution.

    Each turn, the model either makes a read_page tool call (which we execute
    by providing real page text or high-res image) or gives a final answer.
    We loop until the model answers or we hit max_turns.

    Args:
        llm: vLLM model
        test_samples: List of test sample dicts with messages, images, page_texts, etc.
        pass_k: Number of independent trajectories per sample (n on Turn 1).
        temperature: Sampling temperature. Should be > 0 when pass_k > 1.
        max_turns: Maximum number of turns before forcing answer extraction.
        zoom_returns_image: If True, read_page returns high-res image from
            images_hires instead of text from page_texts.
        return_trajectories: If True, include the full conversation history
            in each returned prediction dict under the "trajectory" key.

    Returns:
        List of prediction dicts, one per trajectory (num_samples * pass_k).
    """
    from vllm import SamplingParams

    # Inference settings
    sampling_params_t1 = SamplingParams(
        max_tokens=2048,
        temperature=temperature,
        n=pass_k,
        stop=["</tool_call>"],
        include_stop_str_in_output=True,
    )
    sampling_params_cont = SamplingParams(
        max_tokens=2048,
        temperature=temperature,
        stop=["</tool_call>"],
        include_stop_str_in_output=True,
    )

    # ===== Turn 1: Send system + user message (images + question) =====
    _qa_log(f"[QA_EVAL] Turn 1: Sending {len(test_samples)} samples "
          f"(n={pass_k}, temp={temperature}, max_turns={max_turns})...")

    turn1_conversations = []
    for sample in test_samples:
        msgs = sample["messages"]
        system_msg = {"role": "system", "content": msgs[0]["content"]}
        user_content = _build_multimodal_content(
            msgs[1]["content"], sample.get("images", [])
        )
        user_msg = {"role": "user", "content": user_content}
        turn1_conversations.append([system_msg, user_msg])

    turn1_outputs = llm.chat(
        turn1_conversations,
        sampling_params=sampling_params_t1,
        add_generation_prompt=True,
    )

    # ===== Initialize per-trajectory state =====
    # Each trajectory tracks its own conversation, zoomed pages, and status.
    trajectories = []  # list of dicts

    for i, (sample, output) in enumerate(zip(test_samples, turn1_outputs)):
        page_texts = sample.get("page_texts", [])
        msgs = sample["messages"]

        # Build the multimodal user content once per sample
        user_content = _build_multimodal_content(
            msgs[1]["content"], sample.get("images", [])
        )

        for trial_idx, completion in enumerate(output.outputs):
            response = _clean_response(completion.text)

            # Start conversation history for this trajectory
            conv = [
                {"role": "system", "content": msgs[0]["content"]},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": response},
            ]

            traj = {
                "sample_idx": i,
                "trial": trial_idx,
                "conversation": conv,
                "zoomed_pages": [],
                "num_turns": 1,
                "status": "active",
                "final_answer": "",
                "generation_tokens": len(completion.token_ids),
            }

            # Process Turn 1 response
            page_num = parse_tool_call(response) if _has_tool_call(response) else None
            if page_num is not None:
                traj["zoomed_pages"].append(page_num)
                tool_resp = _build_tool_response(page_num, sample, zoom_returns_image)
                if tool_resp is not None:
                    conv.append({"role": "user", "content": tool_resp})
                else:
                    # Out-of-range page — complete with empty answer
                    traj["status"] = "complete"
            else:
                # No tool call — model answered directly
                traj["status"] = "complete"
                traj["final_answer"] = extract_answer(response)

            trajectories.append(traj)

    # ===== Turns 2..max_turns: iterate until all trajectories are complete =====
    for turn in range(2, max_turns + 1):
        active_indices = [i for i, t in enumerate(trajectories) if t["status"] == "active"]
        if not active_indices:
            break

        _qa_log(f"[QA_EVAL] Turn {turn}: {len(active_indices)} active trajectories...")

        active_convs = [trajectories[i]["conversation"] for i in active_indices]
        outputs = llm.chat(
            active_convs,
            sampling_params=sampling_params_cont,
            add_generation_prompt=True,
        )

        for j, output in enumerate(outputs):
            traj = trajectories[active_indices[j]]
            sample = test_samples[traj["sample_idx"]]
            page_texts = sample.get("page_texts", [])

            response = _clean_response(output.outputs[0].text)
            traj["conversation"].append({"role": "assistant", "content": response})
            traj["num_turns"] = turn
            traj["generation_tokens"] += len(output.outputs[0].token_ids)

            page_num = parse_tool_call(response) if _has_tool_call(response) else None
            if page_num is not None:
                traj["zoomed_pages"].append(page_num)
                tool_resp = _build_tool_response(page_num, sample, zoom_returns_image)
                if tool_resp is not None:
                    traj["conversation"].append({"role": "user", "content": tool_resp})
                else:
                    traj["status"] = "complete"
                    traj["final_answer"] = extract_answer(response)

                # If we've hit max turns, force completion
                if turn >= max_turns:
                    traj["status"] = "complete"
                    traj["final_answer"] = extract_answer(response)
            else:
                # No tool call — model gave an answer
                traj["status"] = "complete"
                traj["final_answer"] = extract_answer(response)

    # ===== Build prediction dicts =====
    predictions = []
    for traj in trajectories:
        sample = test_samples[traj["sample_idx"]]
        gt_pages = sample.get("gt_pages", [])
        zoomed = traj["zoomed_pages"]

        pred = {
            "sample_id": sample.get("id", f"sample_{traj['sample_idx']}"),
            "question": sample.get("question", ""),
            "answer": sample.get("answer", ""),
            "answers": sample.get("answers", [sample.get("answer", "")]),
            "dataset": sample.get("dataset", "unknown"),
            "predicted_answer": traj["final_answer"],
            "zoomed_pages": zoomed,
            "zoom_correct": sorted(zoomed) == sorted(gt_pages) if gt_pages else False,
            "zoom_hit": any(p in gt_pages for p in zoomed),
            "gt_pages": gt_pages,
            "num_turns": traj["num_turns"],
            "generation_tokens": traj["generation_tokens"],
            "trial": traj["trial"],
        }

        # Include full conversation history if requested
        if return_trajectories:
            pred["trajectory"] = traj["conversation"]

        # Token savings metric (only when lowres data has pixel counts)
        pixels_hires_all = sample.get("pixels_hires_total", 0)
        if pixels_hires_all > 0:
            pixels_lowres = sample.get("pixels_lowres_total", 0)
            # Sum hires pixels for zoomed pages only
            hires_images = sample.get("images_hires", [])
            pixels_hires_zoomed = 0
            for page_num in zoomed:
                if 1 <= page_num <= len(hires_images):
                    try:
                        from PIL import Image
                        with Image.open(hires_images[page_num - 1]) as img:
                            pixels_hires_zoomed += img.size[0] * img.size[1]
                    except Exception:
                        pass

            tokens_actual = (pixels_lowres + pixels_hires_zoomed) / 1024
            tokens_full_hires = pixels_hires_all / 1024
            token_savings = 1 - tokens_actual / tokens_full_hires if tokens_full_hires > 0 else 0

            pred["pixels_lowres"] = pixels_lowres
            pred["pixels_hires_zoomed"] = pixels_hires_zoomed
            pred["pixels_hires_all"] = pixels_hires_all
            pred["token_savings"] = token_savings

        predictions.append(pred)

    # Stats
    n = len(predictions)
    num_samples = len(test_samples)
    zoom_correct = sum(1 for p in predictions if p["zoom_correct"])
    zoom_hit = sum(1 for p in predictions if p["zoom_hit"])
    avg_turns = sum(p["num_turns"] for p in predictions) / n if n else 0
    _qa_log(f"[QA_EVAL] Trajectories: {n} ({num_samples} samples x {pass_k} trials)")
    _qa_log(f"[QA_EVAL] Zoom correct (exact match): {zoom_correct}/{n} "
          f"({zoom_correct/n*100:.1f}%)")
    _qa_log(f"[QA_EVAL] Zoom hit (any gt page): {zoom_hit}/{n} "
          f"({zoom_hit/n*100:.1f}%)")
    _qa_log(f"[QA_EVAL] Avg turns: {avg_turns:.1f}")
    avg_gen_tokens = sum(p["generation_tokens"] for p in predictions) / n if n else 0
    _qa_log(f"[QA_EVAL] Avg generation tokens: {avg_gen_tokens:.1f}")

    # Token savings summary (if lowres data present)
    savings_preds = [p for p in predictions if "token_savings" in p]
    if savings_preds:
        mean_savings = sum(p["token_savings"] for p in savings_preds) / len(savings_preds)
        _qa_log(f"[QA_EVAL] Token savings: {mean_savings*100:.1f}% "
              f"(mean over {len(savings_preds)} predictions)")

    return predictions


# ===== Metrics computation =====

def compute_heuristic_metrics(predictions: List[dict]) -> Tuple[Dict, List[dict]]:
    """Compute EM and F1 for each prediction.

    Returns (aggregate_metrics, updated_predictions).
    """
    total_em = 0
    total_f1 = 0.0
    per_dataset = defaultdict(lambda: {"em": 0, "f1": 0.0, "total": 0})

    for pred in predictions:
        answer = pred.get("predicted_answer", "")
        gold = pred.get("answer", "")
        gold_list = pred.get("answers", [gold])

        # EM: match any gold answer
        em = any(exact_match(answer, g) for g in gold_list)
        # F1: best across gold answers
        f1 = max((f1_score(answer, g) for g in gold_list), default=0.0)

        pred["em"] = em
        pred["f1"] = f1

        total_em += int(em)
        total_f1 += f1

        ds = pred.get("dataset", "unknown")
        per_dataset[ds]["em"] += int(em)
        per_dataset[ds]["f1"] += f1
        per_dataset[ds]["total"] += 1

    n = len(predictions)
    metrics = {
        "em": total_em / n if n else 0,
        "f1": total_f1 / n if n else 0,
    }

    for ds, dm in per_dataset.items():
        t = dm["total"]
        dm["em"] = dm["em"] / t if t else 0
        dm["f1"] = dm["f1"] / t if t else 0

    return metrics, predictions


def compute_pass_at_k(predictions: List[dict], max_k: int) -> Dict:
    """Compute unbiased pass@k estimator for k=1..max_k (Codex paper).

    Per sample with n trials and c correct:
      pass@k = 1 - C(n-c, k) / C(n, k)

    Groups predictions by sample_id and uses judge_correct for correctness.

    Returns dict with:
        "overall": {1: float, 2: float, ..., max_k: float}
        "per_dataset": {dataset_name: {1: float, ..., max_k: float}}
    """
    # Group by sample_id
    sample_trials = defaultdict(list)
    for pred in predictions:
        sample_trials[pred["sample_id"]].append(pred)

    # Also group by dataset
    dataset_samples = defaultdict(lambda: defaultdict(list))
    for pred in predictions:
        dataset_samples[pred["dataset"]][pred["sample_id"]].append(pred)

    def _pass_at_k_for_samples(grouped: Dict[str, List[dict]]) -> Dict[int, float]:
        """Compute pass@k for a group of samples."""
        results = {}
        for k in range(1, max_k + 1):
            pass_rates = []
            for sample_id, trials in grouped.items():
                n = len(trials)
                c = sum(1 for t in trials if t.get("judge_correct", False))
                if n < k:
                    # Not enough trials — skip this sample for this k
                    continue
                if n - c < k:
                    pass_rates.append(1.0)
                else:
                    pass_rates.append(1.0 - comb(n - c, k) / comb(n, k))
            results[k] = sum(pass_rates) / len(pass_rates) if pass_rates else 0.0
        return results

    overall = _pass_at_k_for_samples(sample_trials)
    per_dataset = {}
    for ds_name, ds_grouped in dataset_samples.items():
        per_dataset[ds_name] = _pass_at_k_for_samples(ds_grouped)

    return {"overall": overall, "per_dataset": per_dataset}


def run_judge_evaluation(
    predictions: List[dict],
    judge_model: str,
    tp: int = 1,
    judge_url: Optional[str] = None,
) -> Tuple[Dict, List[dict]]:
    """Run LLM judge evaluation on predictions.

    Returns (metrics_dict, updated_predictions).
    """
    if judge_url:
        _qa_log(f"[QA_EVAL] Running LLM judge via remote API: {judge_url}")
        judge = LLMJudgeEvaluator(
            api_base_url=judge_url,
            api_model=judge_model,
        )
    else:
        _qa_log(f"[QA_EVAL] Running LLM judge evaluation with {judge_model}...")
        judge = LLMJudgeEvaluator(
            model_name=judge_model,
            tensor_parallel_size=tp,
            max_model_len=4096,
            gpu_memory_utilization=0.5,
        )

    # Build samples for judge
    judge_samples = []
    for pred in predictions:
        judge_samples.append({
            "sample_id": pred["sample_id"],
            "question": pred["question"],
            "answer": pred.get("answers", [pred.get("answer", "")]),
            "prediction": pred.get("predicted_answer", ""),
        })

    results, metrics = judge.evaluate_batch(judge_samples, prediction_key="prediction")

    # Merge judge results back into predictions
    per_dataset = defaultdict(lambda: {"correct": 0, "total": 0})

    for pred, result in zip(predictions, results):
        pred["judge_correct"] = result.correct
        pred["judge_response"] = result.judge_response

        ds = pred.get("dataset", "unknown")
        per_dataset[ds]["total"] += 1
        if result.correct:
            per_dataset[ds]["correct"] += 1

    for ds, dm in per_dataset.items():
        dm["accuracy"] = dm["correct"] / dm["total"] if dm["total"] else 0

    metrics["per_dataset"] = dict(per_dataset)

    # Unload judge model
    if hasattr(judge, "_llm") and judge._llm is not None:
        _unload_vllm(judge._llm)
        judge._llm = None

    return metrics, predictions


# ===== Main =====

def main():
    parser = argparse.ArgumentParser(description="QA evaluation")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to LoRA adapter. If not set, evaluates base model only.")
    parser.add_argument("--test_data", type=str, required=True,
                        help="Path to the eval JSON")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--checkpoint_step", type=int, default=0)
    parser.add_argument("--judge_model", type=str, default="Qwen/Qwen3.5-397B-A17B-FP8")
    parser.add_argument("--judge_url", type=str, default=None,
                        help="Remote vLLM judge URL (e.g. http://host:8000/v1). If set, uses remote judge instead of local.")
    parser.add_argument("--tensor_parallel", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Limit number of test samples (for debugging)")
    parser.add_argument("--base_only", action="store_true",
                        help="Evaluate base model without adapter")
    parser.add_argument("--pass_k", type=int, default=8,
                        help="Number of trajectories per sample for pass@k evaluation (default: 8)")
    parser.add_argument("--pass_k_temperature", type=float, default=0.7,
                        help="Temperature when pass_k > 1 (default: 0.7)")
    parser.add_argument("--max_turns", type=int, default=6,
                        help="Max turns per trajectory (default: 6)")
    parser.add_argument("--skip_judge", action="store_true",
                        help="Skip LLM judge. Only run inference and save predictions. Use eval/judge.py later.")
    parser.add_argument("--zoom_returns_image", action="store_true",
                        help="Tool read_page returns high-res image instead of text (requires images_hires in data)")
    parser.add_argument("--max_model_len", type=int, default=262144,
                        help="vLLM max model context length (default: 262144 for Qwen3.5 native context)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85,
                        help="vLLM GPU memory utilization (default: 0.85)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load test data
    _qa_log(f"[QA_EVAL] Loading test data: {args.test_data}")
    with open(args.test_data) as f:
        test_samples = json.load(f)

    # Make image paths absolute (relative to test data directory)
    test_data_dir = os.path.dirname(os.path.abspath(args.test_data))
    for sample in test_samples:
        if "images" in sample:
            sample["images"] = [
                os.path.join(test_data_dir, p) if not os.path.isabs(p) else p
                for p in sample["images"]
            ]
        if "images_hires" in sample:
            sample["images_hires"] = [
                os.path.join(test_data_dir, p) if not os.path.isabs(p) else p
                for p in sample["images_hires"]
            ]

    if args.num_samples:
        test_samples = test_samples[:args.num_samples]
    _qa_log(f"[QA_EVAL] Test samples: {len(test_samples)}")

    # Dataset distribution
    ds_dist = defaultdict(int)
    for s in test_samples:
        ds_dist[s.get("dataset", "unknown")] += 1
    _qa_log(f"[QA_EVAL] Dataset distribution: {dict(ds_dist)}")

    # Make test samples available to _load_vllm for computing max images
    global _current_test_samples
    _current_test_samples = test_samples

    # Load model
    merge_dir = None
    if args.adapter_path and not args.base_only:
        llm, merge_dir = _merge_lora_and_load(
            args.base_model, args.adapter_path, tp=args.tensor_parallel,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    else:
        llm = _load_vllm(
            args.base_model, tp=args.tensor_parallel,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

    # Run multi-turn inference
    temperature = args.pass_k_temperature if args.pass_k > 1 else 0.0
    predictions = run_multi_turn_inference(
        llm, test_samples, pass_k=args.pass_k, temperature=temperature,
        max_turns=args.max_turns, zoom_returns_image=args.zoom_returns_image,
    )

    # Unload VLM
    _qa_log("[QA_EVAL] Unloading VLM...")
    _unload_vllm(llm)

    # Clean up merged model directory
    if merge_dir and os.path.exists(merge_dir):
        import shutil
        shutil.rmtree(merge_dir, ignore_errors=True)
        _qa_log(f"[QA_EVAL] Cleaned up merged model: {merge_dir}")

    # Compute heuristic metrics (EM, F1)
    heuristic_metrics, predictions = compute_heuristic_metrics(predictions)
    _qa_log(f"[QA_EVAL] Heuristic metrics - EM: {heuristic_metrics['em']:.4f}, "
          f"F1: {heuristic_metrics['f1']:.4f}")

    # Save predictions (before judging) for later re-judging with eval/judge.py
    predictions_path = os.path.join(args.output_dir, "predictions.json")
    with open(predictions_path, "w") as f:
        json.dump([
            {
                "sample_id": p["sample_id"],
                "question": p["question"],
                "answer": p["answer"],
                "answers": p.get("answers", [p.get("answer", "")]),
                "dataset": p["dataset"],
                "predicted_answer": p.get("predicted_answer", ""),
                "zoomed_pages": p.get("zoomed_pages", []),
                "zoom_correct": p.get("zoom_correct", False),
                "zoom_hit": p.get("zoom_hit", False),
                "gt_pages": p.get("gt_pages", []),
                "num_turns": p.get("num_turns", 1),
                "num_tool_calls": len(p.get("zoomed_pages", [])),
                "generation_tokens": p.get("generation_tokens", 0),
                "em": p.get("em", False),
                "f1": p.get("f1", 0.0),
                "trial": p.get("trial", 0),
                **({"token_savings": p["token_savings"],
                    "pixels_lowres": p["pixels_lowres"],
                    "pixels_hires_zoomed": p["pixels_hires_zoomed"],
                    "pixels_hires_all": p["pixels_hires_all"]}
                   if "token_savings" in p else {}),
            }
            for p in predictions
        ], f, indent=2)
    _qa_log(f"[QA_EVAL] Predictions saved to: {predictions_path}")

    if args.skip_judge:
        _qa_log("[QA_EVAL] Skipping LLM judge (--skip_judge). Use eval/judge.py to score later.")
        return 0

    # Run LLM judge
    judge_metrics, predictions = run_judge_evaluation(
        predictions, args.judge_model, tp=1, judge_url=args.judge_url
    )
    _qa_log(f"[QA_EVAL] Judge accuracy: {judge_metrics['accuracy']:.4f} "
          f"({judge_metrics['correct']}/{judge_metrics['total']})")

    # Aggregate metrics
    n = len(predictions)
    zoom_correct = sum(1 for p in predictions if p["zoom_correct"])
    zoom_hit = sum(1 for p in predictions if p["zoom_hit"])
    avg_turns = sum(p["num_turns"] for p in predictions) / n if n else 0
    avg_tool_calls = sum(len(p.get("zoomed_pages", [])) for p in predictions) / n if n else 0
    avg_gen_tokens = sum(p.get("generation_tokens", 0) for p in predictions) / n if n else 0

    # Per-dataset breakdown combining judge + heuristic
    per_dataset_combined = {}
    for ds_name in ds_dist:
        ds_preds = [p for p in predictions if p.get("dataset") == ds_name]
        ds_n = len(ds_preds)
        ds_correct = sum(1 for p in ds_preds if p.get("judge_correct", False))
        ds_em = sum(1 for p in ds_preds if p.get("em", False))
        ds_f1 = sum(p.get("f1", 0.0) for p in ds_preds)
        ds_zoom_correct = sum(1 for p in ds_preds if p.get("zoom_correct", False))
        ds_zoom_hit = sum(1 for p in ds_preds if p.get("zoom_hit", False))
        ds_tool_calls = sum(len(p.get("zoomed_pages", [])) for p in ds_preds)
        ds_gen_tokens = sum(p.get("generation_tokens", 0) for p in ds_preds)
        per_dataset_combined[ds_name] = {
            "accuracy": ds_correct / ds_n if ds_n else 0,
            "em": ds_em / ds_n if ds_n else 0,
            "f1": ds_f1 / ds_n if ds_n else 0,
            "correct": ds_correct,
            "total": ds_n,
            "zoom_correct_rate": ds_zoom_correct / ds_n if ds_n else 0,
            "zoom_hit_rate": ds_zoom_hit / ds_n if ds_n else 0,
            "avg_tool_calls": ds_tool_calls / ds_n if ds_n else 0,
            "avg_generation_tokens": ds_gen_tokens / ds_n if ds_n else 0,
        }

    # Unweighted accuracy: mean of per-dataset accuracies
    per_ds_accs = [dm["accuracy"] for dm in per_dataset_combined.values()]
    unweighted_accuracy = sum(per_ds_accs) / len(per_ds_accs) if per_ds_accs else 0

    metrics = {
        "accuracy": judge_metrics["accuracy"],
        "accuracy_unweighted": unweighted_accuracy,
        "em": heuristic_metrics["em"],
        "f1": heuristic_metrics["f1"],
        "correct": judge_metrics["correct"],
        "total": n,
        "zoom_correct_rate": zoom_correct / n if n else 0,
        "zoom_hit_rate": zoom_hit / n if n else 0,
        "avg_turns": avg_turns,
        "avg_tool_calls": avg_tool_calls,
        "avg_generation_tokens": avg_gen_tokens,
        "per_dataset": per_dataset_combined,
    }

    # Token savings (if lowres data)
    savings_preds = [p for p in predictions if "token_savings" in p]
    if savings_preds:
        metrics["mean_token_savings"] = sum(p["token_savings"] for p in savings_preds) / len(savings_preds)

    # Compute pass@k if pass_k > 1
    if args.pass_k > 1:
        pass_at_k_results = compute_pass_at_k(predictions, args.pass_k)
        metrics["pass_at_k"] = pass_at_k_results["overall"]
        metrics["pass_at_k_per_dataset"] = pass_at_k_results["per_dataset"]

    # Save results
    results = {
        "checkpoint_step": args.checkpoint_step,
        "base_model": args.base_model,
        "adapter_path": args.adapter_path,
        "judge_model": args.judge_model,
        "pass_k": args.pass_k,
        "max_turns": args.max_turns,
        "metrics": metrics,
        "predictions": [
            {
                "sample_id": p["sample_id"],
                "question": p["question"],
                "answer": p["answer"],
                "dataset": p["dataset"],
                "predicted_answer": p.get("predicted_answer", ""),
                "zoomed_pages": p.get("zoomed_pages", []),
                "zoom_correct": p.get("zoom_correct", False),
                "zoom_hit": p.get("zoom_hit", False),
                "gt_pages": p.get("gt_pages", []),
                "num_turns": p.get("num_turns", 1),
                "num_tool_calls": len(p.get("zoomed_pages", [])),
                "generation_tokens": p.get("generation_tokens", 0),
                "em": p.get("em", False),
                "f1": p.get("f1", 0.0),
                "judge_correct": p.get("judge_correct", False),
                "judge_response": p.get("judge_response", ""),
                "trial": p.get("trial", 0),
                **({"token_savings": p["token_savings"]}
                   if "token_savings" in p else {}),
            }
            for p in predictions
        ],
    }

    output_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    _qa_log(f"[QA_EVAL] Results saved to: {output_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"QA EVALUATION SUMMARY (Step {args.checkpoint_step})")
    print(f"{'='*60}")
    print(f"  Accuracy (Judge): {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
    print(f"  Acc Unweighted:   {metrics['accuracy_unweighted']:.4f} (mean of per-dataset)")
    print(f"  Exact Match:      {metrics['em']:.4f}")
    print(f"  F1 Score:         {metrics['f1']:.4f}")
    print(f"  Zoom Correct:     {metrics['zoom_correct_rate']*100:.1f}%")
    print(f"  Zoom Hit:         {metrics['zoom_hit_rate']*100:.1f}%")
    print(f"  Avg Turns:        {metrics['avg_turns']:.1f}")
    print(f"  Avg Tool Calls:   {metrics['avg_tool_calls']:.2f}")
    print(f"  Avg Gen Tokens:   {metrics['avg_generation_tokens']:.1f}")
    if "mean_token_savings" in metrics:
        print(f"  Token Savings:    {metrics['mean_token_savings']*100:.1f}%")
    print(f"\n  Per-Dataset:")
    for ds_name in sorted(per_dataset_combined):
        dm = per_dataset_combined[ds_name]
        print(f"    {ds_name:>12}: Acc={dm['accuracy']:.3f}  EM={dm['em']:.3f}  "
              f"F1={dm['f1']:.3f}  ZoomOK={dm['zoom_correct_rate']:.3f}  ZoomHit={dm['zoom_hit_rate']:.3f}  "
              f"ToolCalls={dm['avg_tool_calls']:.2f}  GenTok={dm['avg_generation_tokens']:.1f}  ({dm['correct']}/{dm['total']})")

    # Print pass@k summary if applicable
    if args.pass_k > 1:
        print(f"\n  Pass@k (n={args.pass_k}, temperature={temperature}):")
        pass_at_k = metrics["pass_at_k"]
        for k in sorted(pass_at_k, key=lambda x: int(x)):
            print(f"    pass@{k}: {pass_at_k[k]:.4f}")
        print(f"\n  Pass@k Per-Dataset:")
        for ds_name in sorted(metrics.get("pass_at_k_per_dataset", {})):
            ds_pak = metrics["pass_at_k_per_dataset"][ds_name]
            vals = "  ".join(f"@{k}={ds_pak[k]:.3f}" for k in sorted(ds_pak, key=lambda x: int(x)))
            print(f"    {ds_name:>12}: {vals}")

    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
