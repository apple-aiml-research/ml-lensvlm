#!/usr/bin/env python3
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""Evaluate LensVLM on prepared data.

Dataset-agnostic: runs the full LensVLM multi-turn tool-use pipeline on any
evaluation set produced by ``scripts/prepare_data.py``.

1. Load pre-rendered eval data (output of scripts/prepare_data.py)
2. Run multi-turn tool-use inference (lensvlm/evaluate.py)
3. Report page-selection + answer metrics (optional LLM-as-judge)

Usage:
    python eval/evaluate.py \
        --model apple/LensVLM-9B \
        --data_path ./data/hotpotqa_10x/eval.json \
        --compression 10x \
        --output_dir ./results/hotpotqa_10x \
        --judge_url http://localhost:8000/v1
"""
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import argparse
import json
import sys


def load_eval_data(data_path: str, max_samples: int = 0) -> list:
    """Load prepared evaluation data from a JSON file.

    Expected format (output of scripts/prepare_data.py):
    [
        {
            "id": "hotpotqa_0",
            "dataset": "hotpotqa",
            "question": "...",
            "answer": "...",
            "answers": ["...", "..."],
            "images": ["path/to/page1.png", ...],
            "page_texts": ["text on page 1", ...],
            "num_pages": N,
            "gt_pages": [1, 3],  # pages containing the evidence
        },
        ...
    ]
    """
    with open(data_path) as f:
        samples = json.load(f)

    print(f"  Loaded {len(samples)} eval samples from {data_path}")

    if max_samples > 0:
        samples = samples[:max_samples]
        print(f"  Limited to {len(samples)} samples")

    return samples


def main():
    parser = argparse.ArgumentParser(description="LensVLM evaluation on prepared QA data")
    parser.add_argument("--model", required=True, help="Model checkpoint path or HF ID")
    parser.add_argument("--data_path", required=True,
                        help="Path to pre-rendered eval JSON (from scripts/prepare_data.py)")
    parser.add_argument("--compression", default="10x", choices=["5x", "10x", "15x"],
                        help="Compression preset (must match what was used during rendering)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max_turns", type=int, default=6)
    parser.add_argument("--max_samples", type=int, default=0, help="0 = all")
    parser.add_argument("--judge_url", type=str, default=None)
    parser.add_argument("--judge_model", type=str, default=None)
    parser.add_argument("--skip_judge", action="store_true")
    args = parser.parse_args()

    # Repo root (eval/ is one level below it) so `import lensvlm` works from anywhere.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from lensvlm.prompts import SYSTEM_PROMPT, build_user_prompt
    from lensvlm.vision_config import load_model
    from lensvlm.evaluate import (
        run_multi_turn_inference, compute_heuristic_metrics,
        run_judge_evaluation, _unload_vllm,
    )

    # Step 1: Load pre-rendered data
    print(f"[1/4] Loading eval data from {args.data_path}")
    samples = load_eval_data(args.data_path, max_samples=args.max_samples)

    # Make image paths absolute relative to the data file directory
    data_dir = os.path.dirname(os.path.abspath(args.data_path))
    for sample in samples:
        if "images" in sample:
            sample["images"] = [
                os.path.join(data_dir, p) if not os.path.isabs(p) else p
                for p in sample["images"]
            ]

    # Step 2: Build eval samples with messages
    print(f"\n[2/4] Building eval samples")
    eval_samples = []
    for s in samples:
        eval_sample = {
            "id": s["id"],
            "dataset": s.get("dataset", "qa"),
            "question": s["question"],
            "answer": s["answer"],
            "answers": s.get("answers", [s["answer"]]),
            "images": s["images"],
            "page_texts": s["page_texts"],
            "num_pages": s["num_pages"],
            "gt_pages": s.get("gt_pages", []),
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(s["question"], s["num_pages"])},
            ],
        }
        eval_samples.append(eval_sample)

    print(f"  Prepared {len(eval_samples)} eval samples")

    # Step 3: Run inference
    print(f"\n[3/4] Running inference on {len(eval_samples)} samples")
    llm = load_model(
        args.model,
        tensor_parallel_size=args.tp,
        max_model_len=32768,
        gpu_memory_utilization=0.9,
    )

    predictions = run_multi_turn_inference(
        llm=llm,
        test_samples=eval_samples,
        max_turns=args.max_turns,
        temperature=0.0,
    )

    _unload_vllm(llm)
    del llm

    # Step 4: Evaluate
    print(f"\n[4/4] Computing metrics")
    metrics, predictions = compute_heuristic_metrics(predictions)
    print(f"  Heuristic: EM={metrics['em']:.3f}, F1={metrics['f1']:.3f}")

    if not args.skip_judge and args.judge_url:
        print(f"  Running LLM judge via {args.judge_url}")
        judge_metrics, predictions = run_judge_evaluation(
            predictions=predictions,
            judge_model=args.judge_model or "Qwen/Qwen3.5-397B-A17B-FP8",
            judge_url=args.judge_url,
        )
        print(f"  Judge accuracy: {judge_metrics.get('accuracy', 0):.3f}")

    # Results summary
    n = len(predictions)
    zoom_hit = sum(1 for p in predictions if p.get("zoom_hit", False))
    zoom_correct = sum(1 for p in predictions if p.get("zoom_correct", False))
    avg_turns = sum(p["num_turns"] for p in predictions) / n if n else 0
    judge_correct = sum(1 for p in predictions if p.get("judge_correct", False))
    dataset = eval_samples[0].get("dataset", "qa") if eval_samples else "qa"

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"LensVLM Results — {dataset} ({args.compression} compression)")
    print(f"{sep}")
    print(f"  Samples:        {n}")
    print(f"  EM:             {metrics['em']*100:.1f}%")
    print(f"  F1:             {metrics['f1']*100:.1f}%")
    if args.judge_url and not args.skip_judge:
        print(f"  Judge Accuracy: {judge_correct/n*100:.1f}% ({judge_correct}/{n})")
    print(f"  Zoom Hit:       {zoom_hit/n*100:.1f}% ({zoom_hit}/{n})")
    print(f"{sep}")

    # Save (full metrics, incl. zoom-correct/avg-turns, kept in metrics.json)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "predictions.json"), "w") as f:
        json.dump(predictions, f, indent=2)
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump({
            "dataset": dataset,
            "heuristic": metrics,
            "compression": args.compression,
            "total": n,
            "judge_accuracy": (judge_correct / n if n else 0),
            "zoom_hit_rate": zoom_hit / n if n else 0,
            "zoom_correct_rate": zoom_correct / n if n else 0,
            "avg_turns": avg_turns,
        }, f, indent=2)
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
