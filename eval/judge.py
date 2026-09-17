#!/usr/bin/env python3
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""Judge predictions using a remote LLM judge API.

Usage:
    python eval/judge.py \
        --predictions ./results/hotpotqa_5x/predictions.json \
        --judge_url http://localhost:8000/v1 \
        --judge_model Qwen/Qwen3.5-397B-A17B-FP8 \
        --output_dir ./results/hotpotqa_5x_judged
"""
import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--judge_url", required=True)
    parser.add_argument("--judge_model", default="Qwen/Qwen3.5-397B-A17B-FP8")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from lensvlm.evaluate import run_judge_evaluation, compute_heuristic_metrics

    with open(args.predictions) as f:
        predictions = json.load(f)
    print(f"Loaded {len(predictions)} predictions")

    # Compute heuristic metrics first
    metrics, predictions = compute_heuristic_metrics(predictions)

    # Run judge
    print(f"Running judge: {args.judge_model} via {args.judge_url}")
    judge_metrics, predictions = run_judge_evaluation(
        predictions=predictions,
        judge_model=args.judge_model,
        judge_url=args.judge_url,
    )
    print(f"Judge accuracy: {judge_metrics.get('accuracy', 0):.3f}")

    # Per-dataset breakdown
    from collections import defaultdict
    per_dataset = defaultdict(lambda: {"total": 0, "hits": 0, "judge_correct": 0})
    for p in predictions:
        ds = p["dataset"]
        per_dataset[ds]["total"] += 1
        per_dataset[ds]["hits"] += int(p.get("zoom_hit", False))
        per_dataset[ds]["judge_correct"] += int(p.get("judge_correct", False))

    sep = "=" * 60
    print(f"\n{sep}")
    print("RESULTS (397B Judge)")
    print(f"{sep}")
    print(f"{'Dataset':<15} {'N':>4} {'Judge Acc':>10} {'Sel.Acc':>8}")
    print("-" * 45)
    for ds in sorted(per_dataset.keys()):
        d = per_dataset[ds]
        n = d["total"]
        print(f"{ds:<15} {n:>4} {d['judge_correct']/n:>10.1%} {d['hits']/n:>8.1%}")
    total = len(predictions)
    total_correct = sum(1 for p in predictions if p.get("judge_correct", False))
    total_hits = sum(1 for p in predictions if p.get("zoom_hit", False))
    print("-" * 45)
    print(f"{'OVERALL':<15} {total:>4} {total_correct/total:>10.1%} {total_hits/total:>8.1%}")
    print(f"\nAvg turns: {sum(p['num_turns'] for p in predictions)/total:.1f}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "predictions.json"), "w") as f:
        json.dump(predictions, f, indent=2)
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump({"judge": judge_metrics, "heuristic": metrics, "per_dataset": dict(per_dataset)}, f, indent=2)
    print(f"Saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
