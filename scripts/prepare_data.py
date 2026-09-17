#!/usr/bin/env python3
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""Prepare QA evaluation data for LensVLM from HuggingFace datasets.

This uses the SAME dataset providers that built the paper's evaluation set
(``lensvlm/dataset_providers``). Each provider loads its source dataset, builds
a long-text context with distractor augmentation, and locates the evidence
character spans. We then render each context into compressed page images and
save the eval JSON consumed by ``eval/evaluate.py``.

Reproducing the paper:
  The paper's reported numbers use the ``train`` split; ``--split validation``
  is an alternative split.

Usage:
    python scripts/prepare_data.py \
        --dataset hotpotqa \
        --output_dir ./data/hotpotqa_10x \
        --compression 10x \
        --max_samples 500
"""
import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from lensvlm.dataset_providers import PROVIDERS


def main():
    parser = argparse.ArgumentParser(
        description="Prepare QA evaluation data for LensVLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/prepare_data.py --dataset hotpotqa --output_dir ./data/hotpotqa_10x
  python scripts/prepare_data.py --dataset nq --output_dir ./data/nq_10x --max_samples 500
  python scripts/prepare_data.py --dataset musique --output_dir ./data/musique_10x
        """,
    )
    parser.add_argument("--dataset", required=True, choices=list(PROVIDERS.keys()))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--compression", default="10x", choices=["5x", "10x", "15x"])
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--min_tokens", type=int, default=3000,
                        help="Minimum context length in tokens (distractors added to reach this)")
    parser.add_argument("--max_tokens", type=int, default=12000,
                        help="Maximum context length in tokens")
    parser.add_argument("--target_tokens", type=int, default=0,
                        help="Target average token count (0 = use min_tokens as floor)")
    parser.add_argument("--split", default="train",
                        help="HF split to draw from. 'train' matches the paper's reported "
                             "numbers; 'validation' is an alternative split.")
    parser.add_argument("--max_hops", type=int, default=3,
                        help="Exclude samples with num_hops >= max_hops (matches the paper). "
                             "0 = keep all hops.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from lensvlm.rendering import parallel_render_pages
    from lensvlm.rendering_config import sample_render_config

    print(f"[1/3] Loading {args.dataset} via provider (split={args.split})")
    provider = PROVIDERS[args.dataset](split=args.split)

    # Multi-hop datasets lose samples to the max_hops filter — over-request so we
    # still reach max_samples after filtering (Musique is ~50% 2-hop).
    n_request = args.max_samples
    if args.max_hops and args.dataset == "musique":
        n_request = int(args.max_samples / 0.5) + 100

    cold_samples = provider.load_samples(
        num_samples=n_request,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        target_tokens=args.target_tokens,
    )
    if args.max_hops:
        cold_samples = [s for s in cold_samples if s.num_hops < args.max_hops]
    cold_samples = cold_samples[:args.max_samples]

    # Convert EvalSample -> render input dicts
    raw_samples = []
    for i, s in enumerate(cold_samples):
        raw_samples.append({
            "id": f"{args.dataset}_{i}",
            "dataset": args.dataset,
            "question": s.question,
            "answer": s.answer,
            "answers": s.answers,
            "context": s.context,
            "evidence_spans": s.evidence_spans,
            "num_hops": s.num_hops,
        })

    print(f"\n[2/3] Rendering {len(raw_samples)} documents at {args.compression} compression")
    config = sample_render_config(compression=args.compression)
    image_dir = os.path.join(args.output_dir, "rendered_images")
    os.makedirs(image_dir, exist_ok=True)

    render_results = parallel_render_pages(
        raw_samples, config,
        output_dir=image_dir,
        num_workers=args.num_workers,
        text_key="context",
        id_key="id",
        evidence_spans_key="evidence_spans",
    )

    print(f"\n[3/3] Building eval dataset")
    eval_samples = []
    for s, r in zip(raw_samples, render_results):
        if r is None:
            continue
        rel_image_paths = [os.path.relpath(p, args.output_dir) for p in r["image_paths"]]
        eval_samples.append({
            "id": s["id"],
            "dataset": args.dataset,
            "question": s["question"],
            "answer": s["answer"],
            "answers": s["answers"],
            "images": rel_image_paths,
            "page_texts": r["page_texts"],
            "num_pages": r["num_pages"],
            "gt_pages": r.get("evidence_pages", []),
        })

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "eval.json")
    with open(output_path, "w") as f:
        json.dump(eval_samples, f, indent=2)

    n = len(eval_samples)
    has_gt = sum(1 for s in eval_samples if s["gt_pages"])
    avg_pages = sum(s["num_pages"] for s in eval_samples) / n if n else 0
    avg_tokens = sum(len("\n".join(s["page_texts"]).split()) for s in eval_samples) / n if n else 0
    print(f"\n  Dataset: {args.dataset} (split={args.split})")
    print(f"  Output: {output_path}")
    print(f"  Samples: {n}")
    print(f"  Avg pages: {avg_pages:.1f}")
    print(f"  Avg tokens: {avg_tokens:.0f}")
    print(f"  Evidence pages found: {has_gt}/{n}")
    print(f"\nTo evaluate:")
    print(f"  python eval/evaluate.py --model apple/LensVLM-9B "
          f"--data_path {output_path} --compression {args.compression} "
          f"--output_dir ./results/{args.dataset}_{args.compression}")


if __name__ == "__main__":
    main()
