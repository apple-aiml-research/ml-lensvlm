#!/usr/bin/env python3
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""Run LensVLM on the bundled HotpotQA demo sample and display the trajectory."""
import json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'


def main():
    import argparse
    parser = argparse.ArgumentParser(description="LensVLM bundled HotpotQA demo")
    parser.add_argument("--model", default="apple/LensVLM-9B")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--output_dir", default=None,
                        help="Where to save rendered page images + result (default: <repo>/demo_output)")
    args = parser.parse_args()

    from lensvlm import load_model, render_pages, run_multi_turn_inference
    from lensvlm.rendering_config import sample_render_config
    from lensvlm.prompts import SYSTEM_PROMPT, build_user_prompt
    import lensvlm.evaluate as _qa
    _qa.QA_EVAL_VERBOSE = False  # silence [QA_EVAL] progress logs in the demo

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = args.output_dir or os.path.join(repo_root, "demo_output")
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    # Load bundled example
    example_path = os.path.join(repo_root, "examples", "hotpotqa_demo.json")
    with open(example_path) as f:
        sample = json.load(f)

    print(f"Question: {sample['question']}")
    print(f"Gold Answer: {sample['answer']}")
    print(f"Context: {len(sample['context'])} chars ({len(sample['context'].split())} tokens)")

    # Render
    config = sample_render_config(compression="10x")
    rendered = render_pages(sample["context"], config)
    print(f"Rendered: {rendered.num_pages} pages at 10x compression")

    # Save rendered page images to a persistent output dir
    image_paths = []
    for i, page in enumerate(rendered.pages):
        path = os.path.join(img_dir, f"page_{i+1:03d}.png")
        page.save(path)
        image_paths.append(path)
    print(f"Saved {len(image_paths)} page images to {img_dir}")

    # Build eval sample
    test_samples = [{
        "id": sample["id"],
        "dataset": sample["dataset"],
        "question": sample["question"],
        "answer": sample["answer"],
        "answers": sample["answers"],
        "images": image_paths,
        "page_texts": rendered.page_texts,
        "num_pages": rendered.num_pages,
        "gt_pages": [],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(sample["question"], rendered.num_pages)},
        ],
    }]

    # Load model and run inference
    print(f"\nLoading model: {args.model}")
    llm = load_model(args.model, tensor_parallel_size=args.tp)

    predictions = run_multi_turn_inference(llm, test_samples, return_trajectories=True)
    p = predictions[0]

    # Print the multi-turn conversation exactly as produced — the model's raw
    # responses, tool calls, and the tool responses — untruncated, unformatted.
    for msg in p.get("trajectory", []):
        role = msg["role"]
        content = msg["content"]
        if isinstance(content, list):
            texts = [c.get("text", "") for c in content
                     if isinstance(c, dict) and c.get("type") == "text"]
            n_img = sum(1 for c in content
                        if isinstance(c, dict) and c.get("type") in ("image", "image_url"))
            content = (f"[{n_img} page images]\n" if n_img else "") + "\n".join(texts)
        print(f"<|{role}|>")
        print(content)
        print()


    # Save the full result (answer + trajectory) for inspection
    result_path = os.path.join(out_dir, "result.json")
    with open(result_path, "w") as f:
        json.dump(p, f, indent=2)
    print(f"\nSaved result to {result_path}")

    del llm


if __name__ == "__main__":
    main()
