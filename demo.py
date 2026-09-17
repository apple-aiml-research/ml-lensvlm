#!/usr/bin/env python3
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""End-to-end LensVLM inference on a custom document.

Given a text document, a question, and a compression rate, renders the text
into compressed page images, runs multi-turn tool-use inference, and prints the
full trajectory (reasoning + tool calls). Rendered page images and the full
result are saved under --output_dir (default: ./demo_output).

Usage:
    python demo.py \
        --model apple/LensVLM-9B \
        --text_file document.txt \
        --question "What is the main finding?" \
        --compression 10x
"""

import argparse
import json
import os

from lensvlm.rendering_config import sample_render_config
from lensvlm.rendering import render_pages
from lensvlm.prompts import SYSTEM_PROMPT, build_user_prompt
from lensvlm.evaluate import (
    parse_tool_call, extract_answer, _clean_response, _has_tool_call,
    _build_tool_response, _build_multimodal_content,
)


def main():
    parser = argparse.ArgumentParser(description="LensVLM end-to-end inference")
    parser.add_argument("--model", type=str, default="apple/LensVLM-9B",
                        help="Model path or HuggingFace model ID")
    parser.add_argument("--text", type=str, default=None, help="Input text")
    parser.add_argument("--text_file", type=str, default=None, help="Path to text file")
    parser.add_argument("--question", type=str, required=True, help="Question to answer")
    parser.add_argument("--compression", type=str, default="10x", choices=["5x", "10x", "15x"],
                        help="Compression rate")
    parser.add_argument("--max_turns", type=int, default=6, help="Max tool-use turns")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--output_dir", type=str, default="./demo_output",
                        help="Where to save rendered page images + result")
    args = parser.parse_args()

    if args.text_file:
        with open(args.text_file) as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        parser.error("Provide either --text or --text_file")

    img_dir = os.path.join(args.output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    # Render
    print(f"[1/3] Rendering text at {args.compression} compression...")
    config = sample_render_config(compression=args.compression)
    rendered = render_pages(text, config)
    if rendered is None:
        print("ERROR: Text too short to render.")
        return
    print(f"      {rendered.num_pages} pages generated ({config.to_key()})")

    # Save rendered page images
    image_paths = []
    for i, page in enumerate(rendered.pages):
        path = os.path.join(img_dir, f"page_{i + 1:03d}.png")
        page.save(path)
        image_paths.append(path)
    print(f"      saved {len(image_paths)} page images to {img_dir}")

    # Load model
    from vllm import LLM, SamplingParams
    from lensvlm.vision_config import QWEN35_MM_PROCESSOR_KWARGS
    print(f"[2/3] Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        dtype="bfloat16",
        trust_remote_code=True,
        # Without this, vLLM sizes the KV cache for the model's full
        # max_position_embeddings (262144), needing ~34 GB of KV cache on top of
        # ~19 GB of weights. Matches the default in lensvlm.vision_config.load_model.
        max_model_len=32768,
        mm_processor_kwargs=QWEN35_MM_PROCESSOR_KWARGS,
        allowed_local_media_path="/",
    )

    # Multi-turn inference
    print(f"[3/3] Running inference (max {args.max_turns} turns)...")
    sampling_params = SamplingParams(
        max_tokens=2048,
        temperature=0.0,
        stop=["</tool_call>"],
        include_stop_str_in_output=True,
    )

    user_text = build_user_prompt(args.question, rendered.num_pages)
    user_content = _build_multimodal_content(user_text, image_paths)
    conv = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    sample = {"page_texts": rendered.page_texts}
    expanded_pages = []

    for turn in range(1, args.max_turns + 1):
        outputs = llm.chat([conv], sampling_params=sampling_params, add_generation_prompt=True)
        response = _clean_response(outputs[0].outputs[0].text)
        conv.append({"role": "assistant", "content": response})

        page_num = parse_tool_call(response) if _has_tool_call(response) else None
        if page_num is not None:
            expanded_pages.append(page_num)
            tool_resp = _build_tool_response(page_num, sample, zoom_returns_image=False)
            if tool_resp is not None:
                conv.append({"role": "user", "content": tool_resp})
                if turn >= args.max_turns:
                    break
            else:
                break
        else:
            break

    answer = extract_answer(response)

    # Print results
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"ANSWER: {answer}")
    print(f"Pages expanded: {expanded_pages}")
    print(f"Turns: {turn}")
    print(sep)
    print("FULL TRAJECTORY:")
    for msg in conv:
        role = msg["role"]
        content = msg["content"] if isinstance(msg["content"], str) else "<multimodal>"
        if role == "system":
            continue
        print(f"\n[{role}]")
        if role == "assistant":
            print(content)  # full model output
        else:
            # tool/user input (page text read back) — cap for readability
            print(content[:500] + ("..." if len(content) > 500 else ""))
    print(sep)

    # Save the full result for inspection
    result = {
        "question": args.question, "answer": answer,
        "pages_expanded": expanded_pages, "turns": turn,
        "compression": args.compression, "num_pages": rendered.num_pages,
    }
    result_path = os.path.join(args.output_dir, "result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved result to {result_path}")


if __name__ == "__main__":
    main()
