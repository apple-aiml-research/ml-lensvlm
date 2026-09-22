# LensVLM: Selective Context Expansion for Compressed Visual Representation of Text

This is the official repo for [LensVLM: Selective Context Expansion for Compressed Visual Representation of Text](https://arxiv.org/abs/2605.07019). LensVLM is 9B Vision Language Model (VLM) that scans compressed images, then selectively expands only the relevant images to their uncompressed form via learned tools.

## Getting Started

### Installation

```bash
pip install -r requirements.txt
```

## Quick Start

Start a example run:

```bash
python scripts/run_demo.py
```

Expected output - the model's raw multi-turn trajectory:

```
Question: What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?

<|system|>
You are a helpful assistant that answers questions about multi-page documents.
You have a tool called `read_page` that reads the text content of any page ...
Always reason inside <think> and </think> tags before taking any action.

<|user|>
[15 page images]
Question: What government position was held by the woman who portrayed Corliss
Archer in the film Kiss and Tell?

<|assistant|>
<think>... Scanning the visible page thumbnails, Page 10 contains text mentioning
"American actress, singer" ... I will read Page 10 to find the specific name of
the actress and any associated government positions.
</think>
<tool_call>
{"name": "read_page", "arguments": {"page": 10}}
</tool_call>

<|user|>
<tool_response>
Text content of Page 10:
... an American actress, singer, dancer, businesswoman, and diplomat who was
Hollywood's number one box-office draw as a child actress from 1935 to 1938. As
an adult, she was named United States ambassador to Ghana and to Czechoslovakia
and also served as Chief of Protocol of the United States. ...
</tool_response>

<|assistant|>
<think>... the text describes an American actress ... who "also served as Chief of
Protocol of the United States." ... The text clearly identifies "Chief of
Protocol of the United States" as a government position held by this individual.
</think>

Chief of Protocol of the United States
```

The model scans 15 compressed page images, identifies Page 10 as relevant, reads its text content via the `read_page` tool, and connects Shirley Temple → Corliss Archer → Chief of Protocol in 2 turns. The rendered page images and the full result (`result.json`) are written to `./demo_output/`.

For a custom document:

```bash
python demo.py \
    --text_file document.txt \
    --question "What is the main finding?" \
    --compression 10x
```

Compression options: `5x`, `10x`, `15x`. Page images and the result are saved under `--output_dir` (default `./demo_output`).

## Data Preparation

Prepare evaluation data from HuggingFace datasets (`--dataset`):

```bash
python scripts/prepare_data.py \
    --dataset hotpotqa \
    --output_dir ./data/hotpotqa_5x \
    --compression 5x \
    --max_samples 500 \
    --split train
```

The dataset providers in `lensvlm/dataset_providers/` — build long-document contexts by augmenting the gold passages with distractor paragraphs from other samples, then `prepare_data.py` renders them into compressed page documents and records the evidence pages (`gt_pages`).

## Evaluation

The evaluation runs LensVLM inference and scores the answers with an LLM judge, so first host the judge server. Any OpenAI-compatible endpoint works; the paper uses `Qwen/Qwen3.5-397B-A17B-FP8` (served on a 8*B200 GPU node):

```bash
vllm serve Qwen/Qwen3.5-397B-A17B-FP8 \
    --port 8000
# -> OpenAI-compatible endpoint at http://localhost:8000/v1
```

Then run the evaluation (inference + judging), pointing `--judge_url` at that server:

```bash
python eval/evaluate.py \
    --model <path-to-model> \
    --data_path ./data/hotpotqa_5x/eval.json \
    --compression 5x \
    --output_dir ./results/hotpotqa_5x \
    --judge_url http://localhost:8000/v1
```

This runs LensVLM inference and judges answer correctness in one pass. Pass `--skip_judge` to compute page-selection metrics without a judge. Predictions (`predictions.json`) and metrics (`metrics.json`) are written to `--output_dir`.

## Project Structure

```
ml-lensvlm/
├── lensvlm/                  # Core library
│   ├── rendering.py          # Text -> compressed page images (+ render_from_page_texts)
│   ├── rendering_config.py   # Compression presets (5x/10x/15x)
│   ├── dataset_providers/    # HF dataset loaders + distractor augmentation (paper's eval-set construction)
│   ├── evaluate.py           # Multi-turn tool-use inference loop
│   ├── evaluator.py          # LLM-as-judge evaluation
│   ├── prompts.py            # System prompts
│   └── vision_config.py      # Model loading (load_model) + vision processor config
├── scripts/
│   ├── prepare_data.py       # Data preparation (HotpotQA / NQ / Musique) with distractors
│   └── run_demo.py           # Bundled HotpotQA demo
├── eval/
│   ├── evaluate.py           # Evaluation pipeline (inference + LLM judge)
│   └── judge.py              # Standalone LLM-as-judge for existing predictions
├── examples/
│   └── hotpotqa_demo.json    # Bundled demo sample
├── fonts/                    # Bundled fonts for deterministic rendering
└── demo.py                   # End-to-end inference on a custom document
```

## Citation

```bibtex
@article{xie2026lensvlm,
  title={LensVLM: Selective Context Expansion for Compressed Visual Representation of Text},
  author={Xie, Roy and Friedman, Dan and Yu, Donghan and Pan, Bowen and Fifty, Christopher and Kim, Jang-Hyun and Du, Xianzhi and Gan, Zhe and Rathod, Vivek and Dhingra, Bhuwan},
  journal={arXiv preprint arXiv:2605.07019},
  year={2026}
}
```

## License

This software and accompanying models have been released under the following
licenses:
- Code: [Apple Sample Code License (ASCL)](./LICENSE)
- ML model: [Apple Machine Learning Research Model License](https://huggingface.co/apple/LensVLM-9B/blob/main/LICENSE)

## Acknowledgements

Our codebase is built using multiple opensource contributions, please see [ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS) for more details. See [NOTICE](NOTICE) for the full attribution and statement of modifications.
