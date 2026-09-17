# Copyright (C) 2026 Apple Inc. All Rights Reserved.
QWEN35_MM_PROCESSOR_KWARGS = {"min_pixels": 1, "max_pixels": 16777216}


def load_model(
    model: str,
    tensor_parallel_size: int = 1,
    dtype: str = "bfloat16",
    trust_remote_code: bool = True,
    max_model_len: int = 32768,
    gpu_memory_utilization: float = 0.9,
    mm_processor_kwargs: dict = None,
    allowed_local_media_path: str = "/",
    **kwargs,
):
    """Load a LensVLM model with sensible defaults via vLLM.

    This wraps vLLM's LLM class with defaults tuned for LensVLM inference,
    so users don't need to pass verbose kwargs.

    Args:
        model: HuggingFace model ID or local path (e.g. "apple/LensVLM-9B").
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        dtype: Model dtype (default "bfloat16").
        trust_remote_code: Whether to trust remote code in the model repo.
        max_model_len: Maximum context length for vLLM.
        gpu_memory_utilization: Fraction of GPU memory vLLM may use.
        mm_processor_kwargs: Multimodal processor kwargs. Defaults to
            QWEN35_MM_PROCESSOR_KWARGS.
        allowed_local_media_path: Path prefix for local file:// image URIs.
        **kwargs: Additional kwargs passed to vLLM LLM constructor.

    Returns:
        A vLLM LLM instance ready for inference.
    """
    from vllm import LLM

    if mm_processor_kwargs is None:
        mm_processor_kwargs = QWEN35_MM_PROCESSOR_KWARGS

    return LLM(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        mm_processor_kwargs=mm_processor_kwargs,
        allowed_local_media_path=allowed_local_media_path,
        **kwargs,
    )
