# Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""
Shared prompt definitions for the document QA pipeline.
"""

# Tool-calling system prompt (default for agentic tool use)
SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about multi-page documents. "
    "You can view the page images and use the read_page tool to read the actual text "
    "content of any page for detailed analysis.\n\n"
    "You have a tool called `read_page` that reads the text content of any page. To use it:\n"
    '<tool_call>{"name": "read_page", "arguments": {"page": PAGE_NUMBER}}</tool_call>\n\n'
    "You can call read_page multiple times on different pages. "
    "Always reason inside <think> and </think> tags before taking any action. "
    "If you need more detail from a specific page, call read_page to get its text content. "
    "Do not read the entire document — only read pages that are likely relevant. "
    "Once you have enough information, provide your final answer."
)

# Non-tool system prompt (baseline: no read_page tool, direct answer from images)
NO_TOOL_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about multi-page documents. "
    "You can view the page images to find and analyze relevant information.\n\n"
    "Always reason inside <think> and </think> tags before providing your answer. "
    "Examine the document page images carefully, identify the pages most likely "
    "to contain relevant information, and use what you can see to answer the question. "
    "Once you have enough information, provide your final answer."
)


# Image-zoom system prompt: thumbnails + original page image (visual eval)
SYSTEM_PROMPT_IMAGE_ZOOM = (
    "You are a helpful assistant that answers questions about multi-page documents. "
    "You can view low-resolution thumbnail images of all pages and use the read_page tool "
    "to get the original full-resolution image of any page for detailed analysis.\n\n"
    "You have a tool called `read_page` that shows the original image of any page. "
    "Pages are numbered starting from 1. To use it:\n"
    '<tool_call>{"name": "read_page", "arguments": {"page": PAGE_NUMBER}}</tool_call>\n\n'
    "You can call read_page multiple times on different pages. "
    "Always reason inside <think> and </think> tags before taking any action. "
    "First examine the thumbnail images to identify which pages likely contain relevant information, "
    "then call read_page to get the original image of those pages. "
    "Once you have enough information, provide your final answer."
)


def build_user_prompt(question: str, num_pages: int) -> str:
    """User prompt = <image> tags + question only."""
    image_tags = "<image>" * num_pages
    return f"{image_tags}\n\nQuestion: {question}\n"
