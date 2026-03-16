"""LLM API pricing mapping for cost-constrained routing analysis.

Pricing based on Together.ai serverless inference (output tokens, $/M).
For models without exact pricing, fallback tiers based on parameter count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .cost_analysis import extract_model_size

# ── Together.ai output pricing ($/M output tokens, as of 2026) ──
# Keys are lowercase substrings matched against LLM names.
_TOGETHER_PRICES = {
    # Llama family
    "llama-3.3-70b": 0.88,
    "llama-3-70b": 0.88,
    "llama-3.1-70b": 0.88,
    "llama-3-8b": 0.10,
    "llama-3.1-8b": 0.10,
    "llama-2-70b": 0.90,
    "llama-2-13b": 0.22,
    "llama-2-7b": 0.20,
    "codellama-34b": 0.78,
    "codellama-13b": 0.22,
    "codellama-7b": 0.20,
    # Mistral family
    "mistral-7b": 0.20,
    "mixtral-8x7b": 0.60,
    "mixtral-8x22b": 1.20,
    # Qwen family
    "qwen2.5-72b": 0.90,
    "qwen2.5-7b": 0.30,
    "qwen2-72b": 0.90,
    "qwen2-7b": 0.30,
    "qwen2-1.5b": 0.10,
    "qwen2-0.5b": 0.05,
    "qwen1.5-72b": 0.90,
    "qwen1.5-32b": 0.80,
    "qwen1.5-14b": 0.30,
    "qwen1.5-7b": 0.20,
    "qwen1.5-4b": 0.10,
    "qwen1.5-1.8b": 0.10,
    "qwen1.5-0.5b": 0.05,
    # Yi family
    "yi-1.5-34b": 0.80,
    "yi-1.5-9b": 0.30,
    "yi-34b": 0.80,
    "yi-6b": 0.20,
    "yi-9b": 0.30,
    # DeepSeek family
    "deepseek-v2": 1.20,
    "deepseek-coder-33b": 0.80,
    "deepseek-coder-6.7b": 0.20,
    "deepseek-llm-67b": 0.90,
    "deepseek-llm-7b": 0.20,
    # Gemma family
    "gemma-2-27b": 0.80,
    "gemma-2-9b": 0.30,
    "gemma-2b": 0.10,
    "gemma-7b": 0.20,
    # Phi family
    "phi-3-mini": 0.10,
    "phi-3-medium": 0.30,
    "phi-3-small": 0.10,
    "phi-2": 0.10,
    # StarCoder
    "starcoder2-15b": 0.30,
    "starcoder2-7b": 0.20,
    "starcoder2-3b": 0.10,
    # SOLAR
    "solar-10.7b": 0.30,
    # Others
    "command-r": 0.50,
    "aya-23-8b": 0.20,
    "neural-chat-7b": 0.20,
    "gpt-neox-20b": 0.60,
    "gpt-neo-2.7b": 0.10,
    "gpt-neo-1.3b": 0.10,
    "gpt-neo-125m": 0.05,
    "pythia-160m": 0.05,
    "redpajama-incite-7b": 0.20,
    "redpajama-incite-3b": 0.10,
    "gpt-jt-6b": 0.20,
}

# Fallback tiers based on parameter count ($/M output tokens)
_SIZE_TIERS = [
    (3.0, 0.05),    # <3B
    (8.0, 0.20),    # 3-8B
    (13.0, 0.30),   # 8-13B
    (34.0, 0.80),   # 13-34B
    (70.0, 2.00),   # 34-70B
    (float("inf"), 5.00),  # >70B
]


def _match_together_price(name: str) -> float | None:
    """Try to match an LLM name against known Together.ai pricing."""
    low = name.lower().replace("__", "/")
    for key, price in _TOGETHER_PRICES.items():
        if key in low:
            return price
    return None


def _fallback_price(size_b: float | None) -> tuple[float, str]:
    """Estimate price from parameter count using tier-based fallback."""
    if size_b is None:
        return 0.20, "fallback_default"  # median tier
    for threshold, price in _SIZE_TIERS:
        if size_b <= threshold:
            return price, "fallback_size"
    return 5.00, "fallback_size"


def get_model_pricing(llm_names: list[str]) -> pd.DataFrame:
    """Map LLM names to output pricing per million tokens.

    Priority: Together.ai exact match > parameter-count fallback tier.

    Returns:
        DataFrame with columns ``[llm_name, price_per_m_tokens, price_source]``.
    """
    rows = []
    n_together = 0
    n_fallback_size = 0
    n_fallback_default = 0

    for name in llm_names:
        price = _match_together_price(name)
        if price is not None:
            rows.append({
                "llm_name": name,
                "price_per_m_tokens": price,
                "price_source": "together_ai",
            })
            n_together += 1
        else:
            size = extract_model_size(name)
            price, source = _fallback_price(size)
            rows.append({
                "llm_name": name,
                "price_per_m_tokens": price,
                "price_source": source,
            })
            if source == "fallback_size":
                n_fallback_size += 1
            else:
                n_fallback_default += 1

    df = pd.DataFrame(rows)

    print(f"\nPricing coverage ({len(df)} models):")
    print(f"  Together.ai match: {n_together}")
    print(f"  Fallback (size-based): {n_fallback_size}")
    print(f"  Fallback (default): {n_fallback_default}")
    print(f"  Price range: ${df['price_per_m_tokens'].min():.2f} - ${df['price_per_m_tokens'].max():.2f}/M tokens")

    return df
