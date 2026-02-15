"""Evaluate standard autoregressive sampling (no wavefront/diffusion).

Baseline comparison for diffusion-style and soft embedding generation.
"""

import torch
import json
import sys
from pathlib import Path
from typing import Optional

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from transformers import AutoTokenizer, GenerationConfig
from recpre.raven_modeling_minimal import RavenForCausalLM
from jsonargparse import CLI

# Same prompts as eval_soft_embeddings.py
REASONING_PROMPTS = [
    "Janet pays $40/hour for 3 hours per week of clarinet lessons and $28/hour for 5 hours a week of piano lessons. How much more does she spend on piano lessons than clarinet lessons per week? Let's think step by step.",
    "A store sells apples for $2 each and oranges for $3 each. If someone buys 4 apples and 6 oranges, what is the total cost? Let's work through this.",
]

FACTUAL_PROMPTS = [
    "The capital of Australia is",
    "The speed of light in a vacuum is approximately",
    "The chemical formula for water is",
]

CREATIVE_PROMPTS = [
    "Once upon a time in a land where rivers flowed uphill,",
    "The most surprising thing about the deep ocean is",
    "Write a short poem about a robot learning to paint:",
]

ALL_PROMPTS = {
    "reasoning": REASONING_PROMPTS,
    "factual": FACTUAL_PROMPTS,
    "creative": CREATIVE_PROMPTS,
}


def run_evaluation(
    model_name: str = "tomg-group-umd/huginn-0125",
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    max_new_tokens: int = 128,
    prompt_categories: list[str] = ["reasoning", "factual", "creative"],
    output_file: Optional[str] = None,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
):
    """Run standard autoregressive sampling evaluation (no wavefront/diffusion).

    Args:
        model_name: HuggingFace model identifier.
        device: Device to run on.
        max_new_tokens: Maximum tokens to generate per prompt.
        prompt_categories: Which prompt categories to test.
        output_file: Path to write JSON results.
        temperature: Sampling temperature.
        top_k: Top-k filtering parameter.
        top_p: Top-p (nucleus) filtering parameter.
    """
    torch_device = torch.device(device)

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = RavenForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    ).to(torch_device)
    model.eval()

    generation_config = GenerationConfig(
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )

    results = {}

    for category in prompt_categories:
        if category not in ALL_PROMPTS:
            print(f"Skipping unknown category: {category}")
            continue

        prompts = ALL_PROMPTS[category]
        results[category] = {}

        for prompt in prompts:
            print(f"\n{'='*80}")
            print(f"Category: {category}")
            print(f"Prompt: {prompt[:80]}...")

            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(torch_device)

            output = model.generate(
                input_ids,
                generation_config=generation_config,
            )

            generated_text = tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)
            print(f"  Generated: {generated_text[:120]}...")

            results[category][prompt] = {
                "generated_text": generated_text,
                "num_tokens_generated": output.shape[1] - input_ids.shape[1],
            }

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for category in results:
        print(f"\n--- {category.upper()} ---")
        for prompt in results[category]:
            entry = results[category][prompt]
            print(f"\nPrompt: {prompt[:60]}...")
            print(f"  Tokens: {entry['num_tokens_generated']}")
            print(f"  Output: {entry['generated_text'][:100]}...")

    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {output_file}")

    return results


if __name__ == "__main__":
    CLI(run_evaluation)
