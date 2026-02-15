"""Evaluate soft token embedding mixing in diffusion-style generation.

Compares generation quality and convergence with varying soft_token_mixing values.
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

# Test prompts covering different uncertainty profiles
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
    mixing_values: list[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
    soft_token_top_k: Optional[int] = None,
    max_new_tokens: int = 128,
    num_steps: int = 32,
    inner_recurrence: int = 4,
    max_wavefront: int = 128,
    headway: int = 1,
    prompt_categories: list[str] = ["reasoning", "factual", "creative"],
    output_file: Optional[str] = None,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
):
    """Run soft embedding mixing evaluation.

    Args:
        model_name: HuggingFace model identifier.
        device: Device to run on.
        mixing_values: List of soft_token_mixing values to sweep.
        soft_token_top_k: Optional override for top-k in soft embedding computation.
        max_new_tokens: Maximum tokens to generate per prompt.
        num_steps: Number of recurrence steps.
        inner_recurrence: Inner recurrence iterations per diffusion step.
        max_wavefront: Maximum wavefront size.
        headway: Number of new positions per step.
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
        return_dict_in_generate=True,
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
            results[category][prompt] = {}

            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(torch_device)

            for mixing in mixing_values:
                print(f"\n  soft_token_mixing={mixing:.2f}")

                output = model.generate_diffusion_style(
                    input_ids,
                    generation_config=generation_config,
                    tokenizer=tokenizer,
                    num_steps=num_steps,
                    inner_recurrence=inner_recurrence,
                    max_wavefront=max_wavefront,
                    headway=headway,
                    soft_token_mixing=mixing,
                    soft_token_top_k=soft_token_top_k,
                    return_analysis_tablets=True,
                )

                sequences = output.sequences
                summary = output.scores
                analysis = output.hidden_states

                generated_text = tokenizer.decode(sequences[0, input_ids.shape[1]:], skip_special_tokens=True)
                print(f"    Generated: {generated_text[:120]}...")

                # Extract metrics
                recurrence_per_pos = summary["recurrence_per_position"]
                entry = {
                    "generated_text": generated_text,
                    "diffusion_steps": summary["diffusion_steps"],
                    "num_core_forward_passes": summary["num_core_forward_passes"],
                    "num_tokens_forward": summary["num_tokens_forward"],
                    "num_cache_clears": summary["num_cache_clears"],
                    "num_standing_waves": summary["num_standing_waves"],
                    "gen_seq_length": summary["gen_seq_length"],
                    "mean_recurrence": float(recurrence_per_pos.float().mean()),
                    "max_recurrence": int(recurrence_per_pos.max()),
                }

                # Soft entropy statistics
                if analysis is not None and analysis.get("soft_entropy_tablet") is not None:
                    ent_tablet = analysis["soft_entropy_tablet"]
                    nonzero_mask = ent_tablet > 0
                    if nonzero_mask.any():
                        entry["mean_soft_entropy"] = float(ent_tablet[nonzero_mask].mean())
                        entry["max_soft_entropy"] = float(ent_tablet[nonzero_mask].max())
                        entry["min_soft_entropy"] = float(ent_tablet[nonzero_mask].min())
                    else:
                        entry["mean_soft_entropy"] = 0.0
                        entry["max_soft_entropy"] = 0.0
                        entry["min_soft_entropy"] = 0.0

                print(f"    Steps: {entry['diffusion_steps']}, "
                      f"Passes: {entry['num_core_forward_passes']}, "
                      f"Mean recurrence: {entry['mean_recurrence']:.1f}")
                if "mean_soft_entropy" in entry:
                    print(f"    Soft entropy: mean={entry['mean_soft_entropy']:.3f}, "
                          f"max={entry['max_soft_entropy']:.3f}")

                results[category][prompt][f"mixing_{mixing:.2f}"] = entry

    # Print summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}")
    for category in results:
        print(f"\n--- {category.upper()} ---")
        for prompt in results[category]:
            print(f"\nPrompt: {prompt[:60]}...")
            for mixing_key in sorted(results[category][prompt]):
                entry = results[category][prompt][mixing_key]
                ent_str = ""
                if "mean_soft_entropy" in entry:
                    ent_str = f", entropy={entry['mean_soft_entropy']:.3f}"
                print(f"  {mixing_key}: steps={entry['diffusion_steps']}, "
                      f"passes={entry['num_core_forward_passes']}, "
                      f"mean_rec={entry['mean_recurrence']:.1f}{ent_str}")

    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {output_file}")

    return results


if __name__ == "__main__":
    CLI(run_evaluation)
