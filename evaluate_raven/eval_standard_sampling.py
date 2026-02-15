"""Evaluate standard autoregressive sampling (no wavefront/diffusion).

Uses generate_diffusion_style with max_wavefront=1 to get true token-by-token
generation with a fixed number of recurrence iterations per token.
"""

import time
import torch
import json
import sys
from pathlib import Path
from typing import Optional

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from transformers import AutoTokenizer, GenerationConfig
from importlib.util import spec_from_file_location, module_from_spec
from jsonargparse import CLI

# Load raven_modeling_minimal directly to avoid pulling in training deps via recpre.__init__
_config_spec = spec_from_file_location("recpre.raven_config_minimal", wd / "recpre" / "raven_config_minimal.py")
_config_mod = module_from_spec(_config_spec)
_config_mod.__package__ = "recpre"
sys.modules["recpre.raven_config_minimal"] = _config_mod
_config_spec.loader.exec_module(_config_mod)

_model_spec = spec_from_file_location("recpre.raven_modeling_minimal", wd / "recpre" / "raven_modeling_minimal.py")
_model_mod = module_from_spec(_model_spec)
_model_mod.__package__ = "recpre"
sys.modules["recpre.raven_modeling_minimal"] = _model_mod
_model_spec.loader.exec_module(_model_mod)
RavenForCausalLM = _model_mod.RavenForCausalLM

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
    num_steps: int = 32,
    inner_recurrence: int = 4,
    prompt_categories: list[str] = ["reasoning", "factual", "creative"],
    output_file: Optional[str] = None,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
):
    """Run standard autoregressive sampling evaluation (no wavefront/diffusion).

    Uses generate_diffusion_style with max_wavefront=1 and headway=1 so that
    tokens are generated one at a time with a fixed recurrence budget.

    Args:
        model_name: HuggingFace model identifier.
        device: Device to run on.
        max_new_tokens: Maximum tokens to generate per prompt.
        num_steps: Number of outer diffusion steps (controls total recurrence budget).
        inner_recurrence: Number of core block iterations per diffusion step.
        prompt_categories: Which prompt categories to test.
        output_file: Path to write JSON results.
        temperature: Sampling temperature.
        top_k: Top-k filtering parameter.
        top_p: Top-p (nucleus) filtering parameter.
    """
    torch_device = torch.device(device)

    print(f"Loading model: {model_name}")
    print(f"Config: num_steps={num_steps}, inner_recurrence={inner_recurrence}, "
          f"max_wavefront=1 (standard autoregressive)")
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

            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(torch_device)

            torch.cuda.synchronize(torch_device)
            t0 = time.perf_counter()

            output = model.generate_diffusion_style(
                input_ids,
                generation_config=generation_config,
                tokenizer=tokenizer,
                num_steps=num_steps,
                inner_recurrence=inner_recurrence,
                max_wavefront=1,
                headway=1,
                return_analysis_tablets=True,
            )

            torch.cuda.synchronize(torch_device)
            t1 = time.perf_counter()

            sequences = output.sequences
            summary = output.scores
            generated_text = tokenizer.decode(sequences[0, input_ids.shape[1]:], skip_special_tokens=True)
            num_tokens = sequences.shape[1] - input_ids.shape[1]
            wall_time = t1 - t0
            tokens_per_sec = num_tokens / wall_time if wall_time > 0 else 0

            print(f"  Generated: {generated_text[:120]}...")
            print(f"  Tokens: {num_tokens}, Time: {wall_time:.2f}s, "
                  f"Tokens/s: {tokens_per_sec:.1f}")
            print(f"  Forward passes: {summary['num_core_forward_passes']}, "
                  f"Tokens forwarded: {summary['num_tokens_forward']}")

            recurrence_per_pos = summary["recurrence_per_position"]
            entry = {
                "generated_text": generated_text,
                "num_tokens_generated": num_tokens,
                "wall_time_s": round(wall_time, 3),
                "tokens_per_sec": round(tokens_per_sec, 2),
                "num_core_forward_passes": summary["num_core_forward_passes"],
                "num_tokens_forward": summary["num_tokens_forward"],
                "num_cache_clears": summary["num_cache_clears"],
                "diffusion_steps": summary["diffusion_steps"],
                "mean_recurrence": float(recurrence_per_pos.float().mean()),
                "max_recurrence": int(recurrence_per_pos.max()),
            }

            results[category][prompt] = entry

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    all_tok_per_sec = []
    all_forward_passes = []
    for category in results:
        print(f"\n--- {category.upper()} ---")
        for prompt in results[category]:
            entry = results[category][prompt]
            all_tok_per_sec.append(entry["tokens_per_sec"])
            all_forward_passes.append(entry["num_core_forward_passes"])
            print(f"\nPrompt: {prompt[:60]}...")
            print(f"  Tokens: {entry['num_tokens_generated']}, "
                  f"Time: {entry['wall_time_s']}s, "
                  f"Tok/s: {entry['tokens_per_sec']}")
            print(f"  Forward passes: {entry['num_core_forward_passes']}, "
                  f"Mean recurrence: {entry['mean_recurrence']:.1f}")
            print(f"  Output: {entry['generated_text'][:100]}...")

    if all_tok_per_sec:
        print(f"\n--- AGGREGATE ---")
        print(f"  Mean tokens/s: {sum(all_tok_per_sec)/len(all_tok_per_sec):.1f}")
        print(f"  Mean forward passes: {sum(all_forward_passes)/len(all_forward_passes):.0f}")

    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {output_file}")

    return results


if __name__ == "__main__":
    CLI(run_evaluation)
