#!/usr/bin/env python3
"""
PersonaBEAM Cross-Model Inference Script
=========================================
Feeds fisheye images x persona prompts to multiple VLMs and collects
(command, reasoning) outputs in a unified CSV format.

Models supported:
  - GPT-5.5          (OpenAI API)
  - Gemini 3.1 Pro   (Vertex AI / Google GenAI API)
  - Claude Opus 4.7  (Anthropic API)
  - Qwen3.6-35B-A3B  (local vLLM server)
  - Gemma 4 31B      (local Ollama server)

Usage:
  # Run all API models on sampled images:
  python run_inference.py --image_dir ./personabeam/images \\
      --models gpt55 gemini claude --sample_per_env 200

  # Run local model (assumes vLLM server running on localhost:8000):
  python run_inference.py --image_dir ./personabeam/images \\
      --models qwen --vllm_url http://localhost:8000/v1 --sample_per_env 200

  # Run all five models:
  python run_inference.py --image_dir ./personabeam/images \\
      --models gpt55 gemini claude qwen gemma \\
      --sample_per_env 200

Environment variables for API keys:
  OPENAI_API_KEY, GOOGLE_CLOUD_PROJECT, ANTHROPIC_API_KEY
"""

import argparse
import base64
import csv
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Persona definitions (identical prompts used for all models)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are controlling a helium-filled blimp robot equipped with a fisheye camera.
The blimp navigates indoor environments. Based on the fisheye image provided,
decide on a single directional command for the blimp.

Available commands:
  F = Forward    R = Reverse    L = Left turn    T = Right turn
  U = Up         D = Down       S = Stop

You MUST respond with ONLY a valid JSON object (no markdown, no extra text):
{{"command": "<one of F,R,L,T,U,D,S>", "reason": "<brief explanation>"}}
"""

# ---------------------------------------------------------------------------
# Persona design rationale
# ---------------------------------------------------------------------------
# The three personas are grounded in the Behavioral Inhibition System /
# Behavioral Activation System (BIS/BAS) framework (Gray, 1982;
# Carver & White, 1994), spanning a well-defined behavioral axis:
#
#   Companion  = High BAS (Behavioral Activation System) / Approach
#   Explorer   = Balanced BAS/BIS / Neutral
#   Observer   = High BIS (Behavioral Inhibition System) / Avoidance
#
# Each prompt specifies four parallel facets:
#   1. Core disposition (approach / neutral / avoidance)
#   2. Social orientation (toward people / indifferent / away from people)
#   3. Movement preference (forward-seeking / systematic / retreat-oriented)
#   4. Reasoning style (enthusiastic / detached / cautious)
# ---------------------------------------------------------------------------

PERSONA_PROMPTS = {
    "companion": (
        "Your persona: EAGER COMPANION (approach-oriented). "
        "Core disposition: You are drawn toward activity and engagement. "
        "Social orientation: Move toward people, approach interesting objects, seek interaction. "
        "Movement preference: Favor forward and upward motions; advance rather than retreat. "
        "Reasoning style: Express enthusiasm and curiosity in your explanations."
    ),
    "observer": (
        "Your persona: CAUTIOUS OBSERVER (avoidance-oriented). "
        "Core disposition: You prioritize safety and maintain distance from uncertainty. "
        "Social orientation: Keep away from people, avoid crowded or unfamiliar areas. "
        "Movement preference: Favor reorientation and small backward steps; retreat rather than advance. "
        "Reasoning style: Express caution and vigilance in your explanations."
    ),
    "explorer": (
        "Your persona: INDIFFERENT EXPLORER (neutral). "
        "Core disposition: You are methodical and detached, neither drawn to nor repelled by stimuli. "
        "Social orientation: Treat people and objects as features of the environment, without preference. "
        "Movement preference: Move in straight lines, scan systematically; balance all directions equally. "
        "Reasoning style: Be objective and matter-of-fact in your explanations."
    ),
}

VALID_COMMANDS = {"F", "R", "L", "T", "U", "D", "S"}

# ---------------------------------------------------------------------------
# Environment mapping from folder names to canonical labels
# ---------------------------------------------------------------------------

ENV_LABELS = {
    "auditorium": "auditorium",
    "institutional_hallway": "institutional_hallway",
    "furnished_lounge": "furnished_lounge",
    "domestic_room": "domestic_room",
}

# ---------------------------------------------------------------------------
# Image sampling
# ---------------------------------------------------------------------------

def sample_images(image_dir: str, sample_per_env: int, seed: int = 42) -> list:
    """
    Sample up to `sample_per_env` images from each environment subfolder.
    Returns list of (image_path, environment_name) tuples.
    """
    random.seed(seed)
    samples = []
    image_dir = Path(image_dir)

    for folder in sorted(image_dir.iterdir()):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        env_name = ENV_LABELS.get(folder.name, folder.name)
        images = sorted(folder.glob("*.jpg"))
        if not images:
            images = sorted(folder.glob("*.png"))
        if not images:
            print(f"  [WARN] No images found in {folder.name}, skipping")
            continue

        if len(images) <= sample_per_env:
            selected = images
        else:
            selected = random.sample(images, sample_per_env)

        for img_path in selected:
            rel_path = str(img_path.relative_to(image_dir.parent))
            samples.append((str(img_path), env_name, rel_path))
        print(f"  {folder.name}: {len(selected)}/{len(images)} images selected")

    envs_found = len(set(s[1] for s in samples))
    print(f"  Total: {len(samples)} images across {envs_found} environments")
    return samples


def encode_image_base64(image_path: str) -> str:
    """Read an image file and return base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def parse_json_response(text: str) -> dict:
    """
    Extract command and reason from model response.
    Handles markdown code blocks, extra text, partial JSON, thinking blocks.
    """
    if not text:
        return {"command": "PARSE_ERROR", "reason": "(empty response)"}

    # Strip <think>...</think> blocks (thinking-mode models)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Strip markdown code fences
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    if not text:
        return {"command": "PARSE_ERROR", "reason": "(only thinking tokens, no visible output)"}

    # Try direct JSON parse
    try:
        obj = json.loads(text)
        cmd = str(obj.get("command", "")).strip().upper()
        reason = str(obj.get("reason", "")).strip()
        if cmd in VALID_COMMANDS:
            return {"command": cmd, "reason": reason}
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in text
    match = re.search(
        r'\{[^}]*"command"\s*:\s*"([^"]+)"[^}]*"reason"\s*:\s*"([^"]*)"[^}]*\}',
        text, re.IGNORECASE,
    )
    if match:
        cmd = match.group(1).strip().upper()
        reason = match.group(2).strip()
        if cmd in VALID_COMMANDS:
            return {"command": cmd, "reason": reason}

    # Try reversed key order
    match = re.search(
        r'\{[^}]*"reason"\s*:\s*"([^"]*)"[^}]*"command"\s*:\s*"([^"]+)"[^}]*\}',
        text, re.IGNORECASE,
    )
    if match:
        cmd = match.group(2).strip().upper()
        reason = match.group(1).strip()
        if cmd in VALID_COMMANDS:
            return {"command": cmd, "reason": reason}

    # Last resort: find any valid command letter
    for c in VALID_COMMANDS:
        if f'"{c}"' in text.upper() or f"'{c}'" in text.upper():
            return {"command": c, "reason": text[:200]}

    return {"command": "PARSE_ERROR", "reason": text[:300]}


# ---------------------------------------------------------------------------
# Model-specific inference functions
# ---------------------------------------------------------------------------

def call_gpt55(image_b64: str, persona_prompt: str, api_key: str) -> dict:
    """Call GPT-5.5 via OpenAI API."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + persona_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
                {"type": "text", "text": "Based on this fisheye camera view, decide the next command."},
            ],
        },
    ]

    response = client.chat.completions.create(
        model="gpt-5.5",
        messages=messages,
        max_completion_tokens=256,
        temperature=0.0,
        reasoning_effort="none",
    )
    msg = response.choices[0].message
    text = msg.content or ""
    if not text.strip():
        return {"command": "EMPTY_RESPONSE", "reason": "Model returned no content"}
    return parse_json_response(text)


def call_gemini(image_b64: str, persona_prompt: str) -> dict:
    """Call Gemini 3.1 Pro via Vertex AI using the google-genai SDK."""
    from google import genai
    from google.genai import types

    client = genai.Client(
        vertexai=True,
        project=os.environ.get("GOOGLE_CLOUD_PROJECT", ""),
        location="global",
    )

    image_bytes = base64.b64decode(image_b64)
    full_prompt = (
        SYSTEM_PROMPT + "\n\n" + persona_prompt
        + "\n\nBased on this fisheye camera view, decide the next command."
    )

    response = client.models.generate_content(
        model="gemini-3.1-pro-preview",
        contents=[
            full_prompt,
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
        ],
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=4096,
            thinking_config=types.ThinkingConfig(thinking_level="LOW"),
        ),
    )
    text = response.text
    if text is None:
        try:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    text = part.text
                    break
        except (IndexError, AttributeError):
            pass
    if not text:
        return {"command": "EMPTY_RESPONSE", "reason": "Gemini returned no text content"}
    return parse_json_response(text)


def call_claude(image_b64: str, persona_prompt: str, api_key: str) -> dict:
    """Call Claude Opus 4.7 via Anthropic API."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=SYSTEM_PROMPT + "\n\n" + persona_prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Based on this fisheye camera view, decide the next command.",
                    },
                ],
            }
        ],
    )
    return parse_json_response(message.content[0].text)


def call_vllm(image_b64: str, persona_prompt: str, vllm_url: str, model_name: str) -> dict:
    """
    Call a local model via OpenAI-compatible API (vLLM or Ollama).
    """
    from openai import OpenAI
    client = OpenAI(base_url=vllm_url, api_key="not-needed")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + persona_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
                {"type": "text", "text": "Based on this fisheye camera view, decide the next command."},
            ],
        },
    ]

    extra_body = {}
    if "qwen" in model_name.lower():
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=1024,
        temperature=0.0,
        extra_body=extra_body if extra_body else None,
    )
    content = response.choices[0].message.content or ""
    if not content.strip():
        return {"command": "EMPTY_RESPONSE", "reason": "Model returned no content"}
    return parse_json_response(content)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

MODEL_DISPLAY_NAMES = {
    "gpt55": "GPT-5.5",
    "gemini": "Gemini-3.1-Pro",
    "claude": "Claude-Opus-4.7",
    "qwen": "Qwen3.6-35B-A3B",
    "gemma": "Gemma-4-31B",
}


def call_model(model_key: str, image_b64: str, persona_prompt: str, args) -> dict:
    """Dispatch to the appropriate model API."""
    if model_key == "gpt55":
        return call_gpt55(image_b64, persona_prompt, os.environ["OPENAI_API_KEY"])
    elif model_key == "gemini":
        return call_gemini(image_b64, persona_prompt)
    elif model_key == "claude":
        return call_claude(image_b64, persona_prompt, os.environ["ANTHROPIC_API_KEY"])
    elif model_key == "qwen":
        url = args.vllm_url_qwen or args.vllm_url or "http://localhost:8000/v1"
        return call_vllm(image_b64, persona_prompt, url, "Qwen/Qwen3.6-35B-A3B-FP8")
    elif model_key == "gemma":
        url = args.vllm_url_gemma or args.vllm_url or "http://localhost:11434/v1"
        return call_vllm(image_b64, persona_prompt, url, "gemma4:31b")
    else:
        raise ValueError(f"Unknown model: {model_key}")


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_inference(args):
    print(f"\n{'='*60}")
    print("PersonaBEAM Cross-Model Inference")
    print(f"{'='*60}")
    print(f"Image dir:    {args.image_dir}")
    print(f"Models:       {', '.join(args.models)}")
    print(f"Sample/env:   {args.sample_per_env}")
    print(f"Output dir:   {args.output_dir}")
    print(f"{'='*60}\n")

    # 1. Sample images
    print("[Step 1] Sampling images...")
    samples = sample_images(args.image_dir, args.sample_per_env, args.seed)
    if not samples:
        print("ERROR: No images found. Check --image_dir path.")
        sys.exit(1)

    total_calls = len(samples) * len(PERSONA_PROMPTS) * len(args.models)
    print(
        f"\nTotal inference calls: {len(samples)} images x "
        f"{len(PERSONA_PROMPTS)} personas x {len(args.models)} models = {total_calls}\n"
    )

    # 2. Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Run inference per model
    for model_key in args.models:
        model_name = MODEL_DISPLAY_NAMES[model_key]
        csv_path = output_dir / f"results_{model_key}.csv"

        # Check for existing results to enable resume
        completed = set()
        if csv_path.exists() and args.resume:
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = (row["image_path"], row["persona"])
                    completed.add(key)
            print(f"[{model_name}] Resuming -- {len(completed)} calls already done")

        print(f"\n{'─'*50}")
        print(f"[{model_name}] Starting inference...")
        print(f"{'─'*50}")

        write_header = not csv_path.exists() or not args.resume
        csvfile = open(
            csv_path, "a" if args.resume and csv_path.exists() else "w", newline=""
        )
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "image_path", "environment", "persona", "model",
                "command", "reason", "timestamp", "latency_ms",
            ],
        )
        if write_header:
            writer.writeheader()

        done = 0
        errors = 0
        t_start = time.time()

        for img_path, env_name, rel_path in samples:
            image_b64 = encode_image_base64(img_path)

            for persona_key, persona_prompt in PERSONA_PROMPTS.items():
                if (rel_path, persona_key) in completed:
                    done += 1
                    continue

                t0 = time.time()
                try:
                    result = call_model(model_key, image_b64, persona_prompt, args)
                    latency = int((time.time() - t0) * 1000)

                    if result["command"] in ("PARSE_ERROR", "EMPTY_RESPONSE"):
                        errors += 1
                        if errors <= 5:
                            print(
                                f"  [PARSE_FAIL] {persona_key}/{os.path.basename(img_path)}: "
                                f"cmd={result['command']}, reason={result['reason'][:120]}",
                                file=sys.stderr,
                            )

                    writer.writerow({
                        "image_path": rel_path,
                        "environment": env_name,
                        "persona": persona_key,
                        "model": model_name,
                        "command": result["command"],
                        "reason": result["reason"],
                        "timestamp": datetime.now().isoformat(),
                        "latency_ms": latency,
                    })
                    csvfile.flush()

                except Exception as e:
                    latency = int((time.time() - t0) * 1000)
                    errors += 1
                    writer.writerow({
                        "image_path": rel_path,
                        "environment": env_name,
                        "persona": persona_key,
                        "model": model_name,
                        "command": "API_ERROR",
                        "reason": str(e)[:200],
                        "timestamp": datetime.now().isoformat(),
                        "latency_ms": latency,
                    })
                    csvfile.flush()

                    if errors <= 3:
                        print(f"  [ERROR] {persona_key}/{os.path.basename(img_path)}: {e}")
                    if errors == 3:
                        print("  [WARN] Suppressing further error messages...")

                    if "rate" in str(e).lower() or "429" in str(e):
                        print("  [RATE LIMIT] Sleeping 30s...")
                        time.sleep(30)
                    else:
                        time.sleep(1)

                done += 1
                if done % 50 == 0:
                    elapsed = time.time() - t_start
                    total_for_model = len(samples) * len(PERSONA_PROMPTS)
                    eta = (elapsed / done) * (total_for_model - done)
                    print(
                        f"  [{model_name}] {done}/{total_for_model} "
                        f"({done / total_for_model * 100:.1f}%) | "
                        f"Errors: {errors} | ETA: {eta / 60:.1f}min"
                    )

        csvfile.close()
        elapsed = time.time() - t_start
        total_for_model = len(samples) * len(PERSONA_PROMPTS)
        print(
            f"\n  [{model_name}] DONE: {total_for_model} calls in {elapsed / 60:.1f}min "
            f"({errors} errors)"
        )
        print(f"  Results saved to: {csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PersonaBEAM cross-model inference pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # API models only:
  python run_inference.py --image_dir ./personabeam/images \\
      --models gpt55 gemini claude --sample_per_env 200

  # Local Qwen via vLLM:
  python run_inference.py --image_dir ./personabeam/images \\
      --models qwen --vllm_url http://localhost:8000/v1

  # All models:
  python run_inference.py --image_dir ./personabeam/images \\
      --models gpt55 gemini claude qwen gemma \\
      --sample_per_env 200
        """,
    )

    parser.add_argument(
        "--image_dir", type=str, required=True,
        help="Path to images directory with environment subfolders",
    )
    parser.add_argument(
        "--models", nargs="+", required=True,
        choices=["gpt55", "gemini", "claude", "qwen", "gemma"],
        help="Which models to run",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs",
        help="Output directory for result CSVs (default: ./outputs)",
    )
    parser.add_argument(
        "--sample_per_env", type=int, default=200,
        help="Max images to sample per environment (default: 200)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from existing partial results")

    # Server URLs for local models
    parser.add_argument("--vllm_url", type=str, default=None, help="Default server URL for local models")
    parser.add_argument("--vllm_url_qwen", type=str, default=None, help="vLLM server URL for Qwen")
    parser.add_argument("--vllm_url_gemma", type=str, default=None, help="Ollama server URL for Gemma")

    args = parser.parse_args()

    # Validate API keys for selected models
    api_models = {"gpt55": "OPENAI_API_KEY", "claude": "ANTHROPIC_API_KEY"}
    for m in args.models:
        if m in api_models and not os.environ.get(api_models[m]):
            print(f"ERROR: {api_models[m]} not set (required for {MODEL_DISPLAY_NAMES[m]})")
            sys.exit(1)
    if "gemini" in args.models and not os.environ.get("GOOGLE_CLOUD_PROJECT"):
        print("ERROR: GOOGLE_CLOUD_PROJECT not set (required for Gemini via Vertex AI)")
        sys.exit(1)

    run_inference(args)


if __name__ == "__main__":
    main()
