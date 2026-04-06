from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import openai
"""
uv run python examples/experimental/negotiation/eval_batch_utilities.py \
  --input-path examples/experimental/negotiation/batch_20260329_184830 \
  --base-url "https://ai-gateway.andrew.cmu.edu" \
  --model "gpt-5-mini" \
  --reasoning-effort low
"""


ISSUES = ("Firewood", "Water", "Food")
PRIORITY_TO_POINTS = {"High": 5, "Medium": 4, "Low": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract final deals from continued negotiation dialogues and compute utilities."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Path to a single continued JSON file or a batch directory containing *_continued.json files.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional output JSON path. Defaults next to the input path.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        required=True,
        help="OpenAI-compatible gateway base URL.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name to use for deal extraction.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("LITELLM_PROXY_API_KEY", "")
        or os.environ.get("OPENAI_API_KEY", ""),
        help="Gateway API key. Defaults to LITELLM_PROXY_API_KEY or OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=None,
        help="Optional max completion tokens for the extraction call. If omitted, use the model/provider default.",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="low",
        choices=["low", "medium", "high"],
        help="Reasoning effort for GPT-5 style models.",
    )
    return parser.parse_args()


def list_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.glob("*_continued.json"))
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def default_output_path(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path.with_name(f"{input_path.stem}_eval.json")
    return input_path / "utility_eval_summary.json"


def build_transcript(dialogue: list[dict[str, Any]]) -> str:
    lines = []
    for turn in dialogue:
        speaker = turn["speaker"]
        turn_index = turn["turn_index"]
        text = turn["text"]
        lines.append(f"Turn {turn_index} | {speaker}: {text}")
    return "\n".join(lines)


def build_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    transcript = build_transcript(record["dialogue"])
    return [
        {
            "role": "system",
            "content": (
                "You extract final negotiated allocations from a two-agent camping-supplies dialogue. "
                "Return JSON only. Do not include markdown."
            ),
        },
        {
            "role": "user",
            "content": (
                "Read the negotiation transcript and infer the final agreed allocation only if a clear final deal "
                "is reached.\n\n"
                "There are exactly 3 Firewood, 3 Water, and 3 Food packages in total.\n"
                "Return strict JSON with this shape:\n"
                "{\n"
                '  "deal_reached": true or false,\n'
                '  "evidence_turn_index": integer or null,\n'
                '  "agent_allocations": {\n'
                '    "mturk_agent_1": {"Firewood": integer, "Water": integer, "Food": integer},\n'
                '    "mturk_agent_2": {"Firewood": integer, "Water": integer, "Food": integer}\n'
                "  },\n"
                '  "confidence": "high" | "medium" | "low",\n'
                '  "notes": "short explanation"\n'
                "}\n"
                "If no clear deal is reached, set deal_reached to false, evidence_turn_index to null, "
                "and all counts to 0.\n\n"
                f"Transcript:\n{transcript}"
            ),
        },
    ]


def extract_first_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in model response: {text!r}")
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : index + 1])
    raise ValueError(f"Incomplete JSON object in model response: {text!r}")


def robust_json_from_response_text(text: str) -> dict[str, Any]:
    stripped = text.strip()
    candidates = [stripped]

    if "```" in stripped:
        import re

        for match in re.findall(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL):
            candidates.append(match.strip())

    seen: set[str] = set()
    last_error: Exception | None = None
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            return extract_first_json_object(candidate)
        except Exception as error:
            last_error = error

    if last_error is not None:
        raise last_error
    raise ValueError(f"No JSON object found in model response: {text!r}")


def normalize_allocations(raw_allocations: dict[str, Any]) -> dict[str, dict[str, int]]:
    normalized: dict[str, dict[str, int]] = {}
    for agent_name in ("mturk_agent_1", "mturk_agent_2"):
        raw_agent = raw_allocations.get(agent_name, {})
        normalized[agent_name] = {
            issue: int(raw_agent.get(issue, 0)) for issue in ISSUES
        }
    return normalized


def validate_allocations(
    allocations: dict[str, dict[str, int]]
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for issue in ISSUES:
        total = sum(allocations[agent][issue] for agent in allocations)
        if total != 3:
            errors.append(f"{issue} totals {total}, expected 3")
    for agent_name, issue_map in allocations.items():
        for issue, count in issue_map.items():
            if count < 0 or count > 3:
                errors.append(f"{agent_name} has invalid {issue} count {count}")
    return (len(errors) == 0, errors)


def compute_utility(
    agent_payload: dict[str, Any], allocation: dict[str, int]
) -> dict[str, Any]:
    issue_points = {
        issue: PRIORITY_TO_POINTS[priority]
        for priority, issue in agent_payload["value2issue"].items()
    }
    breakdown = {
        issue: allocation[issue] * issue_points[issue] for issue in ISSUES
    }
    return {
        "allocation": allocation,
        "issue_points": issue_points,
        "breakdown": breakdown,
        "total_utility": sum(breakdown.values()),
    }


def make_client(base_url: str, api_key: str) -> openai.OpenAI:
    return openai.OpenAI(api_key=api_key, base_url=base_url)


def call_extractor(
    client: openai.OpenAI,
    model: str,
    record: dict[str, Any],
    max_completion_tokens: int | None,
    reasoning_effort: str,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": build_messages(record),
    }
    if max_completion_tokens is not None:
        kwargs["max_completion_tokens"] = max_completion_tokens
    if "gpt-5" in model:
        kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}
    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content or ""
    finish_reason = response.choices[0].finish_reason
    if not content.strip():
        usage = getattr(response, "usage", None)
        raise ValueError(
            "Model returned empty content before JSON parsing. "
            f"finish_reason={finish_reason!r}, usage={usage!r}. "
            "This often means the model spent the budget on reasoning or returned no visible text. "
            "If you explicitly set --max-completion-tokens, try a larger value or omit that flag entirely."
        )
    parsed = robust_json_from_response_text(content)
    parsed["raw_response"] = content
    parsed["finish_reason"] = finish_reason
    return parsed


def call_extractor_with_retry(
    client: openai.OpenAI,
    model: str,
    record: dict[str, Any],
    max_completion_tokens: int | None,
    reasoning_effort: str,
    max_attempts: int = 2,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return call_extractor(
                client=client,
                model=model,
                record=record,
                max_completion_tokens=max_completion_tokens,
                reasoning_effort=reasoning_effort,
            )
        except Exception as error:
            last_error = error
            if attempt < max_attempts:
                print(
                    f"  Extraction attempt {attempt} failed, retrying once: {error}"
                )
    assert last_error is not None
    raise last_error


def evaluate_record(
    record: dict[str, Any],
    extraction: dict[str, Any],
    source_file: Path,
) -> dict[str, Any]:
    deal_reached = bool(extraction.get("deal_reached", False))
    allocations = normalize_allocations(extraction.get("agent_allocations", {}))
    allocation_valid, allocation_errors = validate_allocations(allocations)

    utilities: dict[str, Any] = {}
    if deal_reached and allocation_valid:
        for agent_name in ("mturk_agent_1", "mturk_agent_2"):
            utilities[agent_name] = compute_utility(
                record["agents"][agent_name],
                allocations[agent_name],
            )
    else:
        for agent_name in ("mturk_agent_1", "mturk_agent_2"):
            utilities[agent_name] = {
                "allocation": allocations[agent_name],
                "issue_points": {
                    issue: PRIORITY_TO_POINTS[priority]
                    for priority, issue in record["agents"][agent_name]["value2issue"].items()
                },
                "breakdown": {issue: 0 for issue in ISSUES},
                "total_utility": 0,
            }

    return {
        "source_file": str(source_file),
        "dialogue_index": record["dialogue_index"],
        "dialogue_name": record["dialogue_name"],
        "perspective": record.get("perspective"),
        "deal_reached": deal_reached,
        "extraction_confidence": extraction.get("confidence"),
        "evidence_turn_index": extraction.get("evidence_turn_index"),
        "allocation_valid": allocation_valid,
        "allocation_errors": allocation_errors,
        "agent_allocations": allocations,
        "utilities": utilities,
        "extraction_notes": extraction.get("notes", ""),
        "raw_extraction_response": extraction.get("raw_response", ""),
        "finish_reason": extraction.get("finish_reason"),
    }


def build_failed_result(
    record: dict[str, Any],
    source_file: Path,
    error: Exception,
) -> dict[str, Any]:
    zero_allocations = {
        "mturk_agent_1": {issue: 0 for issue in ISSUES},
        "mturk_agent_2": {issue: 0 for issue in ISSUES},
    }
    utilities = {}
    for agent_name in ("mturk_agent_1", "mturk_agent_2"):
        utilities[agent_name] = {
            "allocation": zero_allocations[agent_name],
            "issue_points": {
                issue: PRIORITY_TO_POINTS[priority]
                for priority, issue in record["agents"][agent_name]["value2issue"].items()
            },
            "breakdown": {issue: 0 for issue in ISSUES},
            "total_utility": 0,
        }

    return {
        "source_file": str(source_file),
        "dialogue_index": record["dialogue_index"],
        "dialogue_name": record["dialogue_name"],
        "perspective": record.get("perspective"),
        "deal_reached": False,
        "extraction_confidence": None,
        "evidence_turn_index": None,
        "allocation_valid": False,
        "allocation_errors": ["Extraction failed"],
        "agent_allocations": zero_allocations,
        "utilities": utilities,
        "extraction_notes": "",
        "raw_extraction_response": "",
        "finish_reason": None,
        "status": "failed",
        "error": str(error),
    }


def aggregate_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    deal_count = sum(1 for item in results if item["deal_reached"])
    valid_count = sum(1 for item in results if item["deal_reached"] and item["allocation_valid"])
    failed_count = sum(1 for item in results if item.get("status") == "failed")
    return {
        "num_files": len(results),
        "num_deals_detected": deal_count,
        "num_valid_allocations": valid_count,
        "num_failed_extractions": failed_count,
        "results": results,
    }


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise ValueError("Missing API key. Pass --api-key or set LITELLM_PROXY_API_KEY.")

    input_files = list_input_files(args.input_path)
    client = make_client(args.base_url, args.api_key)
    results: list[dict[str, Any]] = []

    print(f"Evaluating {len(input_files)} files")
    print(f"Using base_url={args.base_url}")
    print(f"Using model={args.model}")

    for path in input_files:
        print(f"Processing {path}")
        record = json.loads(path.read_text(encoding="utf-8"))
        try:
            extraction = call_extractor_with_retry(
                client=client,
                model=args.model,
                record=record,
                max_completion_tokens=args.max_completion_tokens,
                reasoning_effort=args.reasoning_effort,
            )
            results.append(evaluate_record(record, extraction, path))
        except Exception as error:
            print(f"  Failed after retry: {error}")
            results.append(build_failed_result(record, path, error))

    summary = aggregate_summary(results)
    output_path = args.output_file or default_output_path(args.input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote evaluation summary to {output_path}")


if __name__ == "__main__":
    main()
