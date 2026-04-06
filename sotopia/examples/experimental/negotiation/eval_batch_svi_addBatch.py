from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import openai


SVI_14 = [
    {
        "qid": "Q1",
        "original_svi_id": 1,
        "text": "How satisfied are you with your own outcome—i.e., the extent to which the terms of your agreement (or lack of agreement) benefit you?",
        "scale": "1 to 7, 1 being Not at all, 4 being Moderately, 7 being Perfectly.",
    },
    {
        "qid": "Q2",
        "original_svi_id": 2,
        "text": "How satisfied are you with the balance between your own outcome and your counterpart's outcome?",
        "scale": "1 to 7, 1 being Not at all, 4 being Moderately, 7 being Perfectly.",
    },
    {
        "qid": "Q3",
        "original_svi_id": 3,
        "text": "Did you feel like you forfeited or 'lost' in this negotiation?",
        "scale": "1 to 7, 1 being Not at all, 4 being Moderately, 7 being A great deal.",
    },
    {
        "qid": "Q4",
        "original_svi_id": 4,
        "text": "Do you think the terms of your agreement are consistent with principles of legitimacy or objective criteria (e.g., fairness, precedent, standard practice, legality, etc.)?",
        "scale": "1 to 7, 1 being Not at all, 4 being Moderately, 7 being Perfectly.",
    },
    {
        "qid": "Q5",
        "original_svi_id": 5,
        "text": "Did you 'lose face' (i.e., damage your sense of pride) in the negotiation?",
        "scale": "1 to 7, 1 being Not at all, 4 being Moderately, 7 being A great deal.",
    },
    {
        "qid": "Q6",
        "original_svi_id": 6,
        "text": "Did this negotiation make you feel more or less competent as a negotiator?",
        "scale": "1 to 7, 1 being It made me feel less competent, 4 being It did not make me feel either more or less competent, 7 being It made me feel more competent.",
    },
    {
        "qid": "Q7",
        "original_svi_id": 9,
        "text": "Do you feel your counterpart listened to your concerns?",
        "scale": "1 to 7, 1 being Not at all, 4 being Moderately, 7 being Perfectly.",
    },
    {
        "qid": "Q8",
        "original_svi_id": 10,
        "text": "Would you characterize the negotiation process as fair?",
        "scale": "1 to 7, 1 being Not at all, 4 being Moderately, 7 being Perfectly.",
    },
    {
        "qid": "Q9",
        "original_svi_id": 11,
        "text": "How satisfied are you with the ease (or difficulty) of reaching an agreement?",
        "scale": "1 to 7, 1 being Not at all, 4 being Moderately, 7 being Perfectly.",
    },
    {
        "qid": "Q10",
        "original_svi_id": 12,
        "text": "Did your counterpart consider your wishes, opinions, or needs?",
        "scale": "1 to 7, 1 being Not at all, 4 being Moderately, 7 being Perfectly.",
    },
    {
        "qid": "Q11",
        "original_svi_id": 13,
        "text": "What kind of overall impression did your counterpart make on you?",
        "scale": "1 to 7, 1 being Extremely Negative, 4 being Neither Positive nor Negative, 7 being Extremely Positive.",
    },
    {
        "qid": "Q12",
        "original_svi_id": 14,
        "text": "How satisfied are you with your relationship with your counterpart as a result of this negotiation?",
        "scale": "1 to 7, 1 being Not at all, 4 being Moderately, 7 being Perfectly.",
    },
    {
        "qid": "Q13",
        "original_svi_id": 15,
        "text": "Did the negotiation make you trust your counterpart?",
        "scale": "1 to 7, 1 being Not at all, 4 being Moderately, 7 being Perfectly.",
    },
    {
        "qid": "Q14",
        "original_svi_id": 16,
        "text": "Did the negotiation build a good foundation for a future relationship with your counterpart?",
        "scale": "1 to 7, 1 being Not at all, 4 being Moderately, 7 being Perfectly.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 14-question SVI evaluation over continued negotiation JSON files."
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
        help="Model name to use for SVI scoring.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("LITELLM_PROXY_API_KEY", "")
        or os.environ.get("OPENAI_API_KEY", ""),
        help="Gateway API key. Defaults to LITELLM_PROXY_API_KEY or OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="low",
        choices=["low", "medium", "high"],
        help="Reasoning effort for GPT-5 style models.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=None,
        help="Optional max completion tokens. If omitted, use the provider default.",
    )
    parser.add_argument(
        "--sleep-between-calls",
        type=float,
        default=0.0,
        help="Optional sleep between successful API calls.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Number of concurrent SVI requests to run. Default 1 keeps serial behavior.",
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
        return input_path.with_name(f"{input_path.stem}_svi_eval.json")
    return input_path / "svi_eval_summary.json"


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def json_dumps_pretty(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def normalize_whitespace(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def remove_control_chars(text: str) -> str:
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)


def robust_json_loads(text: str) -> dict[str, Any]:
    raw = text.strip()
    candidates = [raw]

    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, flags=re.DOTALL)
    if fence_match:
        candidates.append(fence_match.group(1).strip())

    brace_match = re.search(r"(\{.*\})", raw, flags=re.DOTALL)
    if brace_match:
        candidates.append(brace_match.group(1).strip())

    tried: list[str] = []
    for candidate in candidates:
        for version in [
            candidate,
            remove_control_chars(candidate),
            remove_control_chars(normalize_whitespace(candidate)),
        ]:
            version = version.strip()
            if not version or version in tried:
                continue
            tried.append(version)
            try:
                return json.loads(version)
            except Exception:
                pass

    raise ValueError(f"Failed to parse model JSON. Raw preview:\n{raw[:2000]}")


def infer_ranked_preferences(participant: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    value2issue = participant.get("value2issue")
    value2reason = participant.get("value2reason")

    ranked_items: list[str] = []
    reason_map: dict[str, Any] = {}

    if isinstance(value2issue, dict):
        lower_map = {safe_str(k).lower(): v for k, v in value2issue.items()}
        high = lower_map.get("high")
        medium = lower_map.get("medium")
        low = lower_map.get("low")
        if high is not None:
            ranked_items.append(safe_str(high))
        if medium is not None:
            ranked_items.append(safe_str(medium))
        if low is not None:
            ranked_items.append(safe_str(low))

    if isinstance(value2reason, dict):
        lower_reason = {safe_str(k).lower(): v for k, v in value2reason.items()}
        for item in ranked_items:
            if item.lower() in lower_reason:
                reason_map[item] = lower_reason[item.lower()]
        rank_keys = ["high", "medium", "low"]
        for idx, item in enumerate(ranked_items[:3]):
            if item not in reason_map and rank_keys[idx] in lower_reason:
                reason_map[item] = lower_reason[rank_keys[idx]]

    return ranked_items, reason_map


def format_preference_section(participant: dict[str, Any]) -> str:
    ranked_items, reason_map = infer_ranked_preferences(participant)
    lines: list[str] = []
    if ranked_items:
        lines.append(
            "Your preference ranking over the three item types is: "
            + " > ".join(ranked_items)
            + "."
        )
        for item in ranked_items:
            reason = reason_map.get(item)
            if reason is not None and safe_str(reason):
                lines.append(f"- {item}: {safe_str(reason)}")
    else:
        lines.append("Your preference information is provided below in raw form.")
        lines.append(json_dumps_pretty(participant.get("value2issue")))
        if participant.get("value2reason") is not None:
            lines.append("Reasons for your preferences:")
            lines.append(json_dumps_pretty(participant.get("value2reason")))
    return "\n".join(lines)


def format_dialogue_as_you_them(
    dialogue: list[dict[str, Any]], pov_agent_id: str
) -> str:
    lines = []
    for turn in dialogue:
        speaker = safe_str(turn.get("speaker") or turn.get("id"))
        text = safe_str(turn.get("text"))
        text = remove_control_chars(normalize_whitespace(text))
        if not text:
            continue
        prefix = "YOU" if speaker == pov_agent_id else "THEM"
        lines.append(f"{prefix}: {text}")
    return "\n".join(lines)


def pretty_block(title: str, value: Any) -> str:
    if value is None:
        return f"{title}: Not provided."
    if isinstance(value, (dict, list)):
        return f"{title}:\n{json_dumps_pretty(value)}"
    return f"{title}: {safe_str(value)}"


def build_prompt(record: dict[str, Any], pov_agent_id: str) -> str:
    participant_info = record["agents"]
    self_info = participant_info[pov_agent_id]
    dialogue_text = format_dialogue_as_you_them(record["dialogue"], pov_agent_id)

    svi_question_lines = []
    for q in SVI_14:
        svi_question_lines.append(
            f'{q["qid"]} (original SVI item {q["original_svi_id"]}, scale {q["scale"]}): {q["text"]}'
        )
    svi_question_block = "\n".join(svi_question_lines)

    output_schema = {
        "svi_scores": {
            q["qid"]: {
                "original_svi_id": q["original_svi_id"],
                "score": "integer from 1 to 7",
                "reason": "brief explanation grounded in the dialogue and this participant's own profile",
            }
            for q in SVI_14
        },
        "brief_overall_reason": "2-4 sentences summarizing the participant's overall subjective experience",
    }

    prompt = f"""
You are evaluating a negotiation from the perspective of one participant.

Scenario:
You are negotiating with another camper over three item types: food, water, and firewood.
There are exactly 3 units available for each item type.
The two of you are trying to decide how to distribute these items through conversation.

Your job:
Read the entire dialogue from THIS participant's point of view.
Then estimate this participant's ratings on the 14-question SVI questionnaire.

Important constraints:
- Stay strictly in this participant's perspective.
- Use only this participant's own profile and the dialogue.
- Do NOT use the opponent's profile.
- Do NOT use outcomes.
- Even if the conversation is incomplete or ambiguous, make the best possible SVI judgment from this participant's perspective.
- For each question, give:
  1) a score from 1 to 7
  2) a short reason
- Return valid JSON only.
- Do not include markdown fences.
- Do not include any text before or after the JSON.

Participant profile:
{pretty_block("Personality", self_info.get("personality"))}

{pretty_block("Demographics", self_info.get("demographics"))}

Preference information:
{format_preference_section(self_info)}

Dialogue transcript:
{dialogue_text}

SVI questions to answer:
{svi_question_block}

Return JSON in exactly this format:
{json_dumps_pretty(output_schema)}
""".strip()

    return remove_control_chars(normalize_whitespace(prompt))


def validate_prediction(pred: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(pred, dict):
        raise ValueError("Prediction is not a dict.")
    if "svi_scores" not in pred:
        raise ValueError("Missing 'svi_scores'.")

    svi_scores = pred["svi_scores"]
    if not isinstance(svi_scores, dict):
        raise ValueError("'svi_scores' must be a dict.")

    cleaned_scores = {}
    for q in SVI_14:
        qid = q["qid"]
        if qid not in svi_scores:
            raise ValueError(f"Missing question {qid}.")
        item = svi_scores[qid]
        if not isinstance(item, dict):
            raise ValueError(f"{qid} must be an object.")

        score = item.get("score")
        reason = item.get("reason", "")
        original_svi_id = item.get("original_svi_id", q["original_svi_id"])

        if isinstance(score, str):
            match = re.search(r"\d+", score)
            if not match:
                raise ValueError(f"Invalid score for {qid}: {score}")
            score = int(match.group())
        elif isinstance(score, (int, float)):
            score = int(score)
        else:
            raise ValueError(f"Invalid score type for {qid}: {type(score)}")

        score = min(7, max(1, score))
        cleaned_scores[qid] = {
            "original_svi_id": int(original_svi_id),
            "score": score,
            "reason": safe_str(reason),
        }

    return {
        "svi_scores": cleaned_scores,
        "brief_overall_reason": safe_str(pred.get("brief_overall_reason", "")),
    }


def make_client(base_url: str, api_key: str) -> openai.OpenAI:
    return openai.OpenAI(api_key=api_key, base_url=base_url)


def query_model(
    client: openai.OpenAI,
    model: str,
    prompt: str,
    reasoning_effort: str,
    max_completion_tokens: int | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a careful evaluator of negotiation experience. "
                    "Return strict JSON only. Scores must be integers from 1 to 7."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }
    if max_completion_tokens is not None:
        kwargs["max_completion_tokens"] = max_completion_tokens
    if "gpt-5" in model:
        kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}

    response = client.chat.completions.create(**kwargs)
    text = response.choices[0].message.content or ""
    if not text.strip():
        usage = getattr(response, "usage", None)
        raise ValueError(
            "Model returned empty content before JSON parsing. "
            f"finish_reason={response.choices[0].finish_reason!r}, usage={usage!r}."
        )
    parsed = robust_json_loads(text)
    validated = validate_prediction(parsed)
    validated["raw_response"] = text
    validated["finish_reason"] = response.choices[0].finish_reason
    return validated


def query_model_with_retry(
    client: openai.OpenAI,
    model: str,
    prompt: str,
    reasoning_effort: str,
    max_completion_tokens: int | None,
    max_attempts: int = 2,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return query_model(
                client=client,
                model=model,
                prompt=prompt,
                reasoning_effort=reasoning_effort,
                max_completion_tokens=max_completion_tokens,
            )
        except Exception as error:
            last_error = error
            if attempt < max_attempts:
                print(f"    POV retry after failure: {error}")
    assert last_error is not None
    raise last_error


def evaluate_pov(
    client: openai.OpenAI,
    model: str,
    record: dict[str, Any],
    source_file: Path,
    pov_agent_id: str,
    reasoning_effort: str,
    max_completion_tokens: int | None,
) -> dict[str, Any]:
    prompt = build_prompt(record, pov_agent_id)
    prediction = query_model_with_retry(
        client=client,
        model=model,
        prompt=prompt,
        reasoning_effort=reasoning_effort,
        max_completion_tokens=max_completion_tokens,
    )
    return {
        "source_file": str(source_file),
        "dialogue_index": record["dialogue_index"],
        "dialogue_name": record["dialogue_name"],
        "intervention_perspective": record.get("perspective"),
        "pov_agent_id": pov_agent_id,
        "status": "ok",
        "prediction": {
            "svi_scores": prediction["svi_scores"],
            "brief_overall_reason": prediction["brief_overall_reason"],
        },
        "raw_response": prediction.get("raw_response", ""),
        "finish_reason": prediction.get("finish_reason"),
    }


def evaluate_pov_task(
    *,
    base_url: str,
    api_key: str,
    model: str,
    record: dict[str, Any],
    source_file: Path,
    pov_agent_id: str,
    reasoning_effort: str,
    max_completion_tokens: int | None,
    sleep_between_calls: float,
) -> dict[str, Any]:
    client = make_client(base_url, api_key)
    result = evaluate_pov(
        client=client,
        model=model,
        record=record,
        source_file=source_file,
        pov_agent_id=pov_agent_id,
        reasoning_effort=reasoning_effort,
        max_completion_tokens=max_completion_tokens,
    )
    if sleep_between_calls > 0:
        time.sleep(sleep_between_calls)
    return result


def build_failed_result(
    record: dict[str, Any],
    source_file: Path,
    pov_agent_id: str,
    error: Exception,
) -> dict[str, Any]:
    return {
        "source_file": str(source_file),
        "dialogue_index": record["dialogue_index"],
        "dialogue_name": record["dialogue_name"],
        "intervention_perspective": record.get("perspective"),
        "pov_agent_id": pov_agent_id,
        "status": "failed",
        "error": str(error),
    }


def aggregate_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_rows": len(results),
        "num_successful_rows": sum(1 for item in results if item["status"] == "ok"),
        "num_failed_rows": sum(1 for item in results if item["status"] == "failed"),
        "results": results,
    }


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise ValueError("Missing API key. Pass --api-key or set LITELLM_PROXY_API_KEY.")

    input_files = list_input_files(args.input_path)
    results: list[dict[str, Any]] = []

    print(f"Evaluating SVI for {len(input_files)} files")
    print(f"Using base_url={args.base_url}")
    print(f"Using model={args.model}")
    print(f"Using max_concurrency={args.max_concurrency}")

    tasks: list[tuple[Path, dict[str, Any], str]] = []
    for path in input_files:
        record = json.loads(path.read_text(encoding="utf-8"))
        for pov_agent_id in ("mturk_agent_1", "mturk_agent_2"):
            tasks.append((path, record, pov_agent_id))

    if args.max_concurrency <= 1:
        for path, record, pov_agent_id in tasks:
            print(f"Processing {path}")
            print(f"  POV {pov_agent_id}")
            try:
                results.append(
                    evaluate_pov_task(
                        base_url=args.base_url,
                        api_key=args.api_key,
                        model=args.model,
                        record=record,
                        source_file=path,
                        pov_agent_id=pov_agent_id,
                        reasoning_effort=args.reasoning_effort,
                        max_completion_tokens=args.max_completion_tokens,
                        sleep_between_calls=args.sleep_between_calls,
                    )
                )
            except Exception as error:
                print(f"    Failed: {error}")
                results.append(build_failed_result(record, path, pov_agent_id, error))
    else:
        future_to_task = {}
        with ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
            for path, record, pov_agent_id in tasks:
                print(f"Queueing {path} | POV {pov_agent_id}")
                future = executor.submit(
                    evaluate_pov_task,
                    base_url=args.base_url,
                    api_key=args.api_key,
                    model=args.model,
                    record=record,
                    source_file=path,
                    pov_agent_id=pov_agent_id,
                    reasoning_effort=args.reasoning_effort,
                    max_completion_tokens=args.max_completion_tokens,
                    sleep_between_calls=args.sleep_between_calls,
                )
                future_to_task[future] = (path, record, pov_agent_id)

            for future in as_completed(future_to_task):
                path, record, pov_agent_id = future_to_task[future]
                try:
                    results.append(future.result())
                    print(f"Completed {path} | POV {pov_agent_id}")
                except Exception as error:
                    print(f"Failed {path} | POV {pov_agent_id}: {error}")
                    results.append(build_failed_result(record, path, pov_agent_id, error))

    summary = aggregate_summary(results)
    output_path = args.output_file or default_output_path(args.input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote SVI summary to {output_path}")


if __name__ == "__main__":
    main()
