import os
import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import openai


# =========================
# Config
# =========================
MAX_DIALOGUES = 50

PARQUET_PATH = "/home/swang4/866/data/train-00000-of-00001.parquet"
OUTPUT_PATH = f"/home/swang4/866/Baseline_2/casino_plan2_svi_baseline_first{MAX_DIALOGUES}.json"
MODEL_NAME = "gpt-5-mini"
TEMPERATURE = 0.2
SLEEP_BETWEEN_CALLS = 0.3

client = openai.OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://ai-gateway.andrew.cmu.edu"
)


# =========================
# SVI 14 questions
# Keep original 1-6 and 9-16
# Drop original 7 and 8
# =========================
SVI_14 = [
    {
        "qid": "Q1",
        "original_svi_id": 1,
        "text": "How satisfied are you with your own outcome—i.e., the extent to which the terms of your agreement (or lack of agreement) benefit you?",
        "scale": "1 to 7, 1 being Not at all, 4 being Moderately, 7 being Perfectly."
    },
    {
        "qid": "Q2",
        "original_svi_id": 2,
        "text": "How satisfied are you with the balance between your own outcome and your counterpart's outcome?",
        "scale": "1 to 7, 1 being Not at all, 4 being Moderately, 7 being Perfectly."
    },
    {
        "qid": "Q3",
        "original_svi_id": 3,
        "text": "Did you feel like you forfeited or 'lost' in this negotiation?",
        "scale": "1 to 7, 1 being Not at all, 4 being Moderately, 7 being A great deal."
    },
    {
        "qid": "Q4",
        "original_svi_id": 4,
        "text": "Do you think the terms of your agreement are consistent with principles of legitimacy or objective criteria (e.g., fairness, precedent, standard practice, legality, etc.)?",
        "scale": "1 to 7, 1 being Not at all, 4 being Moderately, 7 being Perfectly."
    },
    {
        "qid": "Q5",
        "original_svi_id": 5,
        "text": "Did you 'lose face' (i.e., damage your sense of pride) in the negotiation?",
        "scale": "1 to 7, 1 being Not at all, 4 being Moderately, 7 being A great deal."
    },
    {
        "qid": "Q6",
        "original_svi_id": 6,
        "text": "Did this negotiation make you feel more or less competent as a negotiator?",
        "scale": "1 to 7, 1 being It made me feel less competent, 4 being It did not make me feel either more or less competent, 7 being It made me feel more competent."
    },
    {
        "qid": "Q7",
        "original_svi_id": 9,
        "text": "Do you feel your counterpart listened to your concerns?",
        "scale": "1 to 7, 1 being Not at all, 4 being Moderately, 7 being Perfectly."
    },
    {
        "qid": "Q8",
        "original_svi_id": 10,
        "text": "Would you characterize the negotiation process as fair?",
        "scale": "1 to 7, 1 being Not at all, 4 being Moderately, 7 being Perfectly."
    },
    {
        "qid": "Q9",
        "original_svi_id": 11,
        "text": "How satisfied are you with the ease (or difficulty) of reaching an agreement?",
        "scale": "1 to 7, 1 being Not at all, 4 being Moderately, 7 being Perfectly."
    },
    {
        "qid": "Q10",
        "original_svi_id": 12,
        "text": "Did your counterpart consider your wishes, opinions, or needs?",
        "scale": "1 to 7, 1 being Not at all, 4 being Moderately, 7 being Perfectly."
    },
    {
        "qid": "Q11",
        "original_svi_id": 13,
        "text": "What kind of overall impression did your counterpart make on you?",
        "scale": "1 to 7, 1 being Extremely Negative, 4 being Neither Positive nor Negative, 7 being Extremely Positive."
    },
    {
        "qid": "Q12",
        "original_svi_id": 14,
        "text": "How satisfied are you with your relationship with your counterpart as a result of this negotiation?",
        "scale": "1 to 7, 1 being Not at all, 4 being Moderately, 7 being Perfectly."
    },
    {
        "qid": "Q13",
        "original_svi_id": 15,
        "text": "Did the negotiation make you trust your counterpart?",
        "scale": "1 to 7, 1 being Not at all, 4 being Moderately, 7 being Perfectly."
    },
    {
        "qid": "Q14",
        "original_svi_id": 16,
        "text": "Did the negotiation build a good foundation for a future relationship with your counterpart?",
        "scale": "1 to 7, 1 being Not at all, 4 being Moderately, 7 being Perfectly."
    },
]


# =========================
# Helpers
# =========================
def to_python(obj: Any) -> Any:
    """Convert pandas / numpy / pyarrow objects into plain Python types."""
    if hasattr(obj, "as_py"):
        return to_python(obj.as_py())

    if isinstance(obj, dict):
        return {str(k): to_python(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [to_python(x) for x in obj]

    if isinstance(obj, tuple):
        return [to_python(x) for x in obj]

    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    return obj


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    return str(x).strip()


def json_dumps_pretty(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def remove_control_chars(text: str) -> str:
    """
    Remove illegal control characters that often break json.loads.
    Keep \n \r \t, remove the rest of ASCII control chars.
    """
    if not isinstance(text, str):
        return text
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)


def normalize_whitespace(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def robust_json_loads(text: str) -> Dict[str, Any]:
    """
    Best-effort JSON parser:
    1) direct parse
    2) parse fenced ```json block
    3) parse first {...} span
    4) sanitize control chars and retry
    """
    if not isinstance(text, str):
        raise ValueError("Model response is not a string.")

    raw = text.strip()

    candidates = [raw]

    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, flags=re.DOTALL)
    if fence_match:
        candidates.append(fence_match.group(1).strip())

    brace_match = re.search(r"(\{.*\})", raw, flags=re.DOTALL)
    if brace_match:
        candidates.append(brace_match.group(1).strip())

    tried = []
    for candidate in candidates:
        for version in [candidate, remove_control_chars(candidate), remove_control_chars(normalize_whitespace(candidate))]:
            version = version.strip()
            if not version or version in tried:
                continue
            tried.append(version)
            try:
                return json.loads(version)
            except Exception:
                pass

    preview = raw[:2000]
    raise ValueError(f"Failed to parse model JSON. Raw preview:\n{preview}")


def get_sorted_participants(participant_info: Dict[str, Any]) -> List[str]:
    keys = list(participant_info.keys())

    def sort_key(x: str):
        m = re.search(r"(\d+)$", x)
        if m:
            return (0, int(m.group(1)), x)
        return (1, 999999, x)

    return sorted(keys, key=sort_key)


def pretty_block(title: str, value: Any) -> str:
    value = to_python(value)
    if value is None:
        return f"{title}: Not provided."
    if isinstance(value, (dict, list)):
        return f"{title}:\n{json_dumps_pretty(value)}"
    return f"{title}: {safe_str(value)}"


# =========================
# Preference formatting
# =========================
def infer_ranked_preferences(participant: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    """
    Try to recover ranking and reasons from CaSiNo-style fields.
    Handles common variants like:
      value2issue = {"High": "Food", "Medium": "Water", "Low": "Firewood"}
      value2reason = {"Food": "...", "Water": "...", "Firewood": "..."}
    """
    value2issue = to_python(participant.get("value2issue"))
    value2reason = to_python(participant.get("value2reason"))

    ranked_items: List[str] = []
    reason_map: Dict[str, Any] = {}

    if isinstance(value2issue, dict):
        lower_map = {safe_str(k).lower(): v for k, v in value2issue.items()}

        high = lower_map.get("high") or lower_map.get("highest") or lower_map.get("top") or lower_map.get("1")
        medium = lower_map.get("medium") or lower_map.get("mid") or lower_map.get("2")
        low = lower_map.get("low") or lower_map.get("lowest") or lower_map.get("3")

        if high is not None or medium is not None or low is not None:
            if high is not None:
                ranked_items.append(safe_str(high))
            if medium is not None:
                ranked_items.append(safe_str(medium))
            if low is not None:
                ranked_items.append(safe_str(low))
        else:
            # Reverse shape: {"Food": "High", "Water": "Medium", "Firewood": "Low"}
            order_map = {
                "high": 0, "highest": 0, "top": 0, "1": 0, "first": 0,
                "medium": 1, "mid": 1, "2": 1, "second": 1,
                "low": 2, "lowest": 2, "3": 2, "third": 2
            }
            tmp = []
            for item, rank in value2issue.items():
                rank_norm = safe_str(rank).lower()
                if rank_norm in order_map:
                    tmp.append((order_map[rank_norm], safe_str(item)))
            tmp.sort()
            ranked_items = [item for _, item in tmp]

    if isinstance(value2reason, dict):
        lower_reason = {safe_str(k).lower(): v for k, v in value2reason.items()}

        # direct item match
        for item in ranked_items:
            if item.lower() in lower_reason:
                reason_map[item] = lower_reason[item.lower()]

        # fallback by rank label
        rank_keys = ["high", "medium", "low"]
        for idx, item in enumerate(ranked_items[:3]):
            if item not in reason_map and idx < len(rank_keys) and rank_keys[idx] in lower_reason:
                reason_map[item] = lower_reason[rank_keys[idx]]

    return ranked_items, reason_map


def format_preference_section(participant: Dict[str, Any]) -> str:
    ranked_items, reason_map = infer_ranked_preferences(participant)
    lines: List[str] = []

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
        lines.append(json_dumps_pretty(to_python(participant.get("value2issue"))))
        raw_reason = to_python(participant.get("value2reason"))
        if raw_reason is not None:
            lines.append("Reasons for your preferences:")
            lines.append(json_dumps_pretty(raw_reason))

    return "\n".join(lines)


# =========================
# Dialogue formatting
# =========================
def format_dialogue_as_you_them(chat_logs: List[Dict[str, Any]], pov_agent_id: str) -> str:
    lines = []
    for turn in chat_logs:
        speaker = safe_str(turn.get("id"))
        text = safe_str(turn.get("text"))
        text = normalize_whitespace(text)
        text = remove_control_chars(text)
        if not text:
            continue
        prefix = "YOU" if speaker == pov_agent_id else "THEM"
        lines.append(f"{prefix}: {text}")
    return "\n".join(lines)


# =========================
# Prompt
# =========================
def build_prompt(example: Dict[str, Any], pov_agent_id: str) -> str:
    participant_info = to_python(example["participant_info"])
    chat_logs = to_python(example["chat_logs"])
    self_info = to_python(participant_info[pov_agent_id])

    dialogue_text = format_dialogue_as_you_them(chat_logs, pov_agent_id)

    svi_question_lines = []
    for q in SVI_14:
        scale = q.get("scale", "1 to 7")
        svi_question_lines.append(
            f'{q["qid"]} (original SVI item {q["original_svi_id"]}, scale {scale}): {q["text"]}'
        )
    svi_question_block = "\n".join(svi_question_lines)

    output_schema = {
        "svi_scores": {
            q["qid"]: {
                "original_svi_id": q["original_svi_id"],
                "score": "integer from 1 to 7",
                "reason": "brief explanation grounded in the dialogue and this participant's own profile"
            }
            for q in SVI_14
        },
        "brief_overall_reason": "2-4 sentences summarizing the participant's overall subjective experience"
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

    prompt = remove_control_chars(normalize_whitespace(prompt))
    return prompt


# =========================
# Validation
# =========================
def validate_prediction(pred: Dict[str, Any]) -> Dict[str, Any]:
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

        # convert score robustly
        if isinstance(score, str):
            m = re.search(r"\d+", score)
            if not m:
                raise ValueError(f"Invalid score for {qid}: {score}")
            score = int(m.group())
        elif isinstance(score, (int, float)):
            score = int(score)
        else:
            raise ValueError(f"Invalid score type for {qid}: {type(score)}")

        if score < 1:
            score = 1
        if score > 7:
            score = 7

        cleaned_scores[qid] = {
            "original_svi_id": int(original_svi_id),
            "score": score,
            "reason": safe_str(reason)
        }

    brief_overall_reason = safe_str(pred.get("brief_overall_reason", ""))

    return {
        "svi_scores": cleaned_scores,
        "brief_overall_reason": brief_overall_reason
    }


# =========================
# API
# =========================
def query_model(prompt: str) -> Dict[str, Any]:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful evaluator of negotiation experience. "
                    "Return strict JSON only. "
                    "Scores must be integers from 1 to 7."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=TEMPERATURE,
    )

    text = response.choices[0].message.content
    text = normalize_whitespace(text)
    parsed = robust_json_loads(text)
    validated = validate_prediction(parsed)
    return validated


# =========================
# Processing
# =========================
def process_example(example: Dict[str, Any], dialogue_index: int) -> List[Dict[str, Any]]:
    participant_info = to_python(example["participant_info"])
    participant_ids = get_sorted_participants(participant_info)

    outputs = []
    for pov_agent_id in participant_ids:
        prompt = build_prompt(example, pov_agent_id)
        pred = query_model(prompt)

        outputs.append({
            "dialogue_index": dialogue_index,
            "pov_agent_id": pov_agent_id,
            "model": MODEL_NAME,
            "prediction": pred
        })

        time.sleep(SLEEP_BETWEEN_CALLS)

    return outputs


def main():
    df = pd.read_parquet(PARQUET_PATH)
    print(f"Loaded dataframe shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    records = [to_python(r) for r in df.to_dict(orient="records")]

    all_results = []
    limit = min(MAX_DIALOGUES, len(records))

    for i in range(limit):
        print(f"Processing dialogue {i + 1}/{limit}")
        example = records[i]

        try:
            rows = process_example(example, i)
            all_results.extend(rows)
            print(f"  Done dialogue {i}")
        except Exception as e:
            print(f"  Failed on dialogue {i}: {e}")
            all_results.append({
                "dialogue_index": i,
                "error": str(e)
            })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"Saved results to: {OUTPUT_PATH}")
    print(f"Total records written: {len(all_results)}")


if __name__ == "__main__":
    main()
