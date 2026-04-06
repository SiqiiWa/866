import os
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Any

import pandas as pd
import numpy as np
import openai

# =========================================================
# CONFIG
# =========================================================

MODEL = "gpt-5-mini"
DATA_PATH = Path("/home/swang4/866/data/train-00000-of-00001.parquet")
OUT_PATH = Path("/home/swang4/866/data/baseline_2/stance_labels.json")

client = openai.OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://ai-gateway.andrew.cmu.edu"
)

# =========================================================
# HELPERS
# =========================================================

def to_python(obj: Any) -> Any:
    """
    Recursively convert parquet / numpy / pandas nested objects
    into plain Python lists / dicts / scalars.
    """
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    # pandas NA / numpy NA
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    if isinstance(obj, np.ndarray):
        return [to_python(x) for x in obj.tolist()]

    if isinstance(obj, list):
        return [to_python(x) for x in obj]

    if isinstance(obj, tuple):
        return [to_python(x) for x in obj]

    if isinstance(obj, dict):
        return {str(k): to_python(v) for k, v in obj.items()}

    # pandas objects that expose to_dict
    if hasattr(obj, "to_dict"):
        try:
            return to_python(obj.to_dict())
        except Exception:
            pass

    # pyarrow struct-like or custom object with __dict__
    if hasattr(obj, "__dict__"):
        try:
            return {str(k): to_python(v) for k, v in vars(obj).items()}
        except Exception:
            pass

    return obj


def maybe_json_load(x: Any) -> Any:
    """
    If x is a JSON string, parse it; otherwise return x.
    """
    if isinstance(x, str):
        s = x.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                return json.loads(s)
            except Exception:
                return x
    return x


def normalize_field(x: Any) -> Any:
    return to_python(maybe_json_load(x))


def safe_len(x: Any) -> int:
    if x is None:
        return 0
    try:
        return len(x)
    except Exception:
        return 0


def parse_dialogue(dialogue_text: str) -> List[Dict]:
    lines = [x.strip() for x in dialogue_text.strip().splitlines() if x.strip()]
    utterances = []
    for i, line in enumerate(lines):
        if line.startswith("YOU:"):
            speaker = "YOU"
            text = line[len("YOU:"):].strip()
        elif line.startswith("THEM:"):
            speaker = "THEM"
            text = line[len("THEM:"):].strip()
        else:
            continue

        utterances.append({
            "turn_index": i,
            "speaker": speaker,
            "text": text,
            "raw": line
        })
    return utterances


def dialogue_so_far(utterances: List[Dict], upto_idx: int) -> str:
    return "\n".join(u["raw"] for u in utterances[:upto_idx + 1])


def build_last_turn_context(annotations: List[Dict], utterances: List[Dict], idx: int) -> str:
    if idx == 0:
        return "No previous annotated turns."

    start = max(0, idx - 2)
    parts = []
    for j in range(start, idx):
        u = utterances[j]
        ann = annotations[j] if j < len(annotations) else None
        if ann is None:
            continue
        parts.append(
            f'{u["speaker"]} said: "{u["text"]}" | '
            f'interpreted stance: {ann.get("stance", "unknown")} | '
            f'action_type: {ann.get("action_type", "unknown")}'
        )
    return "\n".join(parts) if parts else "No previous annotated turns."


def safe_json_response(response):
    return json.loads(response.output_text)


def call_with_retry(fn, *args, max_retries=5, sleep_base=2, **kwargs):
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            print(f"    retry {attempt}/{max_retries} failed: {repr(e)}")
            if attempt < max_retries:
                time.sleep(sleep_base ** (attempt - 1))
    raise last_err


def ensure_parent_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def load_existing_results(out_path: Path) -> Dict:
    if not out_path.exists():
        return {
            "meta": {
                "data_path": str(DATA_PATH),
                "model": MODEL,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "results": []
        }

    with open(out_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_results(payload: Dict, out_path: Path):
    ensure_parent_dir(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def done_key(dialogue_index: int, perspective: str) -> str:
    return f"{dialogue_index}|||{perspective}"


def build_done_set(payload: Dict) -> set:
    done = set()
    for item in payload.get("results", []):
        done.add(done_key(item["dialogue_index"], item["perspective"]))
    return done


def format_dialogue_from_chat_logs(chat_logs: List[Dict], perspective_agent: str) -> str:
    lines = []
    for turn in chat_logs:
        turn = normalize_field(turn)
        speaker = "YOU" if turn.get("id") == perspective_agent else "THEM"
        text = str(turn.get("text", "")).strip()
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def extract_own_preferences(row: Dict, perspective_agent: str) -> str:
    participant_info = normalize_field(row.get("participant_info", {})) or {}
    agent_info = participant_info.get(perspective_agent, {})

    if not isinstance(agent_info, dict):
        return json.dumps(agent_info, ensure_ascii=False)

    # keep minimal change: prefer value2issue
    if "value2issue" in agent_info:
        return json.dumps(agent_info["value2issue"], ensure_ascii=False)

    return json.dumps(agent_info, ensure_ascii=False)


# =========================================================
# JSON SCHEMAS
# =========================================================

PREFERENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "preference_order": {
            "type": "array",
            "items": {"type": "string", "enum": ["food", "water", "firewood"]},
            "minItems": 3,
            "maxItems": 3
        },
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "evidence_type": {"type": "string", "enum": ["none", "implicit", "explicit"]}
    },
    "required": ["reasoning", "preference_order", "confidence", "evidence_type"],
    "additionalProperties": False
}

STANCE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "stance": {"type": "string", "enum": ["proself", "prosocial", "neutral"]},
        "action_type": {"type": "string", "enum": ["info", "influence", "offer", "non-strategic"]},
        "xml_output": {"type": "string"}
    },
    "required": ["reasoning", "stance", "action_type", "xml_output"],
    "additionalProperties": False
}

# =========================================================
# MODEL CALL 1: INFER OPPONENT PREFERENCE
# =========================================================

def infer_opponent_preference(current_dialogue: str, last_preference_inference: Optional[Dict]) -> Dict:
    prompt = f"""
You are annotating a camping negotiation.

Task:
Infer the OPPONENT'S preference ranking from the dialogue so far.

Rules:
- Items are food, water, firewood.
- Output a strict ranking of all 3 items from highest to lowest preference.
- If the opponent explicitly stated a preference ranking or explicitly revealed the highest priority item earlier,
  preserve that and do not revise it later based on concessions.
- Only update when there is genuinely new evidence.
- Focus on inferring what the opponent values, not what they merely concede to.
- Keep reasoning concise.

Previous preference inference:
{json.dumps(last_preference_inference, ensure_ascii=False) if last_preference_inference else "None"}

Dialogue so far:
{current_dialogue}
""".strip()

    response = client.responses.create(
        model=MODEL,
        reasoning={"effort": "low"},
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "opponent_preference_inference",
                "schema": PREFERENCE_SCHEMA,
                "strict": True,
            }
        },
    )
    return safe_json_response(response)

# =========================================================
# MODEL CALL 2: STANCE JUDGMENT
# =========================================================

def judge_stance(
    current_dialogue: str,
    last_turn_context: str,
    own_preferences: str,
    inferred_opponent_preference: Dict,
    current_speaker: str,
) -> Dict:
    prompt = f"""
You are annotating a camping negotiation from the assigned agent's perspective.

Setting:
There are 3 firewoods, 3 water, and 3 foods total.
The goal is to negotiate who gets what.

Task:
Label ONLY the LAST utterance in the dialogue so far.

You must output:
1. stance: proself / prosocial / neutral
2. action_type: info / influence / offer / non-strategic

Definitions:
- Proself: competitive, selfish, more zero-sum, oriented toward own gain
- Prosocial: cooperative, integrative, seeking joint good, helping create tradeoffs
- Neutral: neither clearly proself nor prosocial

Action types:
- info = sharing or asking for information
- influence = affective or rational persuasion
- offer = making or revising a proposed allocation
- non-strategic = small talk or aimless talk

Hard rules:
- If the opponent's action makes the situation more zero-sum, perceive it as proself.
- If the opponent's action makes the situation more integrative, perceive it as prosocial.
- If a speaker explicitly stated a preference ranking, preserve it.
- Different action types can make the stance signal stronger or weaker.

Assigned agent preferences:
{own_preferences}

Current inferred opponent preference:
{json.dumps(inferred_opponent_preference, ensure_ascii=False)}

Previous annotated turn context:
{last_turn_context}

Dialogue so far:
{current_dialogue}

The last utterance speaker is: {current_speaker}

In reasoning, explain:
- how you arrived at the stance judgment
- what action type the utterance is
- whether the action type makes the stance signal stronger or weaker

Also produce:
xml_output = <reason>...</reason><answer>STANCE</answer>
""".strip()

    response = client.responses.create(
        model=MODEL,
        reasoning={"effort": "low"},
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "stance_annotation",
                "schema": STANCE_SCHEMA,
                "strict": True,
            }
        },
    )
    return safe_json_response(response)

# =========================================================
# MAIN ANNOTATION
# =========================================================

def annotate_dialogue(dialogue_name: str, dialogue_text: str, own_preferences: str) -> List[Dict]:
    utterances = parse_dialogue(dialogue_text)
    annotations = []
    last_opponent_pref = None

    total_turns = len(utterances)
    print(f"\n=== Starting {dialogue_name} ({total_turns} turns) ===")

    for i, utt in enumerate(utterances):
        current_text = dialogue_so_far(utterances, i)
        turn_start = time.time()

        print(f"\n[{dialogue_name}] Turn {i+1}/{total_turns}")
        print(f"Speaker: {utt['speaker']}")
        print(f"Text   : {utt['text']}")

        print("  -> Call 1: inferring opponent preference...")
        pref_result = call_with_retry(
            infer_opponent_preference,
            current_dialogue=current_text,
            last_preference_inference=last_opponent_pref
        )
        last_opponent_pref = pref_result
        print(
            f"     preference_order = {pref_result['preference_order']}, "
            f"confidence = {pref_result['confidence']}, "
            f"evidence = {pref_result['evidence_type']}"
        )

        last_turn_ctx = build_last_turn_context(annotations, utterances, i)

        print("  -> Call 2: judging stance/action type...")
        stance_result = call_with_retry(
            judge_stance,
            current_dialogue=current_text,
            last_turn_context=last_turn_ctx,
            own_preferences=own_preferences,
            inferred_opponent_preference=pref_result,
            current_speaker=utt["speaker"],
        )
        print(
            f"     stance = {stance_result['stance']}, "
            f"action_type = {stance_result['action_type']}"
        )
        print(f"     completed in {time.time() - turn_start:.2f}s")

        annotations.append({
            "turn_index": utt["turn_index"],
            "speaker": utt["speaker"],
            "text": utt["text"],
            "opponent_preference_inference": pref_result,
            "stance": stance_result["stance"],
            "action_type": stance_result["action_type"],
            "reasoning": stance_result["reasoning"],
            "xml_output": stance_result["xml_output"],
        })

    print(f"=== Finished {dialogue_name} ===\n")
    return annotations

# =========================================================
# DATA LOADING
# =========================================================

def load_first_n_dialogues(data_path: Path, n: int = 100) -> List[Dict]:
    df = pd.read_parquet(data_path)
    df = df.iloc[:n].reset_index(drop=True)

    rows = []
    for _, row in df.iterrows():
        row_dict = {}
        for col in df.columns:
            row_dict[col] = normalize_field(row[col])
        rows.append(row_dict)

    return rows


def load_dialogue_range(data_path: Path, start_idx: int, end_idx: int) -> List[Dict]:
    df = pd.read_parquet(data_path)
    df = df.iloc[start_idx:end_idx].reset_index(drop=True)

    rows = []
    for _, row in df.iterrows():
        row_dict = {}
        for col in df.columns:
            row_dict[col] = normalize_field(row[col])
        rows.append(row_dict)

    return rows


def run_dialogue_range(start_idx: int, end_idx: int, out_path: Path) -> None:
    print(f"Reading dataset from: {DATA_PATH}")
    rows = load_dialogue_range(DATA_PATH, start_idx=start_idx, end_idx=end_idx)

    ensure_parent_dir(out_path)
    all_results = load_existing_results(out_path)
    done = build_done_set(all_results)

    print(f"Incremental outputs to: {out_path}")
    print(f"Dialogue index range: [{start_idx}, {end_idx})")
    print(f"Already completed: {len(done)} dialogue-perspective pairs")

    perspectives = ["mturk_agent_1", "mturk_agent_2"]

    for offset, row in enumerate(rows):
        dialogue_index = start_idx + offset
        dialogue_name = f"dialogue_{dialogue_index:05d}"

        chat_logs = normalize_field(row.get("chat_logs", []))
        if chat_logs is None:
            chat_logs = []

        if not isinstance(chat_logs, list):
            print(f"Skipping {dialogue_name}: chat_logs is not list, got {type(chat_logs)}")
            continue

        if len(chat_logs) == 0:
            print(f"Skipping {dialogue_name}: empty chat_logs")
            continue

        for perspective in perspectives:
            key = done_key(dialogue_index, perspective)
            if key in done:
                print(f"Skipping {dialogue_name} / {perspective}: already done")
                continue

            try:
                print(f"\nRunning {dialogue_name} / {perspective} ...")

                dialogue_text = format_dialogue_from_chat_logs(
                    chat_logs=chat_logs,
                    perspective_agent=perspective
                )

                own_preferences = extract_own_preferences(
                    row=row,
                    perspective_agent=perspective
                )

                annotations = annotate_dialogue(
                    dialogue_name=f"{dialogue_name}__{perspective}",
                    dialogue_text=dialogue_text,
                    own_preferences=own_preferences,
                )

                result_payload = {
                    "dialogue_index": dialogue_index,
                    "dialogue_name": dialogue_name,
                    "perspective": perspective,
                    "own_preferences": own_preferences,
                    "source_annotations": normalize_field(row.get("annotations", [])),
                    "annotations": annotations,
                }

                all_results["results"].append(result_payload)
                save_results(all_results, out_path)
                done.add(key)

                print(f"Saved: {dialogue_name} / {perspective}")

            except Exception as e:
                print(f"FAILED on {dialogue_name} / {perspective}: {repr(e)}")
                save_results(all_results, out_path)
                continue

    print("\nAll done.")

# =========================================================
# MAIN
# =========================================================

def main():
    run_dialogue_range(start_idx=0, end_idx=100, out_path=OUT_PATH)


if __name__ == "__main__":
    main()
