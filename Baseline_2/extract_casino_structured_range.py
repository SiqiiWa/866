import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def to_python(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj

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
    if hasattr(obj, "to_dict"):
        try:
            return to_python(obj.to_dict())
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return {str(k): to_python(v) for k, v in vars(obj).items()}
        except Exception:
            pass
    return obj


def maybe_json_load(x: Any) -> Any:
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


def build_dialogue_name(dialogue_index: int) -> str:
    return f"dialogue_{dialogue_index:05d}"


def get_other_agent(agent_id: str) -> str:
    if agent_id == "mturk_agent_1":
        return "mturk_agent_2"
    return "mturk_agent_1"


def build_dialogue(chat_logs: list[dict[str, Any]], dialogue_name: str) -> list[dict[str, Any]]:
    structured_dialogue = []
    for offset, turn in enumerate(chat_logs, start=1):
        turn = normalize_field(turn) or {}
        speaker = turn.get("id", "")
        structured_dialogue.append(
            {
                "text": str(turn.get("text", "")),
                "task_data": normalize_field(turn.get("task_data", {})) or {},
                "id": speaker,
                "turn_index": offset,
                "turn_id": f"{dialogue_name}_turn_{offset:02d}",
                "speaker": speaker,
                "listener": get_other_agent(speaker),
            }
        )
    return structured_dialogue


def extract_range(data_path: Path, start: int, end: int) -> list[dict[str, Any]]:
    df = pd.read_parquet(data_path)
    rows = []

    for dialogue_index in range(start, end):
        row = df.iloc[dialogue_index]
        dialogue_name = build_dialogue_name(dialogue_index)
        participant_info = normalize_field(row["participant_info"]) or {}
        chat_logs = normalize_field(row["chat_logs"]) or []
        annotations = normalize_field(row.get("annotations", [])) or []

        rows.append(
            {
                "dialogue_index": dialogue_index,
                "dialogue_name": dialogue_name,
                "agents": participant_info,
                "dialogue": build_dialogue(chat_logs, dialogue_name),
                "annotations": annotations,
            }
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    payload = extract_range(args.data_path, args.start, args.end)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(payload)} dialogues to {args.out}")


if __name__ == "__main__":
    main()
