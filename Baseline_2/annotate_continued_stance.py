import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any

from stance_prompt import (
    MODEL,
    build_last_turn_context,
    build_done_set,
    call_with_retry,
    ensure_parent_dir,
    infer_opponent_preference,
    judge_stance,
    load_existing_results,
    save_results,
)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_key(dialogue_index: int, perspective: str) -> tuple[int, str]:
    return (dialogue_index, perspective)


def build_base_lookup(payload: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    lookup: dict[tuple[int, str], dict[str, Any]] = {}
    for item in payload.get("results", []):
        lookup[make_key(item["dialogue_index"], item["perspective"])] = item
    return lookup


def format_dialogue_from_continued(dialogue: list[dict[str, Any]], perspective: str) -> list[dict[str, Any]]:
    utterances = []
    for idx, turn in enumerate(dialogue):
        speaker = "YOU" if turn.get("id") == perspective else "THEM"
        text = str(turn.get("text", "")).strip()
        utterances.append(
            {
                "turn_index": idx,
                "speaker": speaker,
                "text": text,
                "raw": f"{speaker}: {text}",
            }
        )
    return utterances


def dialogue_so_far(utterances: list[dict[str, Any]], upto_idx: int) -> str:
    return "\n".join(u["raw"] for u in utterances[: upto_idx + 1])


def annotate_new_turns(
    continued_record: dict[str, Any],
    base_result: dict[str, Any],
) -> dict[str, Any]:
    perspective = continued_record["perspective"]
    own_preferences = json.dumps(
        continued_record["agents"][perspective]["value2issue"], ensure_ascii=False
    )

    utterances = format_dialogue_from_continued(continued_record["dialogue"], perspective)
    preserved_turn_count = int(continued_record["preserved_through_turn_index"])

    base_annotations = base_result["annotations"]
    combined_annotations = list(base_annotations[:preserved_turn_count])

    if preserved_turn_count > 0:
        last_opponent_pref = combined_annotations[-1]["opponent_preference_inference"]
    else:
        last_opponent_pref = None

    for i in range(preserved_turn_count, len(utterances)):
        utt = utterances[i]
        current_text = dialogue_so_far(utterances, i)

        pref_result = call_with_retry(
            infer_opponent_preference,
            current_dialogue=current_text,
            last_preference_inference=last_opponent_pref,
        )
        last_opponent_pref = pref_result

        last_turn_ctx = build_last_turn_context(combined_annotations, utterances, i)

        stance_result = call_with_retry(
            judge_stance,
            current_dialogue=current_text,
            last_turn_context=last_turn_ctx,
            own_preferences=own_preferences,
            inferred_opponent_preference=pref_result,
            current_speaker=utt["speaker"],
        )

        combined_annotations.append(
            {
                "turn_index": utt["turn_index"],
                "speaker": utt["speaker"],
                "text": utt["text"],
                "opponent_preference_inference": pref_result,
                "stance": stance_result["stance"],
                "action_type": stance_result["action_type"],
                "reasoning": stance_result["reasoning"],
                "xml_output": stance_result["xml_output"],
            }
        )

    return {
        "dialogue_index": continued_record["dialogue_index"],
        "dialogue_name": continued_record["dialogue_name"],
        "perspective": perspective,
        "own_preferences": own_preferences,
        "source_annotations": base_result.get("source_annotations", []),
        "annotations": combined_annotations,
        "continued_dialogue_path": continued_record.get("_source_file", ""),
        "preserved_through_turn_index": continued_record["preserved_through_turn_index"],
        "continued_from_turn_index": continued_record["continued_from_turn_index"],
        "selected_important_turn": continued_record.get("selected_important_turn"),
    }


def process_continued_file(
    path: Path,
    base_lookup: dict[tuple[int, str], dict[str, Any]],
) -> tuple[tuple[int, str], str, dict[str, Any]]:
    record = load_json(path)
    record["_source_file"] = str(path)
    key = make_key(record["dialogue_index"], record["perspective"])
    if key not in base_lookup:
        raise ValueError(f"Missing base stance result for {key}")
    merged = annotate_new_turns(record, base_lookup[key])
    return key, path.name, merged


def sort_results(results: list[dict[str, Any]]) -> None:
    results.sort(key=lambda item: (item["dialogue_index"], item["perspective"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--continued-dir", type=Path, required=True)
    parser.add_argument("--base-stance-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    base_payload = load_json(args.base_stance_path)
    base_lookup = build_base_lookup(base_payload)

    output = load_existing_results(args.out)
    output["meta"] = {
        "base_stance_path": str(args.base_stance_path),
        "continued_dir": str(args.continued_dir),
        "model": MODEL,
        "task": "continued_dialogue_stance_with_preserved_prefix",
    }
    done = build_done_set(output)

    continued_files = sorted(args.continued_dir.glob("*_continued.json"))
    pending_files: list[Path] = []
    for path in continued_files:
        record = load_json(path)
        key = make_key(record["dialogue_index"], record["perspective"])
        if key not in base_lookup:
            raise ValueError(f"Missing base stance result for {key}")
        if key in done:
            print(f"Skipping {path.name} (already in output)")
            continue
        pending_files.append(path)

    if not pending_files:
        ensure_parent_dir(args.out)
        sort_results(output["results"])
        save_results(output, args.out)
        print(f"Nothing to do. Saved to {args.out}")
        return

    max_workers = max(1, args.max_workers)
    print(f"Submitting {len(pending_files)} files with max_workers={max_workers}")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(process_continued_file, path, base_lookup): path
            for path in pending_files
        }
        for future in as_completed(future_to_path):
            key, path_name, merged = future.result()
            print(f"Completed {path_name}")
            output["results"].append(merged)
            done.add(key)
            sort_results(output["results"])
            save_results(output, args.out)

    ensure_parent_dir(args.out)
    sort_results(output["results"])
    save_results(output, args.out)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
