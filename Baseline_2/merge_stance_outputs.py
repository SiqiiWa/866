import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()

    merged = None
    all_results = []

    for path in args.inputs:
        payload = load_json(path)
        if merged is None:
            merged = {
                "meta": payload.get("meta", {}),
                "results": [],
            }
        all_results.extend(payload.get("results", []))

    all_results.sort(key=lambda item: (item["dialogue_index"], item["perspective"]))
    merged["results"] = all_results
    save_json(args.out, merged)

    print(f"Merged {len(args.inputs)} files into {args.out}")
    print(f"Total results: {len(all_results)}")


if __name__ == "__main__":
    main()
