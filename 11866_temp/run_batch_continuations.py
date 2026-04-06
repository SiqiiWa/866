from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = (
    REPO_ROOT
    / "examples"
    / "experimental"
    / "negotiation"
    / "two_agent_casino_negotiation.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run continuation generation for many dialogue indices."
    )
    parser.add_argument("--base-url", required=True, help="LiteLLM proxy base URL.")
    parser.add_argument("--model", required=True, help="Model name.")
    parser.add_argument(
        "--api-key",
        default="",
        help="Optional API key. If omitted, the child process uses existing env vars.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First dialogue_index to run.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="How many dialogue indices to run.",
    )
    parser.add_argument(
        "--perspective",
        choices=["mturk_agent_1", "mturk_agent_2", "both"],
        default="mturk_agent_1",
        help="Which perspective(s) to run for each scenario.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=20,
        help="Max turns passed to the continuation script.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=768,
        help="Max completion tokens passed to the continuation script.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default="low",
        help="Reasoning effort passed to the continuation script.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated files. Defaults to a timestamped batch directory.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the batch immediately if any run fails.",
    )
    return parser.parse_args()


def build_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        REPO_ROOT
        / "examples"
        / "experimental"
        / "negotiation"
        / f"batch_{timestamp}"
    )


def perspectives_to_run(perspective_arg: str) -> list[str]:
    if perspective_arg == "both":
        return ["mturk_agent_1", "mturk_agent_2"]
    return [perspective_arg]


def build_output_file(output_dir: Path, dialogue_index: int, perspective: str) -> Path:
    return output_dir / f"dialogue_{dialogue_index:05d}_{perspective}_continued.json"


def main() -> None:
    args = parse_args()
    output_dir = build_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    perspectives = perspectives_to_run(args.perspective)
    results: list[dict[str, object]] = []

    total_runs = args.count * len(perspectives)
    current_run = 0

    for dialogue_index in range(args.start_index, args.start_index + args.count):
        for perspective in perspectives:
            current_run += 1
            output_file = build_output_file(output_dir, dialogue_index, perspective)

            cmd = [
                sys.executable,
                str(SCRIPT_PATH),
                "--base-url",
                args.base_url,
                "--model",
                args.model,
                "--dialogue-index",
                str(dialogue_index),
                "--perspective",
                perspective,
                "--max-turns",
                str(args.max_turns),
                "--max-completion-tokens",
                str(args.max_completion_tokens),
                "--reasoning-effort",
                args.reasoning_effort,
                "--output-file",
                str(output_file),
            ]
            if args.api_key:
                cmd.extend(["--api-key", args.api_key])

            print(
                f"[{current_run}/{total_runs}] dialogue_index={dialogue_index} perspective={perspective}",
                flush=True,
            )

            completed = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )

            result = {
                "dialogue_index": dialogue_index,
                "perspective": perspective,
                "output_file": str(output_file),
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
            results.append(result)

            if completed.returncode == 0:
                print(f"  success -> {output_file}", flush=True)
            else:
                print("  failed", flush=True)
                if completed.stdout:
                    print(completed.stdout, flush=True)
                if completed.stderr:
                    print(completed.stderr, flush=True)
                if args.stop_on_error:
                    summary_path = output_dir / "batch_summary.json"
                    with summary_path.open("w", encoding="utf-8") as file:
                        json.dump({"results": results}, file, ensure_ascii=False, indent=2)
                    raise SystemExit(1)

    summary_path = output_dir / "batch_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump({"results": results}, file, ensure_ascii=False, indent=2)

    successes = sum(1 for item in results if item["returncode"] == 0)
    failures = len(results) - successes
    print(f"Completed batch. success={successes} failure={failures}")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
