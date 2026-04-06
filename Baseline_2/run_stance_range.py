import argparse
from pathlib import Path

from stance_prompt import run_dialogue_range


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    run_dialogue_range(
        start_idx=args.start,
        end_idx=args.end,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
