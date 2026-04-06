from pathlib import Path

from stance_prompt import run_dialogue_range


OUT_PATH = Path("/home/swang4/866/data/baseline_2/stance_labels_100-199.json")


if __name__ == "__main__":
    run_dialogue_range(start_idx=100, end_idx=200, out_path=OUT_PATH)
