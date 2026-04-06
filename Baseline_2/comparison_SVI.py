import json
import math
from typing import Any, Dict, List, Optional

import pandas as pd


# =========================================================
# Paths
# =========================================================
PARQUET_PATH = "/home/swang4/866/data/train-00000-of-00001.parquet"
MODEL_JSON_PATH = "/home/swang4/866/Baseline_2/casino_plan2_svi_baseline_first50.json"

OUTPUT_SCORES_PATH = "/home/swang4/866/Baseline_2/casino_svi_scored_first50.json"
OUTPUT_CORR_PATH = "/home/swang4/866/Baseline_2/casino_svi_correlations_first50.json"


# =========================================================
# Dataset label mappings (5-point scales)
# =========================================================
SATISFACTION_MAP = {
    "Extremely dissatisfied": 1,
    "Slightly dissatisfied": 2,
    "Undecided": 3,
    "Slightly satisfied": 4,
    "Extremely satisfied": 5,
}

OPPONENT_LIKENESS_MAP = {
    "Extremely dislike": 1,
    "Slightly dislike": 2,
    "Undecided": 3,
    "Slightly like": 4,
    "Extremely like": 5,
}


# =========================================================
# Helpers
# =========================================================
def to_python(obj: Any) -> Any:
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


def safe_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(round(x))
    if isinstance(x, str):
        x = x.strip()
        if x.isdigit():
            return int(x)
        # try to extract first integer
        digits = "".join(ch for ch in x if ch.isdigit())
        if digits:
            return int(digits)
    return None


def reverse_score_1_to_7(x: float) -> float:
    return 8 - x


def mean_ignore_none(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def pearson_corr(x: List[float], y: List[float]) -> Optional[float]:
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return None
    sx = pd.Series(x, dtype="float64")
    sy = pd.Series(y, dtype="float64")
    val = sx.corr(sy, method="pearson")
    if pd.isna(val):
        return None
    return float(val)


def spearman_corr(x: List[float], y: List[float]) -> Optional[float]:
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return None
    sx = pd.Series(x, dtype="float64")
    sy = pd.Series(y, dtype="float64")
    val = sx.corr(sy, method="spearman")
    if pd.isna(val):
        return None
    return float(val)


def collect_xy(rows: List[Dict[str, Any]], x_key: str, y_key: str) -> Dict[str, Any]:
    xs = []
    ys = []
    used_rows = 0

    for row in rows:
        x = row.get(x_key)
        y = row.get(y_key)
        if x is None or y is None:
            continue
        xs.append(float(x))
        ys.append(float(y))
        used_rows += 1

    return {
        "n": used_rows,
        "pearson": pearson_corr(xs, ys),
        "spearman": spearman_corr(xs, ys),
    }


# =========================================================
# Parse model SVI predictions
# =========================================================
def parse_model_scores(prediction: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """
    Model output format assumed:
    {
      "svi_scores": {
        "Q1": {"score": ...},
        ...
        "Q14": {"score": ...}
      },
      ...
    }
    """
    svi_scores = prediction.get("svi_scores", {})
    q = {}

    for i in range(1, 15):
        key = f"Q{i}"
        item = svi_scores.get(key, {})
        score = safe_int(item.get("score"))
        if score is None:
            q[key] = None
        else:
            score = max(1, min(7, score))
            q[key] = float(score)

    # reverse-scored items according to SVI
    q3r = reverse_score_1_to_7(q["Q3"]) if q["Q3"] is not None else None
    q5r = reverse_score_1_to_7(q["Q5"]) if q["Q5"] is not None else None

    instrumental = mean_ignore_none([q["Q1"], q["Q2"], q3r])
    legitimacy = q["Q4"]
    self_score = mean_ignore_none([q5r, q["Q6"]])
    process = mean_ignore_none([q["Q7"], q["Q8"], q["Q9"], q["Q10"]])
    relationship = mean_ignore_none([q["Q11"], q["Q12"], q["Q13"], q["Q14"]])

    all_items_after_reverse = [
        q["Q1"], q["Q2"], q3r, q["Q4"], q5r, q["Q6"],
        q["Q7"], q["Q8"], q["Q9"], q["Q10"],
        q["Q11"], q["Q12"], q["Q13"], q["Q14"]
    ]
    overall_item_mean = mean_ignore_none(all_items_after_reverse)

    overall_dim_mean = mean_ignore_none([
        instrumental, legitimacy, self_score, process, relationship
    ])

    satisfaction_proxy_3way = mean_ignore_none([instrumental, self_score, process])

    return {
        "Q1": q["Q1"],
        "Q2": q["Q2"],
        "Q3": q["Q3"],
        "Q4": q["Q4"],
        "Q5": q["Q5"],
        "Q6": q["Q6"],
        "Q7": q["Q7"],
        "Q8": q["Q8"],
        "Q9": q["Q9"],
        "Q10": q["Q10"],
        "Q11": q["Q11"],
        "Q12": q["Q12"],
        "Q13": q["Q13"],
        "Q14": q["Q14"],
        "Q3_reversed": q3r,
        "Q5_reversed": q5r,
        "instrumental": instrumental,
        "legitimacy": legitimacy,
        "self": self_score,
        "process": process,
        "relationship": relationship,
        "overall_item_mean": overall_item_mean,
        "overall_dim_mean": overall_dim_mean,
        "satisfaction_proxy_3way": satisfaction_proxy_3way,
    }


# =========================================================
# Load raw dataset utility range
# =========================================================
def get_global_points_range(records: List[Dict[str, Any]]) -> Dict[str, int]:
    all_points = []

    for ex in records:
        participant_info = to_python(ex.get("participant_info", {}))
        for agent_id, info in participant_info.items():
            outcomes = to_python(info.get("outcomes", {}))
            p = outcomes.get("points_scored")
            if p is not None:
                p = safe_int(p)
                if p is not None:
                    all_points.append(p)

    return {
        "min_points": min(all_points),
        "max_points": max(all_points),
    }


def normalize_points(points: Optional[int], min_points: int, max_points: int) -> Optional[float]:
    if points is None:
        return None
    if max_points == min_points:
        return None
    return (points - min_points) / (max_points - min_points)


# =========================================================
# Main
# =========================================================
def main():
    print("Loading parquet...")
    df = pd.read_parquet(PARQUET_PATH)
    records = [to_python(r) for r in df.to_dict(orient="records")]

    print("Loading model outputs...")
    with open(MODEL_JSON_PATH, "r", encoding="utf-8") as f:
        model_rows = json.load(f)

    global_range = get_global_points_range(records)
    min_points = global_range["min_points"]
    max_points = global_range["max_points"]

    print(f"Global points range from dataset: min={min_points}, max={max_points}")

    scored_rows = []

    # First pass: participant-level scores
    for row in model_rows:
        if "error" in row:
            continue

        dialogue_index = row.get("dialogue_index")
        pov_agent_id = row.get("pov_agent_id")
        prediction = row.get("prediction", {})

        if dialogue_index is None or pov_agent_id is None:
            continue
        if dialogue_index < 0 or dialogue_index >= len(records):
            continue

        ex = records[dialogue_index]
        participant_info = to_python(ex.get("participant_info", {}))

        if pov_agent_id not in participant_info:
            continue

        agent_info = to_python(participant_info[pov_agent_id])
        outcomes = to_python(agent_info.get("outcomes", {}))

        points_scored = safe_int(outcomes.get("points_scored"))
        satisfaction_label = outcomes.get("satisfaction")
        opponent_likeness_label = outcomes.get("opponent_likeness")

        satisfaction_5 = SATISFACTION_MAP.get(satisfaction_label)
        opponent_likeness_5 = OPPONENT_LIKENESS_MAP.get(opponent_likeness_label)
        utility_norm = normalize_points(points_scored, min_points, max_points)

        model_svi = parse_model_scores(prediction)

        out_row = {
            "dialogue_index": dialogue_index,
            "pov_agent_id": pov_agent_id,
            "dataset_points_scored": points_scored,
            "dataset_satisfaction_label": satisfaction_label,
            "dataset_satisfaction_5": satisfaction_5,
            "dataset_opponent_likeness_label": opponent_likeness_label,
            "dataset_opponent_likeness_5": opponent_likeness_5,
            "utility_norm": utility_norm,
        }
        out_row.update(model_svi)

        scored_rows.append(out_row)

    # Second pass: add joint utility per dialogue
    by_dialogue = {}
    for row in scored_rows:
        by_dialogue.setdefault(row["dialogue_index"], []).append(row)

    for dialogue_index, rows in by_dialogue.items():
        utilities = [r.get("utility_norm") for r in rows if r.get("utility_norm") is not None]

        joint_utility_geom = None
        if len(utilities) == 2:
            joint_utility_geom = math.sqrt(utilities[0] * utilities[1])

        for r in rows:
            r["joint_utility_geom"] = joint_utility_geom

    # Save participant-level scored file
    with open(OUTPUT_SCORES_PATH, "w", encoding="utf-8") as f:
        json.dump(scored_rows, f, ensure_ascii=False, indent=2)

    print(f"\nSaved scored SVI rows to:\n{OUTPUT_SCORES_PATH}")
    print(f"Number of participant rows: {len(scored_rows)}")

    # =====================================================
    # Correlations
    # =====================================================
    correlations = {
        "relationship_vs_opponent_likeness": collect_xy(
            scored_rows, "relationship", "dataset_opponent_likeness_5"
        ),

        "instrumental_vs_satisfaction": collect_xy(
            scored_rows, "instrumental", "dataset_satisfaction_5"
        ),
        "self_vs_satisfaction": collect_xy(
            scored_rows, "self", "dataset_satisfaction_5"
        ),
        "process_vs_satisfaction": collect_xy(
            scored_rows, "process", "dataset_satisfaction_5"
        ),
        "satisfaction_proxy_3way_vs_satisfaction": collect_xy(
            scored_rows, "satisfaction_proxy_3way", "dataset_satisfaction_5"
        ),
        "overall_item_mean_vs_satisfaction": collect_xy(
            scored_rows, "overall_item_mean", "dataset_satisfaction_5"
        ),
        "overall_dim_mean_vs_satisfaction": collect_xy(
            scored_rows, "overall_dim_mean", "dataset_satisfaction_5"
        ),

        "utility_norm_vs_instrumental": collect_xy(
            scored_rows, "utility_norm", "instrumental"
        ),
        "utility_norm_vs_self": collect_xy(
            scored_rows, "utility_norm", "self"
        ),
        "utility_norm_vs_process": collect_xy(
            scored_rows, "utility_norm", "process"
        ),
        "utility_norm_vs_relationship": collect_xy(
            scored_rows, "utility_norm", "relationship"
        ),
        "utility_norm_vs_overall_item_mean": collect_xy(
            scored_rows, "utility_norm", "overall_item_mean"
        ),
        "utility_norm_vs_overall_dim_mean": collect_xy(
            scored_rows, "utility_norm", "overall_dim_mean"
        ),

        "joint_utility_vs_instrumental": collect_xy(
            scored_rows, "joint_utility_geom", "instrumental"
        ),
        "joint_utility_vs_self": collect_xy(
            scored_rows, "joint_utility_geom", "self"
        ),
        "joint_utility_vs_process": collect_xy(
            scored_rows, "joint_utility_geom", "process"
        ),
        "joint_utility_vs_relationship": collect_xy(
            scored_rows, "joint_utility_geom", "relationship"
        ),
        "joint_utility_vs_overall_item_mean": collect_xy(
            scored_rows, "joint_utility_geom", "overall_item_mean"
        ),
        "joint_utility_vs_overall_dim_mean": collect_xy(
            scored_rows, "joint_utility_geom", "overall_dim_mean"
        ),

        "utility_norm_vs_dataset_satisfaction": collect_xy(
            scored_rows, "utility_norm", "dataset_satisfaction_5"
        ),
        "utility_norm_vs_dataset_opponent_likeness": collect_xy(
            scored_rows, "utility_norm", "dataset_opponent_likeness_5"
        ),
        "joint_utility_vs_dataset_satisfaction": collect_xy(
            scored_rows, "joint_utility_geom", "dataset_satisfaction_5"
        ),
        "joint_utility_vs_dataset_opponent_likeness": collect_xy(
            scored_rows, "joint_utility_geom", "dataset_opponent_likeness_5"
        ),
    }

    with open(OUTPUT_CORR_PATH, "w", encoding="utf-8") as f:
        json.dump(correlations, f, ensure_ascii=False, indent=2)

    print(f"\nSaved correlation summary to:\n{OUTPUT_CORR_PATH}")

    print("\n================ CORRELATIONS ================\n")
    for name, stats in correlations.items():
        print(name)
        print(f"  n        = {stats['n']}")
        print(f"  pearson  = {stats['pearson']}")
        print(f"  spearman = {stats['spearman']}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()