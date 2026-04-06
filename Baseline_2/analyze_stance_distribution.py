import argparse
import json
from collections import Counter
from pathlib import Path


def load_payload(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dedupe_results(results: list[dict]) -> tuple[list[dict], int]:
    deduped: dict[tuple[int, str], dict] = {}
    duplicate_count = 0
    for item in results:
        key = (item.get("dialogue_index"), item.get("perspective"))
        if key in deduped:
            duplicate_count += 1
        deduped[key] = item
    ordered = sorted(deduped.values(), key=lambda x: (x["dialogue_index"], x["perspective"]))
    return ordered, duplicate_count


def filtered_annotations(item: dict, continued_only: bool) -> list[dict]:
    annotations = item.get("annotations", [])
    if not continued_only:
        return annotations

    continued_from = item.get("continued_from_turn_index")
    if continued_from is None:
        return annotations

    return [ann for ann in annotations if ann.get("turn_index", -1) >= continued_from]


def summarize_payload(path: Path, continued_only: bool = False, dedupe: bool = True) -> dict:
    payload = load_payload(path)
    raw_results = payload.get("results", [])
    duplicate_count = 0
    if dedupe:
        results, duplicate_count = dedupe_results(raw_results)
    else:
        results = raw_results

    overall_counter: Counter[str] = Counter()
    speaker_counters: dict[str, Counter[str]] = {
        "YOU": Counter(),
        "THEM": Counter(),
        "OTHER": Counter(),
    }

    dialogue_count = len(results)
    annotation_count = 0

    for item in results:
        for ann in filtered_annotations(item, continued_only):
            stance = ann.get("stance", "<missing>")
            speaker = ann.get("speaker", "OTHER")
            if speaker not in speaker_counters:
                speaker = "OTHER"

            overall_counter[stance] += 1
            speaker_counters[speaker][stance] += 1
            annotation_count += 1

    return {
        "path": str(path),
        "dialogue_count": dialogue_count,
        "annotation_count": annotation_count,
        "continued_only": continued_only,
        "deduped": dedupe,
        "raw_result_count": len(raw_results),
        "duplicate_result_count": duplicate_count,
        "overall": dict(overall_counter),
        "by_speaker": {speaker: dict(counter) for speaker, counter in speaker_counters.items()},
    }


def normalize(counter_dict: dict[str, int], labels: list[str]) -> dict[str, float]:
    total = sum(counter_dict.get(label, 0) for label in labels)
    if total == 0:
        return {label: 0.0 for label in labels}
    return {label: counter_dict.get(label, 0) / total for label in labels}


def build_comparison(base_summary: dict, rl_summary: dict) -> dict:
    labels = sorted(set(base_summary["overall"]) | set(rl_summary["overall"]))

    base_overall = base_summary["overall"]
    rl_overall = rl_summary["overall"]
    base_pct = normalize(base_overall, labels)
    rl_pct = normalize(rl_overall, labels)

    comparison = {
        "labels": labels,
        "baseline": {
            "path": base_summary["path"],
            "dialogue_count": base_summary["dialogue_count"],
            "annotation_count": base_summary["annotation_count"],
            "continued_only": base_summary["continued_only"],
            "raw_result_count": base_summary["raw_result_count"],
            "duplicate_result_count": base_summary["duplicate_result_count"],
            "counts": {label: base_overall.get(label, 0) for label in labels},
            "percentages": base_pct,
            "by_speaker": {},
        },
        "rl": {
            "path": rl_summary["path"],
            "dialogue_count": rl_summary["dialogue_count"],
            "annotation_count": rl_summary["annotation_count"],
            "continued_only": rl_summary["continued_only"],
            "raw_result_count": rl_summary["raw_result_count"],
            "duplicate_result_count": rl_summary["duplicate_result_count"],
            "counts": {label: rl_overall.get(label, 0) for label in labels},
            "percentages": rl_pct,
            "by_speaker": {},
        },
        "delta": {
            "counts": {
                label: rl_overall.get(label, 0) - base_overall.get(label, 0) for label in labels
            },
            "percentage_points": {
                label: (rl_pct[label] - base_pct[label]) * 100 for label in labels
            },
        },
    }

    speakers = sorted(
        set(base_summary["by_speaker"].keys()) | set(rl_summary["by_speaker"].keys())
    )
    for speaker in speakers:
        base_speaker_counts = base_summary["by_speaker"].get(speaker, {})
        rl_speaker_counts = rl_summary["by_speaker"].get(speaker, {})
        base_speaker_pct = normalize(base_speaker_counts, labels)
        rl_speaker_pct = normalize(rl_speaker_counts, labels)
        comparison["baseline"]["by_speaker"][speaker] = {
            "counts": {label: base_speaker_counts.get(label, 0) for label in labels},
            "percentages": base_speaker_pct,
        }
        comparison["rl"]["by_speaker"][speaker] = {
            "counts": {label: rl_speaker_counts.get(label, 0) for label in labels},
            "percentages": rl_speaker_pct,
        }

    return comparison


def print_table(comparison: dict) -> None:
    labels = comparison["labels"]
    print("Dataset summary")
    print("-" * 79)
    print(
        f"baseline: dialogues={comparison['baseline']['dialogue_count']}, "
        f"annotations={comparison['baseline']['annotation_count']}, "
        f"continued_only={comparison['baseline']['continued_only']}, "
        f"duplicates_removed={comparison['baseline']['duplicate_result_count']}"
    )
    print(
        f"rl:       dialogues={comparison['rl']['dialogue_count']}, "
        f"annotations={comparison['rl']['annotation_count']}, "
        f"continued_only={comparison['rl']['continued_only']}, "
        f"duplicates_removed={comparison['rl']['duplicate_result_count']}"
    )
    print()
    print("Overall stance distribution")
    print("-" * 79)
    print(
        f"{'stance':<12} {'baseline':>10} {'baseline_%':>12} "
        f"{'rl':>10} {'rl_%':>10} {'delta':>10} {'delta_pp':>12}"
    )
    for label in labels:
        print(
            f"{label:<12} "
            f"{comparison['baseline']['counts'][label]:>10} "
            f"{comparison['baseline']['percentages'][label] * 100:>11.2f}% "
            f"{comparison['rl']['counts'][label]:>10} "
            f"{comparison['rl']['percentages'][label] * 100:>9.2f}% "
            f"{comparison['delta']['counts'][label]:>10} "
            f"{comparison['delta']['percentage_points'][label]:>11.2f}"
        )

    print("\nBy speaker")
    print("-" * 79)
    for speaker in ("YOU", "THEM", "OTHER"):
        baseline = comparison["baseline"]["by_speaker"].get(speaker)
        rl = comparison["rl"]["by_speaker"].get(speaker)
        if not baseline and not rl:
            continue
        base_total = sum(baseline["counts"].values()) if baseline else 0
        rl_total = sum(rl["counts"].values()) if rl else 0
        if base_total == 0 and rl_total == 0:
            continue
        print(f"[{speaker}] baseline={base_total}, rl={rl_total}")
        for label in labels:
            print(
                f"  {label:<10} "
                f"base={baseline['counts'][label]:>4} ({baseline['percentages'][label] * 100:>6.2f}%) "
                f"rl={rl['counts'][label]:>4} ({rl['percentages'][label] * 100:>6.2f}%) "
                f"delta={rl['counts'][label] - baseline['counts'][label]:>4}"
            )


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def plot_comparison(comparison: dict, out_dir: Path) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib is not installed, writing SVG plots instead.")
        return write_svg_plots(comparison, out_dir)

    labels = comparison["labels"]
    baseline_counts = [comparison["baseline"]["counts"][label] for label in labels]
    rl_counts = [comparison["rl"]["counts"][label] for label in labels]
    delta_pp = [comparison["delta"]["percentage_points"][label] for label in labels]

    out_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    width = 0.36
    x = range(len(labels))

    axes[0].bar([i - width / 2 for i in x], baseline_counts, width=width, label="baseline")
    axes[0].bar([i + width / 2 for i in x], rl_counts, width=width, label="rl")
    axes[0].set_title("Stance Counts")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Count")
    axes[0].legend()

    colors = ["#2a9d8f" if value >= 0 else "#e76f51" for value in delta_pp]
    axes[1].bar(labels, delta_pp, color=colors)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_title("RL - Baseline Percentage Points")
    axes[1].set_ylabel("Percentage points")

    fig.tight_layout()
    comparison_plot = out_dir / "stance_distribution_comparison.png"
    fig.savefig(comparison_plot, dpi=200, bbox_inches="tight")
    plt.close(fig)
    written_paths.append(comparison_plot)

    fig, ax = plt.subplots(figsize=(8, 5))
    base_pct = [comparison["baseline"]["percentages"][label] * 100 for label in labels]
    rl_pct = [comparison["rl"]["percentages"][label] * 100 for label in labels]
    ax.plot(labels, base_pct, marker="o", label="baseline")
    ax.plot(labels, rl_pct, marker="o", label="rl")
    ax.set_title("Normalized Stance Distribution")
    ax.set_ylabel("Percent")
    ax.legend()
    fig.tight_layout()
    percent_plot = out_dir / "stance_distribution_percentages.png"
    fig.savefig(percent_plot, dpi=200, bbox_inches="tight")
    plt.close(fig)
    written_paths.append(percent_plot)

    return written_paths


def write_svg_plots(comparison: dict, out_dir: Path) -> list[Path]:
    labels = comparison["labels"]
    baseline_counts = [comparison["baseline"]["counts"][label] for label in labels]
    rl_counts = [comparison["rl"]["counts"][label] for label in labels]
    delta_pp = [comparison["delta"]["percentage_points"][label] for label in labels]
    base_pct = [comparison["baseline"]["percentages"][label] * 100 for label in labels]
    rl_pct = [comparison["rl"]["percentages"][label] * 100 for label in labels]

    out_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []

    comparison_svg = out_dir / "stance_distribution_comparison.svg"
    comparison_svg.write_text(
        build_grouped_bar_svg(labels, baseline_counts, rl_counts, "baseline", "rl", "Stance Counts"),
        encoding="utf-8",
    )
    written_paths.append(comparison_svg)

    delta_svg = out_dir / "stance_distribution_delta.svg"
    delta_svg.write_text(
        build_delta_bar_svg(labels, delta_pp, "RL - Baseline Percentage Points"),
        encoding="utf-8",
    )
    written_paths.append(delta_svg)

    percentage_svg = out_dir / "stance_distribution_percentages.svg"
    percentage_svg.write_text(
        build_line_svg(labels, base_pct, rl_pct, "baseline", "rl", "Normalized Stance Distribution"),
        encoding="utf-8",
    )
    written_paths.append(percentage_svg)

    return written_paths


def build_grouped_bar_svg(
    labels: list[str],
    series_a: list[int],
    series_b: list[int],
    label_a: str,
    label_b: str,
    title: str,
) -> str:
    width = 900
    height = 520
    margin_left = 90
    margin_right = 40
    margin_top = 60
    margin_bottom = 90
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_value = max(series_a + series_b + [1])
    group_width = plot_width / max(len(labels), 1)
    bar_width = group_width * 0.28
    color_a = "#457b9d"
    color_b = "#e76f51"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-size="24" font-family="Arial">{title}</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="black"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="black"/>',
    ]

    for tick in range(6):
        value = max_value * tick / 5
        y = margin_top + plot_height - (value / max_value) * plot_height
        parts.append(
            f'<line x1="{margin_left - 5}" y1="{y:.1f}" x2="{margin_left + plot_width}" y2="{y:.1f}" stroke="#dddddd"/>'
        )
        parts.append(
            f'<text x="{margin_left - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="12" font-family="Arial">{value:.0f}</text>'
        )

    for idx, label in enumerate(labels):
        group_x = margin_left + idx * group_width + group_width / 2
        x_a = group_x - bar_width - 4
        x_b = group_x + 4
        h_a = (series_a[idx] / max_value) * plot_height
        h_b = (series_b[idx] / max_value) * plot_height
        y_a = margin_top + plot_height - h_a
        y_b = margin_top + plot_height - h_b
        parts.append(
            f'<rect x="{x_a:.1f}" y="{y_a:.1f}" width="{bar_width:.1f}" height="{h_a:.1f}" fill="{color_a}"/>'
        )
        parts.append(
            f'<rect x="{x_b:.1f}" y="{y_b:.1f}" width="{bar_width:.1f}" height="{h_b:.1f}" fill="{color_b}"/>'
        )
        parts.append(
            f'<text x="{group_x:.1f}" y="{height - 36}" text-anchor="middle" font-size="14" font-family="Arial">{label}</text>'
        )
        parts.append(
            f'<text x="{x_a + bar_width / 2:.1f}" y="{y_a - 8:.1f}" text-anchor="middle" font-size="12" font-family="Arial">{series_a[idx]}</text>'
        )
        parts.append(
            f'<text x="{x_b + bar_width / 2:.1f}" y="{y_b - 8:.1f}" text-anchor="middle" font-size="12" font-family="Arial">{series_b[idx]}</text>'
        )

    parts.extend(
        [
            f'<rect x="{width - 220}" y="52" width="16" height="16" fill="{color_a}"/>',
            f'<text x="{width - 196}" y="65" font-size="13" font-family="Arial">{label_a}</text>',
            f'<rect x="{width - 120}" y="52" width="16" height="16" fill="{color_b}"/>',
            f'<text x="{width - 96}" y="65" font-size="13" font-family="Arial">{label_b}</text>',
            "</svg>",
        ]
    )
    return "\n".join(parts)


def build_delta_bar_svg(labels: list[str], values: list[float], title: str) -> str:
    width = 900
    height = 520
    margin_left = 90
    margin_right = 40
    margin_top = 60
    margin_bottom = 90
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_abs = max(max((abs(v) for v in values), default=1), 1)
    zero_y = margin_top + plot_height / 2
    bar_width = plot_width / max(len(labels), 1) * 0.45

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-size="24" font-family="Arial">{title}</text>',
        f'<line x1="{margin_left}" y1="{zero_y:.1f}" x2="{margin_left + plot_width}" y2="{zero_y:.1f}" stroke="black"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="black"/>',
    ]

    for tick in range(-2, 3):
        value = max_abs * tick / 2
        y = zero_y - (value / max_abs) * (plot_height / 2)
        parts.append(
            f'<line x1="{margin_left - 5}" y1="{y:.1f}" x2="{margin_left + plot_width}" y2="{y:.1f}" stroke="#dddddd"/>'
        )
        parts.append(
            f'<text x="{margin_left - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="12" font-family="Arial">{value:.1f}</text>'
        )

    step = plot_width / max(len(labels), 1)
    for idx, label in enumerate(labels):
        center_x = margin_left + idx * step + step / 2
        value = values[idx]
        bar_height = abs(value) / max_abs * (plot_height / 2)
        y = zero_y - bar_height if value >= 0 else zero_y
        color = "#2a9d8f" if value >= 0 else "#e76f51"
        parts.append(
            f'<rect x="{center_x - bar_width / 2:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{color}"/>'
        )
        value_y = y - 8 if value >= 0 else y + bar_height + 18
        parts.append(
            f'<text x="{center_x:.1f}" y="{value_y:.1f}" text-anchor="middle" font-size="12" font-family="Arial">{value:.2f}</text>'
        )
        parts.append(
            f'<text x="{center_x:.1f}" y="{height - 36}" text-anchor="middle" font-size="14" font-family="Arial">{label}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def build_line_svg(
    labels: list[str],
    series_a: list[float],
    series_b: list[float],
    label_a: str,
    label_b: str,
    title: str,
) -> str:
    width = 900
    height = 520
    margin_left = 90
    margin_right = 40
    margin_top = 60
    margin_bottom = 90
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_value = max(series_a + series_b + [1.0])
    color_a = "#457b9d"
    color_b = "#e76f51"

    def to_points(values: list[float]) -> list[tuple[float, float]]:
        step = plot_width / max(len(labels) - 1, 1)
        points = []
        for idx, value in enumerate(values):
            x = margin_left + idx * step
            y = margin_top + plot_height - (value / max_value) * plot_height
            points.append((x, y))
        return points

    def polyline(points: list[tuple[float, float]]) -> str:
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)

    points_a = to_points(series_a)
    points_b = to_points(series_b)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-size="24" font-family="Arial">{title}</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="black"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="black"/>',
    ]

    for tick in range(6):
        value = max_value * tick / 5
        y = margin_top + plot_height - (value / max_value) * plot_height
        parts.append(
            f'<line x1="{margin_left - 5}" y1="{y:.1f}" x2="{margin_left + plot_width}" y2="{y:.1f}" stroke="#dddddd"/>'
        )
        parts.append(
            f'<text x="{margin_left - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="12" font-family="Arial">{value:.1f}</text>'
        )

    for idx, label in enumerate(labels):
        x = margin_left + idx * (plot_width / max(len(labels) - 1, 1))
        parts.append(
            f'<text x="{x:.1f}" y="{height - 36}" text-anchor="middle" font-size="14" font-family="Arial">{label}</text>'
        )

    parts.append(
        f'<polyline fill="none" stroke="{color_a}" stroke-width="3" points="{polyline(points_a)}"/>'
    )
    parts.append(
        f'<polyline fill="none" stroke="{color_b}" stroke-width="3" points="{polyline(points_b)}"/>'
    )

    for points, color, values in ((points_a, color_a, series_a), (points_b, color_b, series_b)):
        for (x, y), value in zip(points, values):
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
            parts.append(
                f'<text x="{x:.1f}" y="{y - 10:.1f}" text-anchor="middle" font-size="12" font-family="Arial">{value:.2f}</text>'
            )

    parts.extend(
        [
            f'<rect x="{width - 220}" y="52" width="16" height="16" fill="{color_a}"/>',
            f'<text x="{width - 196}" y="65" font-size="13" font-family="Arial">{label_a}</text>',
            f'<rect x="{width - 120}" y="52" width="16" height="16" fill="{color_b}"/>',
            f'<text x="{width - 96}" y="65" font-size="13" font-family="Arial">{label_b}</text>',
            "</svg>",
        ]
    )
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--rl", type=Path, required=True)
    parser.add_argument(
        "--rl-continued-only",
        action="store_true",
        help="Only count RL annotations whose turn_index is >= continued_from_turn_index.",
    )
    parser.add_argument(
        "--baseline-continued-only",
        action="store_true",
        help="Apply the same continued-only filtering to baseline if that file also has continued metadata.",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Do not deduplicate repeated (dialogue_index, perspective) entries.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/home/swang4/866/data/baseline_2/analysis"),
    )
    args = parser.parse_args()

    dedupe = not args.no_dedupe
    baseline_summary = summarize_payload(
        args.baseline,
        continued_only=args.baseline_continued_only,
        dedupe=dedupe,
    )
    rl_summary = summarize_payload(
        args.rl,
        continued_only=args.rl_continued_only,
        dedupe=dedupe,
    )
    comparison = build_comparison(baseline_summary, rl_summary)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.out_dir / "stance_distribution_comparison.json", comparison)
    print_table(comparison)

    plot_paths = plot_comparison(comparison, args.out_dir)
    print(f"\nSaved JSON summary to {args.out_dir / 'stance_distribution_comparison.json'}")
    for path in plot_paths:
        print(f"Saved plot to {path}")


if __name__ == "__main__":
    main()
