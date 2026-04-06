from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


SVI_QIDS = [f"Q{i}" for i in range(1, 15)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize utility and SVI evaluation outputs for a negotiation batch."
    )
    parser.add_argument("--utility-summary", type=Path, required=True)
    parser.add_argument("--svi-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", type=str, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def round_or_none(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(mean(values))


def summarize_utility(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload.get("results", [])
    valid_results = [r for r in results if r.get("allocation_valid")]
    deal_results = [r for r in results if r.get("deal_reached")]

    agent_utility_values: dict[str, list[float]] = defaultdict(list)
    agent_issue_values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    social_welfare_values: list[float] = []
    utility_gap_values: list[float] = []
    nash_product_values: list[float] = []
    min_utility_values: list[float] = []

    for record in valid_results:
        utilities = record.get("utilities", {})
        totals: dict[str, float] = {}
        for agent_id, agent_payload in utilities.items():
            total_utility = float(agent_payload.get("total_utility", 0))
            totals[agent_id] = total_utility
            agent_utility_values[agent_id].append(total_utility)

            breakdown = agent_payload.get("breakdown", {})
            for issue, issue_value in breakdown.items():
                agent_issue_values[agent_id][issue].append(float(issue_value))

        if totals:
            total_list = list(totals.values())
            social_welfare_values.append(sum(total_list))
            min_utility_values.append(min(total_list))
            if len(total_list) >= 2:
                utility_gap_values.append(abs(total_list[0] - total_list[1]))
                nash_product_values.append(total_list[0] * total_list[1])

    utility_by_agent = {}
    for agent_id in sorted(agent_utility_values):
        utility_by_agent[agent_id] = {
            "mean_total_utility": round_or_none(safe_mean(agent_utility_values[agent_id])),
            "mean_issue_utility": {
                issue: round_or_none(safe_mean(values))
                for issue, values in sorted(agent_issue_values[agent_id].items())
            },
            "num_valid_records": len(agent_utility_values[agent_id]),
        }

    return {
        "num_files": payload.get("num_files"),
        "num_deals_detected": payload.get("num_deals_detected"),
        "num_valid_allocations": payload.get("num_valid_allocations"),
        "num_failed_extractions": payload.get("num_failed_extractions"),
        "deal_rate": round_or_none(len(deal_results) / len(results) if results else None),
        "valid_allocation_rate": round_or_none(
            len(valid_results) / len(results) if results else None
        ),
        "mean_social_welfare": round_or_none(safe_mean(social_welfare_values)),
        "mean_min_utility": round_or_none(safe_mean(min_utility_values)),
        "mean_utility_gap": round_or_none(safe_mean(utility_gap_values)),
        "mean_nash_product": round_or_none(safe_mean(nash_product_values)),
        "utility_by_agent": utility_by_agent,
    }


def summarize_svi(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload.get("results", [])
    successful = [r for r in results if r.get("status") == "ok"]
    qid_values: dict[str, list[float]] = defaultdict(list)
    pov_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    overall_means: list[float] = []

    for record in successful:
        prediction = record.get("prediction", {})
        svi_scores = prediction.get("svi_scores", {})
        row_scores: list[float] = []
        pov_agent_id = record.get("pov_agent_id", "unknown")
        for qid in SVI_QIDS:
            q_payload = svi_scores.get(qid)
            if not isinstance(q_payload, dict):
                continue
            score = q_payload.get("score")
            if score is None:
                continue
            score_value = float(score)
            qid_values[qid].append(score_value)
            pov_values[pov_agent_id][qid].append(score_value)
            row_scores.append(score_value)
        if row_scores:
            overall_means.append(float(mean(row_scores)))

    question_means = {
        qid: round_or_none(safe_mean(values)) for qid, values in sorted(qid_values.items())
    }
    by_pov_agent = {}
    for pov_agent_id in sorted(pov_values):
        per_q = {
            qid: round_or_none(safe_mean(values))
            for qid, values in sorted(pov_values[pov_agent_id].items())
        }
        all_values = [value for values in pov_values[pov_agent_id].values() for value in values]
        by_pov_agent[pov_agent_id] = {
            "mean_overall_svi": round_or_none(safe_mean(all_values)),
            "question_means": per_q,
        }

    return {
        "num_rows": payload.get("num_rows"),
        "num_successful_rows": payload.get("num_successful_rows"),
        "num_failed_rows": payload.get("num_failed_rows"),
        "success_rate": round_or_none(len(successful) / len(results) if results else None),
        "mean_overall_svi": round_or_none(safe_mean(overall_means)),
        "question_means": question_means,
        "by_pov_agent": by_pov_agent,
    }


def write_summary_csv(path: Path, tag: str, utility_summary: dict[str, Any], svi_summary: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []

    def add_row(metric: str, value: Any, group: str = "overall") -> None:
        rows.append({"tag": tag, "group": group, "metric": metric, "value": value})

    add_row("deal_rate", utility_summary.get("deal_rate"))
    add_row("valid_allocation_rate", utility_summary.get("valid_allocation_rate"))
    add_row("mean_social_welfare", utility_summary.get("mean_social_welfare"))
    add_row("mean_min_utility", utility_summary.get("mean_min_utility"))
    add_row("mean_utility_gap", utility_summary.get("mean_utility_gap"))
    add_row("mean_nash_product", utility_summary.get("mean_nash_product"))
    add_row("mean_overall_svi", svi_summary.get("mean_overall_svi"))
    add_row("svi_success_rate", svi_summary.get("success_rate"))

    for agent_id, agent_payload in utility_summary.get("utility_by_agent", {}).items():
        add_row("mean_total_utility", agent_payload.get("mean_total_utility"), group=agent_id)
        for issue, issue_value in agent_payload.get("mean_issue_utility", {}).items():
            add_row(f"mean_{issue.lower()}_utility", issue_value, group=agent_id)

    for qid, qmean in svi_summary.get("question_means", {}).items():
        add_row(f"{qid}_mean", qmean, group="svi_question")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tag", "group", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, tag: str, utility_summary: dict[str, Any], svi_summary: dict[str, Any]) -> None:
    lines = [
        f"# Psych/Econ Outcome Summary: {tag}",
        "",
        "## Utility",
        "",
        f"- Deal rate: {utility_summary.get('deal_rate')}",
        f"- Valid allocation rate: {utility_summary.get('valid_allocation_rate')}",
        f"- Mean social welfare: {utility_summary.get('mean_social_welfare')}",
        f"- Mean min utility: {utility_summary.get('mean_min_utility')}",
        f"- Mean utility gap: {utility_summary.get('mean_utility_gap')}",
        f"- Mean Nash product: {utility_summary.get('mean_nash_product')}",
        "",
        "## SVI",
        "",
        f"- Mean overall SVI: {svi_summary.get('mean_overall_svi')}",
        f"- Success rate: {svi_summary.get('success_rate')}",
        "",
        "## Agent Utility Means",
        "",
    ]

    for agent_id, agent_payload in utility_summary.get("utility_by_agent", {}).items():
        lines.append(
            f"- {agent_id}: total={agent_payload.get('mean_total_utility')}, "
            + ", ".join(
                f"{issue}={value}" for issue, value in agent_payload.get("mean_issue_utility", {}).items()
            )
        )

    lines.extend(["", "## SVI Question Means", ""])
    for qid, value in svi_summary.get("question_means", {}).items():
        lines.append(f"- {qid}: {value}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    utility_payload = load_json(args.utility_summary)
    svi_payload = load_json(args.svi_summary)

    utility_summary = summarize_utility(utility_payload)
    svi_summary = summarize_svi(svi_payload)

    final_summary = {
        "tag": args.tag,
        "utility_summary_path": str(args.utility_summary),
        "svi_summary_path": str(args.svi_summary),
        "utility": utility_summary,
        "svi": svi_summary,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.tag}_psych_econ_summary.json"
    csv_path = args.output_dir / f"{args.tag}_psych_econ_summary.csv"
    md_path = args.output_dir / f"{args.tag}_psych_econ_summary.md"

    json_path.write_text(json.dumps(final_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_summary_csv(csv_path, args.tag, utility_summary, svi_summary)
    write_markdown(md_path, args.tag, utility_summary, svi_summary)

    print(f"Saved JSON summary to {json_path}")
    print(f"Saved CSV summary to {csv_path}")
    print(f"Saved Markdown summary to {md_path}")


if __name__ == "__main__":
    main()
