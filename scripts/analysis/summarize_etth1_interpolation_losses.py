#!/usr/bin/env python3
"""Summarize ETTh1 raw, PCHIP15, FACL, and PS forecasting results."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


MODELS = (
    "PatchTST",
    "Informer",
    "DLinear",
    "iTransformer",
    "TimeMixer",
    "TimesNet",
    "FEDformer",
    "TFPS",
)
METRIC_NAMES = ("mae", "mse", "rmse", "mape", "mspe")
CONDITIONS = ("raw_mse", "pchip_mse", "pchip_facl", "pchip_ps")


def identify_model(name):
    for model in MODELS:
        if f"_{model}_" in name:
            return model
    return None


def identify_condition(name):
    if "PCHIP15" not in name:
        return "raw_mse"
    if "_facl_" in name:
        return "pchip_facl"
    if "_ps_" in name:
        return "pchip_ps"
    return "pchip_mse"


def read_metrics(result_dir, condition):
    json_path = result_dir / "metrics_real_only.json"
    if condition.startswith("pchip") and json_path.is_file():
        with json_path.open(encoding="utf-8") as handle:
            metrics = json.load(handle)
        return {name: float(metrics[name]) for name in METRIC_NAMES}, "metrics_real_only.json"

    metrics_path = result_dir / "metrics.npy"
    values = np.load(metrics_path)
    if values.shape != (len(METRIC_NAMES),):
        raise ValueError(f"Expected five metrics in {metrics_path}, got shape {values.shape}.")
    return dict(zip(METRIC_NAMES, map(float, values))), "metrics.npy"


def percent_change(candidate, baseline):
    if baseline == 0:
        return None
    return 100.0 * (candidate - baseline) / baseline


def format_number(value):
    return "-" if value is None else f"{value:.6f}"


def format_percent(value):
    return "-" if value is None else f"{value:+.2f}%"


def print_table(headers, rows):
    string_rows = [[str(value) for value in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in string_rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def render(row):
        return " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))

    print(render(headers))
    print("-+-".join("-" * width for width in widths))
    for row in string_rows:
        print(render(row))


def main():
    parser = argparse.ArgumentParser(
        description="Compare ETTh1 raw 96->96 and PCHIP15 384->384 experiment results."
    )
    parser.add_argument("--results_dir", default="results", help="Path to the TSLib results directory.")
    parser.add_argument("--csv", default=None, help="Optional CSV path for the full experiment table.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    records = []
    skipped = []
    for result_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        model = identify_model(result_dir.name)
        if model is None:
            continue
        condition = identify_condition(result_dir.name)
        try:
            metrics, source = read_metrics(result_dir, condition)
        except (OSError, ValueError, KeyError) as exc:
            skipped.append((result_dir.name, str(exc)))
            continue
        records.append(
            {
                "directory": result_dir.name,
                "model": model,
                "condition": condition,
                "source": source,
                **metrics,
            }
        )

    if not records:
        raise RuntimeError(f"No recognized ETTh1 experiment results found in {results_dir}.")

    print("\nComplete experiment results")
    print_table(
        ("Model", "Condition", "MAE", "MSE", "RMSE", "Metric source"),
        [
            (
                record["model"],
                record["condition"],
                format_number(record["mae"]),
                format_number(record["mse"]),
                format_number(record["rmse"]),
                record["source"],
            )
            for record in sorted(records, key=lambda record: (MODELS.index(record["model"]), record["condition"]))
        ],
    )

    grouped = {}
    duplicates = []
    for record in records:
        key = (record["model"], record["condition"])
        if key in grouped:
            duplicates.append((key, grouped[key]["directory"], record["directory"]))
        else:
            grouped[key] = record

    summary_rows = []
    for model in MODELS:
        raw = grouped.get((model, "raw_mse"))
        pchip = grouped.get((model, "pchip_mse"))
        facl = grouped.get((model, "pchip_facl"))
        ps = grouped.get((model, "pchip_ps"))
        summary_rows.append(
            (
                model,
                format_number(raw["mse"] if raw else None),
                format_number(pchip["mse"] if pchip else None),
                format_percent(percent_change(pchip["mse"], raw["mse"]) if raw and pchip else None),
                format_number(facl["mse"] if facl else None),
                format_percent(percent_change(facl["mse"], pchip["mse"]) if facl and pchip else None),
                format_number(ps["mse"] if ps else None),
                format_percent(percent_change(ps["mse"], pchip["mse"]) if ps and pchip else None),
            )
        )

    print("\nPaired MSE comparison (negative percentage means improvement)")
    print_table(
        (
            "Model",
            "Raw",
            "PCHIP",
            "PCHIP vs Raw",
            "FACL",
            "FACL vs PCHIP",
            "PS",
            "PS vs PCHIP",
        ),
        summary_rows,
    )

    for condition, baseline_condition, label in (
        ("pchip_mse", "raw_mse", "PCHIP vs Raw"),
        ("pchip_facl", "pchip_mse", "FACL vs PCHIP"),
        ("pchip_ps", "pchip_mse", "PS vs PCHIP"),
    ):
        changes = [
            percent_change(grouped[(model, condition)]["mse"], grouped[(model, baseline_condition)]["mse"])
            for model in MODELS
            if (model, condition) in grouped and (model, baseline_condition) in grouped
        ]
        if changes:
            improved = sum(change < 0 for change in changes)
            print(
                f"{label}: {improved}/{len(changes)} models improve in MSE; "
                f"mean change {np.mean(changes):+.2f}%."
            )

    if duplicates:
        print("\nDuplicate model/condition records (only the first is used in paired comparisons):")
        for key, first, second in duplicates:
            print(f"  {key}: {first} | {second}")
    if skipped:
        print("\nUnreadable result directories:")
        for directory, reason in skipped:
            print(f"  {directory}: {reason}")

    if args.csv:
        csv_path = Path(args.csv)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("model", "condition", "mae", "mse", "rmse", "mape", "mspe", "source", "directory"),
            )
            writer.writeheader()
            writer.writerows(records)
        print(f"\nWrote full experiment table to {csv_path}")


if __name__ == "__main__":
    main()
