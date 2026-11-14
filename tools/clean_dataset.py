"""Write a deduplicated dataset copy plus a Markdown quality report.

The input file is never modified. ARFF inputs keep their original header so
nominal attributes and the relation name survive the round trip.
"""

import argparse
import datetime
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.core import DataYabCore

ATTRIBUTE = re.compile(
    r"^\s*@attribute\s+(?:'((?:[^'\\]|\\.)*)'|\"((?:[^\"\\]|\\.)*)\"|(\S+))"
    r"\s+(.*)$",
    re.IGNORECASE,
)


def split_values(line):
    """Split one ARFF data row, respecting single or double quotes."""
    values, current, quote = [], [], None
    index = 0
    while index < len(line):
        char = line[index]
        if quote:
            if char == "\\" and index + 1 < len(line):
                current.append(line[index + 1])
                index += 2
                continue
            if char == quote:
                quote = None
            else:
                current.append(char)
        elif char in ("'", '"'):
            quote = char
        elif char == ",":
            values.append("".join(current).strip())
            current = []
        else:
            current.append(char)
        index += 1
    values.append("".join(current).strip())
    return values


def parse_arff(text):
    lines = text.splitlines()
    header, rows = [], []
    in_data = False
    for line in lines:
        if in_data:
            if line.strip() and not line.lstrip().startswith("%"):
                rows.append(line)
        else:
            header.append(line)
            if line.strip().lower() == "@data":
                in_data = True
    if not in_data:
        raise ValueError("ARFF file has no @data section.")
    return header, rows


def arff_attributes(header):
    attributes = []
    for line in header:
        match = ATTRIBUTE.match(line)
        if not match:
            continue
        name = match.group(1) or match.group(2) or match.group(3)
        declaration = match.group(4).strip()
        nominal = None
        if declaration.startswith("{") and declaration.endswith("}"):
            nominal = set(split_values(declaration[1:-1]))
        attributes.append((name, declaration, nominal))
    return attributes


def arff_checks(header, rows, attributes):
    width_issues = sum(
        1 for row in rows if len(split_values(row)) != len(attributes)
    )
    invalid = 0
    unknown = {}
    for row in rows:
        values = split_values(row)
        if len(values) != len(attributes):
            continue
        for (name, _, nominal), value in zip(attributes, values):
            if nominal is None or value == "?":
                continue
            if value not in nominal:
                invalid += 1
                unknown[name] = unknown.get(name, 0) + 1
    return width_issues, invalid, unknown


def dedupe_arff_rows(rows):
    seen = set()
    kept = []
    for row in rows:
        key = row.strip()
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)
    return kept


def build_report(
    source,
    relation,
    original,
    cleaned,
    quality,
    target,
    removed,
    arff_summary,
    output_path,
):
    lines = [
        "# Dataset quality report",
        "",
        f"- Source: `{source}`",
        f"- Generated: {datetime.datetime.now():%Y-%m-%d %H:%M}",
        f"- Dataset relation: {relation or 'n/a'}",
        f"- Original rows: {len(original):,}",
        f"- Cleaned rows: {len(cleaned):,}",
        f"- Columns: {len(original.columns)}",
        f"- Duplicate rows removed: {removed:,}",
        f"- Cleaned file: `{output_path}`",
        "",
        "## Issues detected",
        "",
        "| Issue | Count | Action |",
        "| --- | --- | --- |",
        f"| Exact duplicate rows | {quality['duplicate_rows']:,} ({quality['duplicate_percent']}%) | Removed in the cleaned copy |",
        f"| Missing cells | {quality['missing_total']:,} | Kept; train-time imputation handles features |",
        f"| Constant columns | {len(quality['constant_columns'])} | Kept for explicit review |",
    ]
    if arff_summary is not None:
        width_issues, invalid_values, unknown = arff_summary
        lines.append(
            f"| ARFF rows with wrong value count | {width_issues:,} | Reported only |"
        )
        lines.append(
            f"| Values outside declared nominal sets | {invalid_values:,} | Reported only |"
        )
        if unknown:
            details = ", ".join(
                f"{name} ({count:,})" for name, count in sorted(unknown.items())
            )
            lines.append(f"| Affected attributes | {details} | Review source coding |")
    lines.extend(["", "## Class balance (cleaned copy)", ""])
    balance = {}
    if target and target in cleaned.columns:
        counts = cleaned[target].value_counts(dropna=True)
        balance = {str(k): int(v) for k, v in counts.head(25).items()}
    if balance:
        lines.extend(["| Class | Rows | Share |", "| --- | --- | --- |"])
        total = sum(balance.values()) or 1
        for label, count in balance.items():
            lines.append(f"| {label} | {count:,} | {count / total * 100:.1f}% |")
    else:
        lines.append(
            "No target was supplied. Pass `--target COLUMN` for class balance."
        )

    lines.extend(
        [
            "",
            "## Correlated columns",
            "",
        ]
    )
    if quality["top_correlations"]:
        lines.extend(["| Column A | Column B | |r| |", "| --- | --- | --- |"])
        for pair in quality["top_correlations"]:
            lines.append(
                f"| {pair['feature_a']} | {pair['feature_b']} | {pair['correlation']:.4f} |"
            )
    else:
        lines.append("No numeric column pair could be compared.")

    lines.extend(
        [
            "",
            "## Column summary",
            "",
            "| Column | Type | Unique | Missing |",
            "| --- | --- | --- | --- |",
        ]
    )
    for column in original.columns:
        lines.append(
            f"| {column} | {original[column].dtype} | "
            f"{original[column].nunique(dropna=True):,} | "
            f"{int(original[column].isna().sum()):,} |"
        )

    lines.extend(
        [
            "",
            "## Cleaning applied",
            "",
            "- Removed exact duplicate rows from the cleaned copy only.",
            "- Preserved every column and every non-duplicate row.",
            "- Left the source file untouched.",
            "- Missing values and outliers were not modified automatically.",
            "",
            "## Next steps",
            "",
            "- Review duplicates before modelling: repeated observations may be genuine or a collection artefact.",
            "- Use a target column and compare models on a held-out split.",
            "- Export models and reports separately from the raw dataset.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Write a deduplicated dataset copy and a quality report."
    )
    parser.add_argument("input", help="Dataset file (ARFF, CSV, Excel, JSON, Parquet, ARR)")
    parser.add_argument("--output", help="Cleaned dataset path (default: <stem>.cleaned<ext>)")
    parser.add_argument("--report", help="Report path (default: <stem>.quality_report.md)")
    parser.add_argument("--target", help="Target column for class balance reporting")
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Do not remove duplicate rows in the cleaned copy",
    )
    parser.add_argument(
        "--csv", action="store_true", help="Also write a CSV copy of the cleaned data"
    )
    args = parser.parse_args()

    source = Path(args.input)
    output = (
        Path(args.output)
        if args.output
        else source.with_name(source.stem + ".cleaned" + source.suffix)
    )
    report_path = (
        Path(args.report)
        if args.report
        else source.with_name(source.stem + ".quality_report.md")
    )

    core = DataYabCore()
    loaded = core.load_file(source)
    if not loaded["success"]:
        raise SystemExit(loaded["message"])
    original = core.df.copy(deep=True)
    relation = None
    arff_summary = None
    text = None
    if source.suffix.lower() == ".arff":
        text = source.read_text(encoding="utf-8-sig", errors="replace")
        header, rows = parse_arff(text)
        for line in header:
            if line.lower().startswith("@relation"):
                relation = line.split(None, 1)[1].strip() if " " in line else ""
        attributes = arff_attributes(header)
        arff_summary = arff_checks(header, rows, attributes)

    if args.target:
        if args.target not in core.df:
            raise SystemExit(f"Target column not found: {args.target}")
        core.target_col = args.target
    quality = core.get_dataset_quality()
    cleaned = (
        original.copy(deep=True)
        if args.keep_duplicates
        else original.drop_duplicates().reset_index(drop=True)
    )
    removed = len(original) - len(cleaned)

    output.parent.mkdir(parents=True, exist_ok=True)
    if text is not None:
        kept_rows = rows if args.keep_duplicates else dedupe_arff_rows(rows)
        output.write_text(
            "\n".join(header + kept_rows) + "\n", encoding="utf-8"
        )
    else:
        core.df = cleaned
        core.export_cleaned_data(str(output))
        core.df = original

    if args.csv:
        csv_path = output.with_suffix(".csv")
        cleaned.to_csv(csv_path, index=False)

    report_path.write_text(
        build_report(
            source,
            relation,
            original,
            cleaned,
            quality,
            args.target,
            removed,
            arff_summary,
            output,
        ),
        encoding="utf-8",
    )
    print(f"Cleaned dataset: {output}")
    if args.csv:
        print(f"CSV copy: {output.with_suffix('.csv')}")
    print(f"Quality report: {report_path}")
    print(f"Rows: {len(original):,} -> {len(cleaned):,} (removed {removed:,})")


if __name__ == "__main__":
    main()
