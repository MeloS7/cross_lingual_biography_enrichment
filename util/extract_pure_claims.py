#!/usr/bin/env python3
"""Extract compact claim-only JSONL files.

The input files produced by claim extraction contain intermediate fields such as
excerpts, source sentences, and structured claim slots. This script keeps only
the record metadata and final claim texts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


META_FIELDS = ("qid", "name", "occupations", "country_group", "target_lang")


def iter_jsonl_files(path: Path, suffix: str) -> Iterable[Path]:
    if path.is_file():
        if path.suffix != ".jsonl":
            raise ValueError(f"Input file is not a .jsonl file: {path}")
        yield path
        return

    if not path.is_dir():
        raise ValueError(f"Input path does not exist: {path}")

    for jsonl_file in sorted(path.rglob("*.jsonl")):
        if jsonl_file.stem.endswith(suffix):
            continue
        yield jsonl_file


def extract_claims(record: dict) -> list[dict[str, str]]:
    pure_claims: list[dict[str, str]] = []

    for result in record.get("decomposition_results") or []:
        if not isinstance(result, dict):
            continue
        for claim_obj in result.get("claims") or []:
            claim_text = None
            if isinstance(claim_obj, dict):
                claim_text = claim_obj.get("claim")
            elif isinstance(claim_obj, str):
                claim_text = claim_obj

            if isinstance(claim_text, str) and claim_text.strip():
                pure_claims.append({"claim": claim_text.strip()})

    return pure_claims


def make_pure_record(record: dict) -> dict:
    pure_record = {field: record.get(field) for field in META_FIELDS}
    pure_record["claims"] = extract_claims(record)
    return pure_record


def output_path_for(input_file: Path, input_root: Path, output_root: Path, suffix: str) -> Path:
    if input_root.is_file():
        relative = input_file.name
    else:
        relative = input_file.relative_to(input_root)

    relative_path = Path(relative)
    output_name = f"{relative_path.stem}{suffix}{relative_path.suffix}"
    return output_root / relative_path.with_name(output_name)


def convert_file(input_file: Path, output_file: Path) -> tuple[int, int]:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    records_n = 0
    claims_n = 0
    with input_file.open(encoding="utf-8") as src, output_file.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {input_file}:{line_no}: {exc}") from exc

            pure_record = make_pure_record(record)
            records_n += 1
            claims_n += len(pure_record["claims"])
            dst.write(json.dumps(pure_record, ensure_ascii=False) + "\n")

    return records_n, claims_n


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create pure claim JSONL files with qid/name/occupations/country_group/target_lang/claims.claim only."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input .jsonl file or directory containing .jsonl files.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where converted files will be written.",
    )
    parser.add_argument(
        "--suffix",
        default="_pure_claims",
        help="Suffix appended before .jsonl in output filenames. Default: _pure_claims",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input
    output_root = args.output_dir

    files = list(iter_jsonl_files(input_root, args.suffix))
    if not files:
        raise SystemExit(f"No .jsonl files found under {input_root}")

    total_records = 0
    total_claims = 0
    for input_file in files:
        output_file = output_path_for(input_file, input_root, output_root, args.suffix)
        records_n, claims_n = convert_file(input_file, output_file)
        total_records += records_n
        total_claims += claims_n
        print(f"{input_file} -> {output_file} ({records_n} records, {claims_n} claims)")

    print(f"Done: {len(files)} files, {total_records} records, {total_claims} claims")


if __name__ == "__main__":
    main()
