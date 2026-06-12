#!/usr/bin/env python3
import csv
import json
import argparse
import os
from datetime import date

# 주의
# error_category는 직접 채워야함
# 
def row_to_doc(row, default_collected_at):
    return {
        "doc_id": row.get("doc_id", "").strip(),
        "title": row.get("title", "").strip(),
        "source_type": row.get("source_type", "official_kb").strip(),
        "source_url": row.get("source_url", "").strip(),
        "trust_level": row.get("trust_level", "high").strip(),
        "collected_at": row.get("collected_at", default_collected_at).strip(),
        "language": row.get("language", "en").strip(),
        "error_category": row.get("error_category", "").strip(),
        "raw_text": row.get("raw_text", "").strip(),
    }


def convert_csv_to_json(input_csv, output_json):
    default_collected_at = date.today().isoformat()
    docs = []

    with open(input_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            docs.append(row_to_doc(row, default_collected_at))

    output_dir = os.path.dirname(output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if output_json.lower().endswith(".jsonl"):
        with open(output_json, "w", encoding="utf-8") as f:
            for doc in docs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    else:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(docs, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV to JSON docs format.")
    parser.add_argument("input_csv", help="입력 CSV 파일 경로")
    parser.add_argument("output_json", help="출력 JSON 파일 경로")
    args = parser.parse_args()

    convert_csv_to_json(args.input_csv, args.output_json)
    print(f"완료: {args.output_json}")
