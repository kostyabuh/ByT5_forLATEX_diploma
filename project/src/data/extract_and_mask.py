import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
UNZIPPED_DIR = RAW_DIR / "unzipped"
INTERIM_DIR = DATA_DIR / "interim"

INPUT_FILENAME = "MathStackExchangeAPI_Part_1_TimeStamps_1512760268_1535031491.json"
OUTPUT_FILENAME = "part1_masked.parquet"

FORMULA_PATTERN = re.compile(r"\$\$.*?\$\$|\$.*?\$", re.DOTALL)


def extract_and_mask_formulas(text: str) -> tuple[str, list[str]]:
    text = (text or "").strip()
    formulas: list[str] = []

    def replacer(match: re.Match) -> str:
        idx = len(formulas)
        formulas.append(match.group(0))
        return f"[[{idx}]]"

    masked_text = FORMULA_PATTERN.sub(replacer, text)
    return masked_text, formulas


def build_rows(data: list[dict]) -> list[dict]:
    rows: list[dict] = []

    for row in data:
        body_markdown = (row.get("body_markdown") or "").strip()
        if not body_markdown:
            continue

        masked_text, formulas = extract_and_mask_formulas(body_markdown)
        if not formulas:
            continue

        rows.append(
            {
                "body_markdown": body_markdown,
                "masked_text": masked_text,
                "formulas": formulas,
            }
        )

    return rows


def main() -> None:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    src = UNZIPPED_DIR / INPUT_FILENAME
    dst = INTERIM_DIR / OUTPUT_FILENAME

    if not src.exists():
        raise FileNotFoundError(f"Файл не найден: {src}")

    with src.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows = build_rows(data)
    df = pd.DataFrame(rows)
    df.to_parquet(dst, index=False)

    print(f"saved: {dst}")
    print(f"rows: {len(df)}")
    print(df.head(3).to_string())


if __name__ == "__main__":
    main()