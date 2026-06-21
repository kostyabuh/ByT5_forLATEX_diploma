import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
UNZIPPED_DIR = RAW_DIR / "unzipped"

src = UNZIPPED_DIR / "MathStackExchangeAPI_Part_1_TimeStamps_1512760268_1535031491.json"
dst = DATA_DIR / "part1_masked.parquet"

with src.open("r", encoding="utf-8") as f:
    data = json.load(f)

pattern = re.compile(r"\$\$.*?\$\$|\$.*?\$", re.DOTALL)

rows = []
for row in data:
    body = (row.get("body_markdown") or "").strip()
    if not body:
        continue

    formulas = []

    def repl(match):
        idx = len(formulas)
        formulas.append(match.group(0))
        return f"[[{idx}]]"

    masked_text = pattern.sub(repl, body)

    if formulas:
        rows.append({
            "body_markdown": body,
            "masked_text": masked_text,
            "formulas": formulas
        })

df = pd.DataFrame(rows)
df.to_parquet(dst, index=False)

print(f"saved: {dst}")
print(f"rows: {len(df)}")
print(df[["masked_text", "formulas"]].head(3).to_string())