import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
UNZIPPED_DIR = RAW_DIR / "unzipped"

src = UNZIPPED_DIR / "MathStackExchangeAPI_Part_1_TimeStamps_1512760268_1535031491.json"
dst = DATA_DIR / "part1_formulas.parquet"

with src.open("r", encoding="utf-8") as f:
    data = json.load(f)

pattern = re.compile(r"\$\$.*?\$\$|\$.*?\$", re.DOTALL)

rows = []
for row in data:
    body = (row.get("body_markdown") or "").strip()
    if not body:
        continue

    formulas = pattern.findall(body)
    if formulas:
        rows.append({
            "body_markdown": body,
            "formulas": formulas
        })

df = pd.DataFrame(rows)
df.to_parquet(dst, index=False)

print(f"saved: {dst}")
print(f"rows: {len(df)}")
print(df.head(3).to_string())