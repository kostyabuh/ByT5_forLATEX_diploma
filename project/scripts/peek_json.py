import json
from pathlib import Path

p = Path(r"..\data\raw\unzipped\MathStackExchangeAPI_Part_1_TimeStamps_1512760268_1535031491.json")

with p.open("r", encoding="utf-8") as f:
    s = f.read(200000)

start = s.find("{")
depth = 0
in_str = False
esc = False
end = None

for i, ch in enumerate(s[start:], start):
    if in_str:
        if esc:
            esc = False
        elif ch == "\\":
            esc = True
        elif ch == '"':
            in_str = False
    else:
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

obj = json.loads(s[start:end])

print(list(obj.keys()))