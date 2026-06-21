from pathlib import Path
import json

import pandas as pd

from latex_normalize import normalized_exact_match


INPUT_PATH = Path(r"project\artifacts\nbest_rerank\unused_rest_base_nbest_rerank.parquet")


def best_match_rank(candidates, target_text: str) -> int | None:
    for cand in candidates:
        pred = cand.get("prediction", "")
        if normalized_exact_match(pred, target_text) == 1:
            return int(cand.get("rank", 0)) or None
    return None


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Файл не найден: {INPUT_PATH}")

    df = pd.read_parquet(INPUT_PATH)

    required = {"input_text", "target_text", "nbest_candidates_json"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"В parquet нет нужных колонок: {required}")

    rows = []

    for row in df.itertuples(index=False):
        candidates = json.loads(row.nbest_candidates_json)
        match_rank = best_match_rank(candidates, row.target_text)

        rows.append(
            {
                "input_text": row.input_text,
                "target_text": row.target_text,
                "oracle_match_rank": match_rank,
                "oracle_at_1": int(match_rank is not None and match_rank <= 1),
                "oracle_at_2": int(match_rank is not None and match_rank <= 2),
                "oracle_at_3": int(match_rank is not None and match_rank <= 3),
                "oracle_at_5": int(match_rank is not None and match_rank <= 5),
            }
        )

    out = pd.DataFrame(rows)

    print(f"rows: {len(out)}")
    print()
    print(f"oracle@1: {out['oracle_at_1'].mean():.4f}")
    print(f"oracle@2: {out['oracle_at_2'].mean():.4f}")
    print(f"oracle@3: {out['oracle_at_3'].mean():.4f}")
    print(f"oracle@5: {out['oracle_at_5'].mean():.4f}")
    print()

    improved_over_top1 = out[
        (out["oracle_at_1"] == 0) &
        (out["oracle_at_5"] == 1)
    ].copy()

    print(f"cases where top1 misses but top5 contains correct answer: {len(improved_over_top1)}")
    print()
    if len(improved_over_top1):
        print(
            improved_over_top1[
                ["input_text", "target_text", "oracle_match_rank"]
            ]
            .head(40)
            .to_string()
        )


if __name__ == "__main__":
    main()