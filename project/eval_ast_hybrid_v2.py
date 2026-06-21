from pathlib import Path
import sys
import warnings

import pandas as pd
from transformers.utils import logging

from latex_normalize import normalized_exact_match


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spoken_math_ast_parser_v2 import ast_parse_v2


warnings.filterwarnings("ignore")
logging.set_verbosity_error()

INPUT_PATH = Path(r"project\artifacts\nbest_rerank\unused_rest_base_nbest_rerank.parquet")
OUT_PATH = Path(r"project\artifacts\ast_hybrid_v2\unused_rest_eval.parquet")

WHITELIST = {
    "fraction",
    "log_base",
    "natlog",
    "sqrt",
    "trig",
    "derivative",
    "limit",
    "sum",
    "relation",
    "power_index",
    "call",
}

CONFIDENCE_THRESHOLD = 0.95


def normalize_text(text: str) -> str:
    return (text or "").strip()


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Файл не найден: {INPUT_PATH}")

    src = pd.read_parquet(INPUT_PATH)

    required = {"input_text", "target_text", "top1_prediction"}
    if not required.issubset(src.columns):
        raise RuntimeError(f"В parquet нет нужных колонок: {required}")

    df = src[["input_text", "target_text", "top1_prediction"]].copy()
    df = df.rename(columns={"top1_prediction": "base_prediction"})

    df["input_text"] = df["input_text"].fillna("").astype(str).map(normalize_text)
    df["target_text"] = df["target_text"].fillna("").astype(str).map(normalize_text)
    df["base_prediction"] = df["base_prediction"].fillna("").astype(str).map(normalize_text)

    rows = []

    for i, row in enumerate(df.itertuples(index=False), start=1):
        ast_pred, ast_family, ast_conf = ast_parse_v2(row.input_text)

        use_ast = (
            ast_pred is not None and
            ast_family in WHITELIST and
            ast_conf >= CONFIDENCE_THRESHOLD
        )

        if use_ast:
            pred = ast_pred
            source = f"ast:{ast_family}"
        else:
            pred = row.base_prediction
            source = "base"

        rows.append(
            {
                "input_text": row.input_text,
                "target_text": row.target_text,
                "base_prediction": row.base_prediction,
                "ast_prediction": ast_pred,
                "ast_family": ast_family,
                "ast_confidence": ast_conf,
                "hybrid_prediction": pred,
                "hybrid_source": source,
                "hybrid_raw_exact_match": int(pred == row.target_text),
                "hybrid_normalized_exact_match": normalized_exact_match(pred, row.target_text),
            }
        )

        if i % 100 == 0 or i == len(df):
            print(f"{i}/{len(df)}")

    out = pd.DataFrame(rows)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)

    base_norm = pd.Series(
        [normalized_exact_match(p, t) for p, t in zip(out["base_prediction"], out["target_text"])]
    ).mean()

    print()
    print(f"saved: {OUT_PATH}")
    print()
    print(f"base raw exact:        {(out['base_prediction'] == out['target_text']).mean():.4f}")
    print(f"base normalized exact: {base_norm:.4f}")
    print()
    print(f"hybrid raw exact:        {out['hybrid_raw_exact_match'].mean():.4f}")
    print(f"hybrid normalized exact: {out['hybrid_normalized_exact_match'].mean():.4f}")
    print()
    print("source counts:")
    print(out["hybrid_source"].value_counts().to_string())
    print()
    print("ast family counts:")
    ast_hits = out[out["ast_prediction"].notna()]["ast_family"].value_counts()
    if len(ast_hits):
        print(ast_hits.to_string())
    else:
        print("no ast hits")
    print()

    improved = out[
        (out["hybrid_source"] != "base") &
        (out["hybrid_normalized_exact_match"] == 1)
    ].copy()

    print(f"successful ast-based hits: {len(improved)}")
    print()
    if len(improved):
        print(
            improved[
                [
                    "input_text",
                    "target_text",
                    "ast_prediction",
                    "ast_family",
                    "ast_confidence",
                ]
            ]
            .head(60)
            .to_string()
        )


if __name__ == "__main__":
    main()