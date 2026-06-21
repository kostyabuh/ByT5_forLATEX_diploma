from pathlib import Path
import warnings

import pandas as pd
from transformers.utils import logging

from latex_normalize import normalize_latex, normalized_exact_match
from route_base_vs_patched import choose_prediction


warnings.filterwarnings("ignore")
logging.set_verbosity_error()

INPUT_PATH = Path(r"project\artifacts\holdout_compare\unused_rest_base_vs_auto_patch.parquet")
OUTPUT_PATH = Path(r"project\artifacts\holdout_compare\unused_rest_base_vs_auto_patch_routed.parquet")


def normalize_text(text: str) -> str:
    return (text or "").strip()


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Файл не найден: {INPUT_PATH}")

    df = pd.read_parquet(INPUT_PATH)

    required = {"input_text", "target_text", "base_prediction", "patched_prediction"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"В parquet нет нужных колонок: {required}")

    df["input_text"] = df["input_text"].fillna("").astype(str).map(normalize_text)
    df["target_text"] = df["target_text"].fillna("").astype(str).map(normalize_text)
    df["base_prediction"] = df["base_prediction"].fillna("").astype(str).map(normalize_text)
    df["patched_prediction"] = df["patched_prediction"].fillna("").astype(str).map(normalize_text)

    routed_predictions = []
    routed_source = []
    routed_base_score = []
    routed_patched_score = []

    for i, row in enumerate(df.itertuples(index=False), start=1):
        chosen, source, base_score, patched_score = choose_prediction(
            row.input_text,
            row.base_prediction,
            row.patched_prediction,
        )
        routed_predictions.append(chosen)
        routed_source.append(source)
        routed_base_score.append(base_score)
        routed_patched_score.append(patched_score)

        if i % 100 == 0 or i == len(df):
            print(f"{i}/{len(df)}")

    df["routed_prediction"] = routed_predictions
    df["routed_source"] = routed_source
    df["routed_base_score"] = routed_base_score
    df["routed_patched_score"] = routed_patched_score

    df["target_norm"] = df["target_text"].map(normalize_latex)
    df["routed_prediction_norm"] = df["routed_prediction"].map(normalize_latex)
    df["routed_raw_exact_match"] = (df["routed_prediction"] == df["target_text"]).astype(int)
    df["routed_normalized_exact_match"] = [
        normalized_exact_match(pred, tgt)
        for pred, tgt in zip(df["routed_prediction"], df["target_text"])
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print()
    print(f"saved: {OUTPUT_PATH}")
    print()

    if "base_raw_exact_match" in df.columns:
        print(f"base raw exact:        {df['base_raw_exact_match'].mean():.4f}")
    if "base_normalized_exact_match" in df.columns:
        print(f"base normalized exact: {df['base_normalized_exact_match'].mean():.4f}")
    print()
    if "patched_raw_exact_match" in df.columns:
        print(f"patched raw exact:        {df['patched_raw_exact_match'].mean():.4f}")
    if "patched_normalized_exact_match" in df.columns:
        print(f"patched normalized exact: {df['patched_normalized_exact_match'].mean():.4f}")
    print()
    print(f"routed raw exact:        {df['routed_raw_exact_match'].mean():.4f}")
    print(f"routed normalized exact: {df['routed_normalized_exact_match'].mean():.4f}")
    print()

    print("chosen source counts:")
    print(df["routed_source"].value_counts().to_string())
    print()

    improved_over_base = df[
        (df["base_normalized_exact_match"] == 0) &
        (df["routed_normalized_exact_match"] == 1)
    ].copy()

    print(f"routed improvements over base: {len(improved_over_base)}")
    print()
    if len(improved_over_base):
        print(
            improved_over_base[
                [
                    "input_text",
                    "target_text",
                    "base_prediction",
                    "patched_prediction",
                    "routed_prediction",
                    "routed_source",
                    "base_normalized_exact_match",
                    "patched_normalized_exact_match",
                    "routed_normalized_exact_match",
                ]
            ].head(40).to_string()
        )


if __name__ == "__main__":
    main()
    