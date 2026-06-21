from pathlib import Path

import pandas as pd

from latex_normalize import normalize_latex, normalized_exact_match
from postprocess_latex_prediction import postprocess_latex_prediction


INPUT_PATH = Path(r"project\artifacts\holdout_compare\medium_clean_holdout_100_compare.parquet")
OUTPUT_PATH = Path(r"project\artifacts\holdout_compare\medium_clean_holdout_100_compare_postprocessed.parquet")


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Файл не найден: {INPUT_PATH}")

    df = pd.read_parquet(INPUT_PATH)

    required = {"input_text", "target_text", "base_prediction"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"В parquet нет нужных колонок: {required}")

    df["input_text"] = df["input_text"].fillna("").astype(str)
    df["target_text"] = df["target_text"].fillna("").astype(str)
    df["base_prediction"] = df["base_prediction"].fillna("").astype(str)

    df["base_prediction_postprocessed"] = df["base_prediction"].map(postprocess_latex_prediction)
    df["target_norm"] = df["target_text"].map(normalize_latex)
    df["base_prediction_norm"] = df["base_prediction"].map(normalize_latex)
    df["base_prediction_postprocessed_norm"] = df["base_prediction_postprocessed"].map(normalize_latex)

    df["base_post_raw_exact_match"] = (
        df["base_prediction_postprocessed"] == df["target_text"]
    ).astype(int)

    df["base_post_normalized_exact_match"] = [
        normalized_exact_match(pred, tgt)
        for pred, tgt in zip(df["base_prediction_postprocessed"], df["target_text"])
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"saved: {OUTPUT_PATH}")
    print()
    if "base_raw_exact_match" in df.columns:
        print(f"base raw exact:                 {df['base_raw_exact_match'].mean():.4f}")
    if "base_normalized_exact_match" in df.columns:
        print(f"base normalized exact:          {df['base_normalized_exact_match'].mean():.4f}")
    print(f"postprocessed raw exact:        {df['base_post_raw_exact_match'].mean():.4f}")
    print(f"postprocessed normalized exact: {df['base_post_normalized_exact_match'].mean():.4f}")
    print()

    improved = df[
        (df.get("base_normalized_exact_match", pd.Series([0] * len(df))) == 0) &
        (df["base_post_normalized_exact_match"] == 1)
    ].copy()

    print(f"improved rows after postprocess: {len(improved)}")
    print()
    if len(improved):
        print(
            improved[
                [
                    "input_text",
                    "target_text",
                    "base_prediction",
                    "base_prediction_postprocessed",
                ]
            ].head(40).to_string()
        )


if __name__ == "__main__":
    main()