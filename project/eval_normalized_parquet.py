from pathlib import Path

import pandas as pd

from latex_normalize import normalize_latex, normalized_exact_match


INPUT_PATH = Path(r"project\artifacts\byt5_base_mathbridge_ru_medium_clean\eval_preview_100.parquet")
OUTPUT_PATH = Path(r"project\artifacts\byt5_base_mathbridge_ru_medium_clean\eval_preview_100_normalized.parquet")


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Файл не найден: {INPUT_PATH}")

    df = pd.read_parquet(INPUT_PATH)

    required = {"input_text", "target_text", "prediction"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"В parquet нет нужных колонок: {required}")

    df["input_text"] = df["input_text"].fillna("").astype(str)
    df["target_text"] = df["target_text"].fillna("").astype(str)
    df["prediction"] = df["prediction"].fillna("").astype(str)

    df["raw_exact_match"] = (df["prediction"] == df["target_text"]).astype(int)
    df["target_norm"] = df["target_text"].map(normalize_latex)
    df["prediction_norm"] = df["prediction"].map(normalize_latex)
    df["normalized_exact_match"] = [
        normalized_exact_match(p, t)
        for p, t in zip(df["prediction"], df["target_text"])
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"saved: {OUTPUT_PATH}")
    print()
    print(f"raw exact mean:        {df['raw_exact_match'].mean():.4f}")
    print(f"normalized exact mean: {df['normalized_exact_match'].mean():.4f}")
    print()

    changed = df[df["raw_exact_match"] != df["normalized_exact_match"]].copy()
    print(f"rows changed by normalization: {len(changed)}")
    print()

    if len(changed):
        print(
            changed[
                [
                    "input_text",
                    "target_text",
                    "prediction",
                    "target_norm",
                    "prediction_norm",
                    "raw_exact_match",
                    "normalized_exact_match",
                ]
            ]
            .head(30)
            .to_string()
        )


if __name__ == "__main__":
    main()