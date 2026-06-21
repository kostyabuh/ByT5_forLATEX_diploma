from pathlib import Path
import warnings

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.utils import logging

from latex_normalize import normalize_latex, normalized_exact_match


warnings.filterwarnings("ignore")
logging.set_verbosity_error()

MODEL_DIR = Path(r"project\artifacts\byt5_base_mathbridge_ru_medium_clean_plus_synth_v1\final_model")
HOLDOUT_PATH = Path(r"project\data\processed\s2l_ru_equations_medium_clean_holdout_100.parquet")
OUT_PATH = Path(r"project\artifacts\byt5_base_mathbridge_ru_medium_clean_plus_synth_v1\holdout_100_eval.parquet")

TASK_PREFIX = "spoken math to latex: "

MAX_INPUT_LENGTH = 192
MAX_NEW_TOKENS = 144

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def normalize_text(text: str) -> str:
    return (text or "").strip()


def generate_text(text: str, tokenizer, model, device: str) -> str:
    text = TASK_PREFIX + text

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=4,
            early_stopping=True,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def main():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Модель не найдена: {MODEL_DIR}")

    if not HOLDOUT_PATH.exists():
        raise FileNotFoundError(f"Файл не найден: {HOLDOUT_PATH}")

    df = pd.read_parquet(HOLDOUT_PATH)

    required = {"input_text", "target_text"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"В holdout нет нужных колонок: {required}")

    df["input_text"] = df["input_text"].fillna("").astype(str).map(normalize_text)
    df["target_text"] = df["target_text"].fillna("").astype(str).map(normalize_text)

    df = df[
        (df["input_text"] != "") &
        (df["target_text"] != "")
    ][["input_text", "target_text"]].copy().reset_index(drop=True)

    print(f"device: {DEVICE}")
    print(f"rows to evaluate: {len(df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(DEVICE)
    model.eval()

    preds = []

    for i, row in enumerate(df.itertuples(index=False), start=1):
        pred = generate_text(row.input_text, tokenizer, model, DEVICE)
        preds.append(pred)
        print(f"{i}/{len(df)}")

    df["prediction"] = preds
    df["prediction_norm"] = df["prediction"].map(normalize_latex)
    df["target_norm"] = df["target_text"].map(normalize_latex)
    df["raw_exact_match"] = (df["prediction"] == df["target_text"]).astype(int)
    df["normalized_exact_match"] = [
        normalized_exact_match(p, t)
        for p, t in zip(df["prediction"], df["target_text"])
    ]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    print(f"saved: {OUT_PATH}")
    print()
    print(f"raw exact mean:        {df['raw_exact_match'].mean():.4f}")
    print(f"normalized exact mean: {df['normalized_exact_match'].mean():.4f}")
    print()
    print(df.head(20).to_string())


if __name__ == "__main__":
    main()