from pathlib import Path
import warnings

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.utils import logging


warnings.filterwarnings("ignore")
logging.set_verbosity_error()

MODEL_DIR = Path(r"project\artifacts\byt5_s2l_ru_bridge\final_model")
VAL_PATH = Path(r"project\data\processed\s2l_ru_equations_bridge_val.parquet")
OUT_PATH = Path(r"project\artifacts\byt5_s2l_ru_bridge\eval_preview_100.parquet")

TASK_PREFIX = "spoken math to latex: "

SAMPLE_N = 100
RANDOM_STATE = 42

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

    if not VAL_PATH.exists():
        raise FileNotFoundError(f"Файл не найден: {VAL_PATH}")

    df = pd.read_parquet(VAL_PATH)
    df["input_text"] = df["input_text"].fillna("").astype(str).map(normalize_text)
    df["target_text"] = df["target_text"].fillna("").astype(str).map(normalize_text)

    df = df[
        (df["input_text"] != "") &
        (df["target_text"] != "")
    ][["input_text", "target_text"]].copy().reset_index(drop=True)

    if len(df) == 0:
        raise RuntimeError("Валидационный датасет пустой.")

    if len(df) > SAMPLE_N:
        df = df.sample(n=SAMPLE_N, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"device: {DEVICE}")
    print(f"rows to evaluate: {len(df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(DEVICE)
    model.eval()

    rows = []

    for i, row in enumerate(df.itertuples(index=False), start=1):
        pred = generate_text(row.input_text, tokenizer, model, DEVICE)

        rows.append(
            {
                "input_text": row.input_text,
                "target_text": row.target_text,
                "prediction": pred,
                "exact_match": int(pred == row.target_text),
            }
        )

        print(f"{i}/{len(df)}")

    out = pd.DataFrame(rows)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)

    print(f"saved: {OUT_PATH}")
    print()
    print(out.head(20).to_string())
    print()
    print(f"exact match mean: {out['exact_match'].mean():.4f}")


if __name__ == "__main__":
    main()