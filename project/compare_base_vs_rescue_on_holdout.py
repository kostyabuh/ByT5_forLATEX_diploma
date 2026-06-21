from pathlib import Path
import warnings

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.utils import logging

from latex_normalize import normalize_latex, normalized_exact_match


warnings.filterwarnings("ignore")
logging.set_verbosity_error()

HOLDOUT_PATH = Path(r"project\data\processed\s2l_ru_equations_medium_clean_holdout_100.parquet")

BASE_MODEL_DIR = Path(r"project\artifacts\byt5_base_mathbridge_ru_medium_clean\final_model")
RESCUE_MODEL_DIR = Path(r"project\artifacts\byt5_base_mathbridge_ru_medium_clean_rescue\final_model")

OUT_COMPARE_PATH = Path(r"project\artifacts\holdout_compare\medium_clean_holdout_100_compare.parquet")

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


def run_model(model_dir: Path, df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if not model_dir.exists():
        raise FileNotFoundError(f"Модель не найдена: {model_dir}")

    print(f"loading model: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(DEVICE)
    model.eval()

    preds = []
    raw = []
    norm = []

    for i, row in enumerate(df.itertuples(index=False), start=1):
        pred = generate_text(row.input_text, tokenizer, model, DEVICE)
        preds.append(pred)
        raw.append(int(pred == row.target_text))
        norm.append(normalized_exact_match(pred, row.target_text))
        print(f"{prefix}: {i}/{len(df)}")

    out = pd.DataFrame(
        {
            f"{prefix}_prediction": preds,
            f"{prefix}_raw_exact_match": raw,
            f"{prefix}_normalized_exact_match": norm,
            f"{prefix}_prediction_norm": [normalize_latex(x) for x in preds],
        }
    )

    return out


def main():
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
    print(f"holdout rows: {len(df)}")
    print()

    base_part = run_model(BASE_MODEL_DIR, df, "base")
    rescue_part = run_model(RESCUE_MODEL_DIR, df, "rescue")

    result = pd.concat([df, base_part, rescue_part], axis=1)
    result["target_norm"] = result["target_text"].map(normalize_latex)

    OUT_COMPARE_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUT_COMPARE_PATH, index=False)

    print()
    print(f"saved: {OUT_COMPARE_PATH}")
    print()
    print(f"base raw exact:        {result['base_raw_exact_match'].mean():.4f}")
    print(f"base normalized exact: {result['base_normalized_exact_match'].mean():.4f}")
    print()
    print(f"rescue raw exact:        {result['rescue_raw_exact_match'].mean():.4f}")
    print(f"rescue normalized exact: {result['rescue_normalized_exact_match'].mean():.4f}")
    print()
    print("base -> rescue improvements:")
    improved = result[
        (result["base_normalized_exact_match"] == 0) &
        (result["rescue_normalized_exact_match"] == 1)
    ]
    print(f"improved rows: {len(improved)}")
    print()
    if len(improved):
        print(
            improved[
                [
                    "input_text",
                    "target_text",
                    "base_prediction",
                    "rescue_prediction",
                    "base_normalized_exact_match",
                    "rescue_normalized_exact_match",
                ]
            ]
            .head(30)
            .to_string()
        )


if __name__ == "__main__":
    main()