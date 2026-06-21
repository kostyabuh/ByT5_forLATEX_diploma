from pathlib import Path
import sys
import warnings

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.utils import logging

from latex_normalize import normalize_latex, normalized_exact_match


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rule_math_parser import rule_parse


warnings.filterwarnings("ignore")
logging.set_verbosity_error()

MODEL_DIR = Path(r"project\artifacts\byt5_base_mathbridge_ru_medium_clean\final_model")
EVAL_PATH = Path(r"project\data\processed\s2l_ru_equations_medium_clean_unused_val_rest.parquet")
OUT_PATH = Path(r"project\artifacts\hybrid_rules_plus_base\unused_rest_eval.parquet")

TASK_PREFIX = "spoken math to latex: "

MAX_INPUT_LENGTH = 192
MAX_NEW_TOKENS = 144
NUM_BEAMS = 8

RULE_CONFIDENCE_THRESHOLD = 0.92

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def normalize_text(text: str) -> str:
    return (text or "").strip()


def generate_base(text: str, tokenizer, model) -> str:
    full_text = TASK_PREFIX + normalize_text(text)

    inputs = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            early_stopping=True,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def main():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Модель не найдена: {MODEL_DIR}")
    if not EVAL_PATH.exists():
        raise FileNotFoundError(f"Файл не найден: {EVAL_PATH}")

    df = pd.read_parquet(EVAL_PATH)

    required = {"input_text", "target_text"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"В eval parquet нет нужных колонок: {required}")

    df["input_text"] = df["input_text"].fillna("").astype(str).map(normalize_text)
    df["target_text"] = df["target_text"].fillna("").astype(str).map(normalize_text)

    df = df[
        (df["input_text"] != "") &
        (df["target_text"] != "")
    ][["input_text", "target_text"]].copy().reset_index(drop=True)

    print(f"device: {DEVICE}")
    print(f"rows to evaluate: {len(df)}")
    print(f"rule threshold: {RULE_CONFIDENCE_THRESHOLD}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(DEVICE)
    model.eval()

    rows = []

    for i, row in enumerate(df.itertuples(index=False), start=1):
        rule_pred, rule_name, rule_conf = rule_parse(row.input_text)

        if rule_pred is not None and rule_conf >= RULE_CONFIDENCE_THRESHOLD:
            hybrid_pred = rule_pred
            source = f"rule:{rule_name}"
            base_pred = None
        else:
            base_pred = generate_base(row.input_text, tokenizer, model)
            hybrid_pred = base_pred
            source = "base"

        rows.append(
            {
                "input_text": row.input_text,
                "target_text": row.target_text,
                "rule_prediction": rule_pred,
                "rule_name": rule_name,
                "rule_confidence": rule_conf,
                "base_prediction": base_pred,
                "hybrid_prediction": hybrid_pred,
                "hybrid_source": source,
                "hybrid_raw_exact_match": int(hybrid_pred == row.target_text),
                "hybrid_normalized_exact_match": normalized_exact_match(hybrid_pred, row.target_text),
            }
        )

        if i % 100 == 0 or i == len(df):
            print(f"{i}/{len(df)}")

    out = pd.DataFrame(rows)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)

    print()
    print(f"saved: {OUT_PATH}")
    print()
    print(f"hybrid raw exact:        {out['hybrid_raw_exact_match'].mean():.4f}")
    print(f"hybrid normalized exact: {out['hybrid_normalized_exact_match'].mean():.4f}")
    print()
    print("source counts:")
    print(out["hybrid_source"].value_counts().to_string())
    print()
    improved = out[
        (out["hybrid_source"] != "base") &
        (out["hybrid_normalized_exact_match"] == 1)
    ]
    print(f"successful rule-based hits: {len(improved)}")
    print()
    if len(improved):
        print(
            improved[
                [
                    "input_text",
                    "target_text",
                    "rule_prediction",
                    "rule_name",
                    "rule_confidence",
                ]
            ]
            .head(40)
            .to_string()
        )


if __name__ == "__main__":
    main()
    