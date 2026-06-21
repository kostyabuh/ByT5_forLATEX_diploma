from pathlib import Path
import warnings
import re

import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from transformers.utils import logging


warnings.filterwarnings("ignore")
logging.set_verbosity_error()

TRANSLATE_MODEL_ID = "facebook/nllb-200-distilled-600M"
LATEX_MODEL_ID = "duanxianpi/IntelliTex"

# сначала easy
VAL_PATH = Path(r"project\data\processed\s2l_ru_equations_easy_val.parquet")
OUT_PATH = Path(r"project\artifacts\nllb_intellitex\eval_easy_100.parquet")

SAMPLE_N = 100
RANDOM_STATE = 42

MAX_TRANSLATION_INPUT_LENGTH = 256
MAX_TRANSLATION_NEW_TOKENS = 256

MAX_LATEX_INPUT_LENGTH = 256
MAX_NEW_TOKENS = 192

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


def normalize_text(text: str) -> str:
    return (text or "").strip()


def normalize_prediction(text: str) -> str:
    text = normalize_text(text)

    if text.startswith("$$") and text.endswith("$$"):
        text = text[2:-2].strip()

    if text.startswith(r"\(") and text.endswith(r"\)"):
        text = text[2:-2].strip()

    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_translation_model():
    tokenizer = AutoTokenizer.from_pretrained(TRANSLATE_MODEL_ID)
    tokenizer.src_lang = "rus_Cyrl"

    model = AutoModelForSeq2SeqLM.from_pretrained(
        TRANSLATE_MODEL_ID,
        torch_dtype=DTYPE,
    ).to(DEVICE)
    model.eval()
    return tokenizer, model


def translate_ru_to_en(text: str, tokenizer, model) -> str:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TRANSLATION_INPUT_LENGTH,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    forced_bos_token_id = tokenizer.convert_tokens_to_ids("eng_Latn")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=MAX_TRANSLATION_NEW_TOKENS,
            num_beams=4,
            early_stopping=True,
        )

    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


def build_latex_model():
    tokenizer = AutoTokenizer.from_pretrained(LATEX_MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(LATEX_MODEL_ID).to(DEVICE)
    model.eval()
    return tokenizer, model


def generate_latex_from_english(text_en: str, tokenizer, model) -> str:
    prompt = f"Convert natural-language math into a STRICT LaTeX equation\n{text_en}"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LATEX_INPUT_LENGTH,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=4,
            early_stopping=True,
            do_sample=False,
        )

    pred = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return normalize_prediction(pred)


def main():
    if not VAL_PATH.exists():
        raise FileNotFoundError(f"Файл не найден: {VAL_PATH}")

    df = pd.read_parquet(VAL_PATH)

    required = {"input_text", "target_text"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"В файле {VAL_PATH} нет нужных колонок {required}")

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
    print(f"translation model: {TRANSLATE_MODEL_ID}")
    print(f"latex model:       {LATEX_MODEL_ID}")
    print(f"rows to evaluate:  {len(df)}")

    tr_tokenizer, tr_model = build_translation_model()
    latex_tokenizer, latex_model = build_latex_model()

    rows = []

    for i, row in enumerate(df.itertuples(index=False), start=1):
        en_text = translate_ru_to_en(row.input_text, tr_tokenizer, tr_model)
        pred = generate_latex_from_english(en_text, latex_tokenizer, latex_model)

        rows.append(
            {
                "input_text": row.input_text,
                "translated_en": en_text,
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