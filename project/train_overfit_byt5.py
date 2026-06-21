from pathlib import Path
import warnings

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.utils import logging


warnings.filterwarnings("ignore")
logging.set_verbosity_error()

SRC_PATH = Path(r"project\data\processed\s2l_ru_equations_train.parquet")
ARTIFACT_DIR = Path(r"project\artifacts\byt5_s2l_ru_debug")
FINAL_MODEL_DIR = ARTIFACT_DIR / "final_model"

MODEL_NAME = "google/byt5-small"
TASK_PREFIX = "spoken math to latex: "

DEBUG_ROWS = 512

MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 96

PER_DEVICE_BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 2e-4
LOGGING_STEPS = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_df(path: Path, limit: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    df = pd.read_parquet(path)

    required = {"input_text", "target_text"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"В файле {path} нет нужных колонок {required}")

    df["input_text"] = df["input_text"].fillna("").astype(str).str.strip()
    df["target_text"] = df["target_text"].fillna("").astype(str).str.strip()

    df = df[
        (df["input_text"] != "") &
        (df["target_text"] != "")
    ][["input_text", "target_text"]].copy()

    df = df.drop_duplicates(subset=["input_text", "target_text"]).reset_index(drop=True)

    if len(df) == 0:
        raise RuntimeError("После очистки датасет пустой.")

    df = df.head(limit).reset_index(drop=True)
    return df


def make_hf_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df, preserve_index=False)


def tokenize_function_builder(tokenizer):
    def tokenize_function(batch):
        inputs = [TASK_PREFIX + x for x in batch["input_text"]]

        model_inputs = tokenizer(
            inputs,
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
        )

        try:
            labels = tokenizer(
                text_target=batch["target_text"],
                max_length=MAX_TARGET_LENGTH,
                truncation=True,
            )
        except TypeError:
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    batch["target_text"],
                    max_length=MAX_TARGET_LENGTH,
                    truncation=True,
                )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return tokenize_function


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
            max_new_tokens=MAX_TARGET_LENGTH,
            num_beams=1,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def main():
    print(f"device: {DEVICE}")

    df = load_df(SRC_PATH, DEBUG_ROWS)
    print(f"debug rows: {len(df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    ds = make_hf_dataset(df)
    tokenize_function = tokenize_function_builder(tokenizer)

    ds_tok = ds.map(
        tokenize_function,
        batched=True,
        remove_columns=ds.column_names,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(ARTIFACT_DIR),
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=LOGGING_STEPS,
        fp16=False,
        save_steps=10**9,
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_tok,
        data_collator=data_collator,
    )

    trainer.train()

    trainer.save_model(str(FINAL_MODEL_DIR))
    tokenizer.save_pretrained(str(FINAL_MODEL_DIR))

    model = AutoModelForSeq2SeqLM.from_pretrained(FINAL_MODEL_DIR).to(DEVICE)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(FINAL_MODEL_DIR)

    exact = 0
    rows = []

    for row in df.itertuples(index=False):
        pred = generate_text(row.input_text, tokenizer, model, DEVICE)
        ok = int(pred == row.target_text)
        exact += ok

        rows.append(
            {
                "input_text": row.input_text,
                "target_text": row.target_text,
                "prediction": pred,
                "exact_match": ok,
            }
        )

    exact_rate = exact / len(df)
    print()
    print(f"exact match on debug subset: {exact}/{len(df)} = {exact_rate:.4f}")
    print()

    out = pd.DataFrame(rows)
    print(out.head(30).to_string())

    out_path = ARTIFACT_DIR / "debug_predictions.parquet"
    out.to_parquet(out_path, index=False)
    print()
    print(f"saved debug predictions: {out_path}")


if __name__ == "__main__":
    main()