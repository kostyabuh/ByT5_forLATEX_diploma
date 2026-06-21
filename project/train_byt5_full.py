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

TRAIN_PATH = Path(r"project\data\processed\s2l_ru_equations_train.parquet")
VAL_PATH = Path(r"project\data\processed\s2l_ru_equations_val.parquet")

MODEL_NAME = "google/byt5-small"
ARTIFACT_DIR = Path(r"project\artifacts\byt5_s2l_ru")
FINAL_MODEL_DIR = ARTIFACT_DIR / "final_model"

TASK_PREFIX = "spoken math to latex: "

MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 96

PER_DEVICE_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 1
NUM_EPOCHS = 12
LEARNING_RATE = 2e-4
LOGGING_STEPS = 50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_df(path: Path) -> pd.DataFrame:
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
        raise RuntimeError(f"После очистки файл пустой: {path}")

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


def main():
    print(f"device: {DEVICE}")

    train_df = load_df(TRAIN_PATH)
    val_df = load_df(VAL_PATH)

    print(f"train rows: {len(train_df)}")
    print(f"val rows:   {len(val_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    train_ds = make_hf_dataset(train_df)
    val_ds = make_hf_dataset(val_df)

    tokenize_function = tokenize_function_builder(tokenizer)

    train_ds = train_ds.map(
        tokenize_function,
        batched=True,
        remove_columns=train_ds.column_names,
    )

    val_ds = val_ds.map(
        tokenize_function,
        batched=True,
        remove_columns=val_ds.column_names,
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
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=LOGGING_STEPS,
        predict_with_generate=True,
        fp16=False,
        save_steps=10**9,
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    trainer.train()

    trainer.save_model(str(FINAL_MODEL_DIR))
    tokenizer.save_pretrained(str(FINAL_MODEL_DIR))

    print(f"saved final model: {FINAL_MODEL_DIR}")


if __name__ == "__main__":
    main()