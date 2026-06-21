from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_DIR = Path(r"project\artifacts\distilbart_baseline\final_model")

TEST_TEXTS = [
    "интеграл от 3 до 5 по икс dx",
    "икс в квадрате плюс игрек в квадрате равно единице",
    "сумма по i от 1 до n от x_i",
    "корень квадратный из a плюс b",
    "предел при x стремящемся к нулю sin x делить на x",
]


def generate_text(text: str, tokenizer, model, device: str) -> str:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def main():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Модель не найдена: {MODEL_DIR}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    for i, text in enumerate(TEST_TEXTS, start=1):
        pred = generate_text(text, tokenizer, model, device)

        print("=" * 80)
        print(f"Пример {i}")
        print("INPUT:")
        print(text)
        print()
        print("OUTPUT:")
        print(pred)
        print()


if __name__ == "__main__":
    main()