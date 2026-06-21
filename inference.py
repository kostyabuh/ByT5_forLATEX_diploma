from pathlib import Path
import time

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class LatexGenerator:
    def __init__(
        self,
        model_dir: str | Path,
        task_prefix: str = "spoken math to latex: ",
        max_input_length: int = 192,
        max_new_tokens: int = 144,
        num_beams: int = 8,
    ):
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Папка модели не найдена: {self.model_dir}")

        self.task_prefix = task_prefix
        self.max_input_length = max_input_length
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = self.model_dir.name

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_dir).to(self.device)
        self.model.eval()

    def generate(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            raise ValueError("Пустой текст для генерации.")

        full_text = self.task_prefix + text

        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                early_stopping=True,
            )

        latex = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        return latex