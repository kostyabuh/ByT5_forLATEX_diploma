from __future__ import annotations

from pathlib import Path
import argparse
import gc
import re
import warnings

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.utils import logging

from latex_normalize import normalize_latex


warnings.filterwarnings("ignore")
logging.set_verbosity_error()

BASE_MODEL_DIR = Path(r"project\artifacts\byt5_base_mathbridge_ru_medium_clean\final_model")
PATCHED_MODEL_DIR = Path(r"project\artifacts\byt5_base_mathbridge_ru_medium_clean_plus_auto_patch\final_model")

TASK_PREFIX = "spoken math to latex: "
MAX_INPUT_LENGTH = 192
MAX_NEW_TOKENS = 144

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


GREEK_HINTS = {
    "альфа": [r"\alpha"],
    "бета": [r"\beta"],
    "гамма": [r"\gamma", r"\Gamma"],
    "дельта": [r"\delta", r"\Delta"],
    "эпсилон": [r"\epsilon", r"\varepsilon"],
    "дзета": [r"\zeta"],
    "эта": [r"\eta"],
    "тета": [r"\theta", r"\Theta"],
    "йота": [r"\iota"],
    "каппа": [r"\kappa"],
    "лямбда": [r"\lambda", r"\Lambda"],
    "мю": [r"\mu"],
    "ню": [r"\nu"],
    "кси": [r"\xi", r"\Xi", r"\chi"],
    "хи": [r"\chi", r"\Xi"],
    "пи": [r"\pi", r"\Pi"],
    "ро": [r"\rho"],
    "сигма": [r"\sigma", r"\Sigma"],
    "тау": [r"\tau"],
    "фи": [r"\phi", r"\Phi", r"\varphi"],
    "пси": [r"\psi", r"\Psi"],
    "омега": [r"\omega", r"\Omega"],
}


def normalize_text(text: str) -> str:
    return (text or "").strip()


def has_unbalanced(s: str) -> bool:
    pairs = [("{", "}"), ("(", ")"), ("[", "]")]
    for left, right in pairs:
        if s.count(left) != s.count(right):
            return True
    return False


def structural_score(pred: str) -> float:
    s = normalize_text(pred)
    score = 0.0

    if not s:
        return -100.0

    if has_unbalanced(s):
        score -= 3.0

    bad_patterns = [
        r"_\{\}",
        r"\^\{\}",
        r"\(\)",
        r"\[\]",
        r"\[objectObject\]",
    ]

    for pat in bad_patterns:
        if re.search(pat, s):
            score -= 2.0

    if s.endswith(("{", "(", "[", "_", "^", "\\", "=", "+", "-", "*", "/")):
        score -= 2.0

    if r"\sum" in s and "_" not in s and "^" not in s:
        score -= 0.5

    if r"\lim" in s and "to" not in s and r"\to" not in s:
        score -= 0.5

    if "*" in s and r"\cdot" not in s:
        score -= 0.5

    if r"\sigma_{}" in s or r"\mu_{}" in s:
        score -= 1.5

    norm = normalize_latex(s)
    if not norm:
        score -= 5.0

    return score


def hint_score(input_text: str, pred: str) -> float:
    x = normalize_text(input_text).lower()
    p = normalize_text(pred)

    score = 0.0

    if "логарифм по основанию" in x:
        if r"\log_" in p:
            score += 2.0
        if r"\ln" in p:
            score -= 1.5

    if "натуральн" in x:
        if r"\ln" in p:
            score += 2.0
        if r"\log_" in p:
            score -= 1.0

    if "производ" in x or "d по d" in x or "дэ по дэ" in x or "штрих" in x:
        if "частн" in x:
            if r"\partial" in p:
                score += 2.0
        else:
            if r"\frac{d}" in p or r"\prime" in p:
                score += 2.0
            if r"\partial" in p and r"\frac{d}" not in p and r"\prime" not in p:
                score -= 1.5

    if "тензор" in x:
        if r"\otimes" in p or r"\odot" in p:
            score += 2.0

    if "подмножеств" in x:
        if r"\subset" in p:
            score += 2.0

    if "пересечен" in x:
        if r"\cap" in p:
            score += 2.0

    if "объединен" in x:
        if r"\cup" in p:
            score += 2.0

    if "сумма" in x:
        if r"\sum" in p:
            score += 2.0
        else:
            score -= 1.0

    if "предел" in x:
        if r"\lim" in p:
            score += 2.0
        else:
            score -= 1.0

    if "корень" in x:
        if r"\sqrt" in p:
            score += 2.0
        else:
            score -= 1.0

    if "дробь" in x or "деленн" in x:
        if r"\frac" in p:
            score += 1.5

    if "индекс" in x or "внизу" in x or "под индекс" in x or "нижн" in x:
        if "_" in p:
            score += 1.0

    if "степени" in x or "в квадрате" in x or "в кубе" in x or "возвед" in x:
        if "^" in p:
            score += 1.0

    if "поддержка" in x and r"\operatorname{supp}" in p:
        score += 2.0

    if "след" in x and (r"\operatorname{Tr}" in p or r"\mathrm{Tr}" in p or r"\textrm{Tr}" in p):
        score += 1.5

    if "математическая" in x and (r"\mathcal" in p or r"\mathbb" in p):
        score += 1.0

    if "кавычки" in x or "углов" in x:
        if r"\langle" in p and r"\rangle" in p:
            score += 2.0

    if "е деленное на пи" in x or "е делённое на пи" in x:
        if r"\mathrm\e" in p:
            score += 2.0

    for ru_name, tex_candidates in GREEK_HINTS.items():
        if ru_name in x:
            if any(tex in p for tex in tex_candidates):
                score += 0.4

    if any(k in x for k in ["логарифм", "предел", "сумма", "производ", "тензор", "подмножеств", "пересечен", "корень", "дробь"]):
        if "\\" not in p:
            score -= 2.0

    return score


def score_prediction(input_text: str, pred: str) -> float:
    return structural_score(pred) + hint_score(input_text, pred)


def choose_prediction(input_text: str, base_pred: str, patched_pred: str) -> tuple[str, str, float, float]:
    base_pred = normalize_text(base_pred)
    patched_pred = normalize_text(patched_pred)

    if base_pred == patched_pred:
        s = score_prediction(input_text, base_pred)
        return base_pred, "same", s, s

    base_score = score_prediction(input_text, base_pred)
    patched_score = score_prediction(input_text, patched_pred)

    if patched_score >= base_score + 0.75:
        return patched_pred, "patched", base_score, patched_score

    return base_pred, "base", base_score, patched_score


def _generate_once(model_dir: Path, text: str) -> str:
    if not model_dir.exists():
        raise FileNotFoundError(f"Модель не найдена: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(DEVICE)
    model.eval()

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
            num_beams=4,
            early_stopping=True,
        )

    pred = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pred


def route_text(text: str) -> dict:
    base_pred = _generate_once(BASE_MODEL_DIR, text)
    patched_pred = _generate_once(PATCHED_MODEL_DIR, text)

    chosen, source, base_score, patched_score = choose_prediction(text, base_pred, patched_pred)

    return {
        "input_text": text,
        "base_prediction": base_pred,
        "patched_prediction": patched_pred,
        "base_score": base_score,
        "patched_score": patched_score,
        "chosen_source": source,
        "chosen_prediction": chosen,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="*", help="Русская словесная запись формулы")
    args = parser.parse_args()

    if args.text:
        text = " ".join(args.text).strip()
    else:
        text = input("Введите выражение: ").strip()

    result = route_text(text)

    print(f"device: {DEVICE}")
    print()
    print("INPUT:")
    print(result["input_text"])
    print()
    print("BASE:")
    print(result["base_prediction"])
    print(f"score = {result['base_score']:.3f}")
    print()
    print("PATCHED:")
    print(result["patched_prediction"])
    print(f"score = {result['patched_score']:.3f}")
    print()
    print("CHOSEN:")
    print(result["chosen_prediction"])
    print(f"source = {result['chosen_source']}")


if __name__ == "__main__":
    main()