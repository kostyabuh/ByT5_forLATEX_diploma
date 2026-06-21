from pathlib import Path
import json
import math
import re
import warnings

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.utils import logging

from latex_normalize import normalize_latex, normalized_exact_match


warnings.filterwarnings("ignore")
logging.set_verbosity_error()

MODEL_DIR = Path(r"project\artifacts\byt5_base_mathbridge_ru_medium_clean\final_model")
EVAL_PATH = Path(r"project\data\processed\s2l_ru_equations_medium_clean_unused_val_rest.parquet")
OUT_PATH = Path(r"project\artifacts\nbest_rerank\unused_rest_base_nbest_rerank.parquet")

TASK_PREFIX = "spoken math to latex: "

MAX_INPUT_LENGTH = 192
MAX_NEW_TOKENS = 144

NUM_BEAMS = 8
NUM_RETURN_SEQUENCES = 5

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

COMMON_FUNCS = [
    r"\sin", r"\cos", r"\tan", r"\cot", r"\sec", r"\csc",
    r"\sinh", r"\cosh", r"\tanh",
    r"\arcsin", r"\arccos", r"\arctan",
    r"\ln", r"\log", r"\exp", r"\sqrt",
    r"\lim", r"\sum", r"\prod", r"\int",
]

COMMON_COMMANDS = [
    r"\operatorname", r"\mathbb", r"\mathcal", r"\mathrm",
    r"\subset", r"\supset", r"\cap", r"\cup", r"\oplus",
    r"\otimes", r"\odot", r"\frac", r"\sqrt",
    r"\langle", r"\rangle", r"\cdot",
]


def normalize_text(text: str) -> str:
    return (text or "").strip()


def has_unbalanced(s: str) -> bool:
    pairs = [("{", "}"), ("(", ")"), ("[", "]")]
    for left, right in pairs:
        if s.count(left) != s.count(right):
            return True
    return False


def count_occurrences(s: str, items: list[str]) -> int:
    return sum(1 for item in items if item in s)


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
        r"\\sigma_\{\}",
        r"\\mu_\{\}",
        r"\\lambda_\{\}",
    ]

    for pat in bad_patterns:
        if re.search(pat, s):
            score -= 2.0

    if s.endswith(("{", "(", "[", "_", "^", "\\", "=", "+", "-", "*", "/")):
        score -= 2.0

    if "*" in s and r"\cdot" not in s:
        score -= 0.5

    if "\\\\" in s:
        score -= 0.15

    if r"\sum" in s and "_" not in s and "^" not in s:
        score -= 0.5

    if r"\lim" in s and "to" not in s and r"\to" not in s:
        score -= 0.5

    if r"\frac" in s and s.count("{") < 2:
        score -= 1.0

    if r"\sqrt" in s and "{" not in s:
        score -= 1.0

    score += 0.05 * count_occurrences(s, COMMON_FUNCS)
    score += 0.03 * count_occurrences(s, COMMON_COMMANDS)

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


def combined_score(input_text: str, pred: str, seq_score: float | None) -> float:
    score = structural_score(pred) + hint_score(input_text, pred)

    # Мягко учитываем мнение модели, но не даём ему всё решить.
    if seq_score is not None and math.isfinite(seq_score):
        score += 0.20 * float(seq_score)

    return score


def generate_nbest(text: str, tokenizer, model) -> list[dict]:
    full_text = TASK_PREFIX + normalize_text(text)

    inputs = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        result = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            num_return_sequences=NUM_RETURN_SEQUENCES,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=True,
        )

    sequences = result.sequences
    seq_scores = getattr(result, "sequences_scores", None)

    candidates = []
    for i, seq in enumerate(sequences):
        pred = tokenizer.decode(seq, skip_special_tokens=True).strip()
        score = None
        if seq_scores is not None and i < len(seq_scores):
            score = float(seq_scores[i].detach().cpu().item())
        candidates.append(
            {
                "rank": i + 1,
                "prediction": pred,
                "sequence_score": score,
            }
        )

    return candidates


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
    print()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(DEVICE)
    model.eval()

    rows = []

    for i, row in enumerate(df.itertuples(index=False), start=1):
        candidates = generate_nbest(row.input_text, tokenizer, model)

        for cand in candidates:
            cand["heuristic_score"] = combined_score(
                row.input_text,
                cand["prediction"],
                cand["sequence_score"],
            )

        candidates_sorted = sorted(
            candidates,
            key=lambda x: x["heuristic_score"],
            reverse=True,
        )

        top1 = candidates[0]["prediction"]
        reranked = candidates_sorted[0]["prediction"]
        reranked_rank = candidates_sorted[0]["rank"]

        rows.append(
            {
                "input_text": row.input_text,
                "target_text": row.target_text,
                "top1_prediction": top1,
                "top1_raw_exact_match": int(top1 == row.target_text),
                "top1_normalized_exact_match": normalized_exact_match(top1, row.target_text),
                "reranked_prediction": reranked,
                "reranked_candidate_rank": reranked_rank,
                "reranked_raw_exact_match": int(reranked == row.target_text),
                "reranked_normalized_exact_match": normalized_exact_match(reranked, row.target_text),
                "nbest_candidates_json": json.dumps(candidates, ensure_ascii=False),
                "reranked_candidates_json": json.dumps(candidates_sorted, ensure_ascii=False),
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
    print(f"top1 raw exact:        {out['top1_raw_exact_match'].mean():.4f}")
    print(f"top1 normalized exact: {out['top1_normalized_exact_match'].mean():.4f}")
    print()
    print(f"reranked raw exact:        {out['reranked_raw_exact_match'].mean():.4f}")
    print(f"reranked normalized exact: {out['reranked_normalized_exact_match'].mean():.4f}")
    print()

    improved = out[
        (out["top1_normalized_exact_match"] == 0) &
        (out["reranked_normalized_exact_match"] == 1)
    ].copy()

    print(f"rerank improvements over top1: {len(improved)}")
    print()
    if len(improved):
        print(
            improved[
                [
                    "input_text",
                    "target_text",
                    "top1_prediction",
                    "reranked_prediction",
                    "reranked_candidate_rank",
                    "top1_normalized_exact_match",
                    "reranked_normalized_exact_match",
                ]
            ].head(40).to_string()
        )


if __name__ == "__main__":
    main()