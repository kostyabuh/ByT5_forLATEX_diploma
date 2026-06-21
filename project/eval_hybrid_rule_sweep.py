from pathlib import Path
import json
import sys
import warnings

import pandas as pd
from transformers.utils import logging

from latex_normalize import normalized_exact_match


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rule_math_parser import rule_parse


warnings.filterwarnings("ignore")
logging.set_verbosity_error()

# Берём уже посчитанный base top1 из n-best eval,
# чтобы не гонять модель заново.
INPUT_PATH = Path(r"project\artifacts\nbest_rerank\unused_rest_base_nbest_rerank.parquet")

SUMMARY_OUT_PATH = Path(r"project\artifacts\hybrid_rule_sweep\sweep_summary.parquet")
BEST_OUT_PATH = Path(r"project\artifacts\hybrid_rule_sweep\best_config_eval.parquet")


def normalize_text(text: str) -> str:
    return (text or "").strip().lower().replace("ё", "е")


def infer_rule_family(input_text: str, rule_prediction: str | None, rule_name: str | None) -> str:
    x = normalize_text(input_text)
    rp = (rule_prediction or "").strip()

    if not rp:
        return "none"

    if "натуральный логарифм" in x:
        return "natlog"

    if "логарифм" in x and "по основанию" in x:
        return "log_base"

    if "корень" in x:
        return "sqrt"

    if "дробь" in x or "деленн" in x:
        # здесь же часто сидят trig(fraction)-шаблоны, но для whitelist это ок
        return "fraction"

    if any(x.startswith(fn + " ") for fn in [
        "синус", "косинус", "тангенс", "котангенс", "секанс", "косеканс",
        "арксинус", "арккосинус", "арктангенс"
    ]):
        return "trig"

    if "производная" in x or "частная производная" in x:
        return "derivative"

    if "предел" in x:
        return "limit"

    if "сумма" in x:
        return "sum"

    if "подмножество" in x or "строгое подмножество" in x:
        return "subset"

    if any(k in x for k in ["в квадрате", "в кубе", "в степени", "с индексом", "нижний индекс", "внизу"]):
        return "power_index"

    if any(k in x for k in ["равно", "больше", "меньше", "равняется", "это равно"]):
        return "relation"

    if rule_name:
        return rule_name

    return "other"


def evaluate_config(
    df: pd.DataFrame,
    config_name: str,
    whitelist: set[str],
    threshold: float,
) -> tuple[pd.DataFrame, dict]:
    rows = []

    for row in df.itertuples(index=False):
        use_rule = (
            row.rule_prediction is not None and
            row.rule_prediction != "" and
            row.rule_confidence >= threshold and
            row.rule_family in whitelist
        )

        if use_rule:
            pred = row.rule_prediction
            source = f"rule:{row.rule_family}"
        else:
            pred = row.base_prediction
            source = "base"

        rows.append(
            {
                "input_text": row.input_text,
                "target_text": row.target_text,
                "base_prediction": row.base_prediction,
                "rule_prediction": row.rule_prediction,
                "rule_name": row.rule_name,
                "rule_confidence": row.rule_confidence,
                "rule_family": row.rule_family,
                "config_name": config_name,
                "hybrid_prediction": pred,
                "hybrid_source": source,
                "hybrid_raw_exact_match": int(pred == row.target_text),
                "hybrid_normalized_exact_match": normalized_exact_match(pred, row.target_text),
            }
        )

    out = pd.DataFrame(rows)

    summary = {
        "config_name": config_name,
        "threshold": threshold,
        "whitelist_json": json.dumps(sorted(list(whitelist)), ensure_ascii=False),
        "hybrid_raw_exact": out["hybrid_raw_exact_match"].mean(),
        "hybrid_normalized_exact": out["hybrid_normalized_exact_match"].mean(),
        "rules_used": int((out["hybrid_source"] != "base").sum()),
        "rule_hits_correct": int(
            ((out["hybrid_source"] != "base") & (out["hybrid_normalized_exact_match"] == 1)).sum()
        ),
    }

    return out, summary


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Файл не найден: {INPUT_PATH}")

    src = pd.read_parquet(INPUT_PATH)

    required = {"input_text", "target_text", "top1_prediction"}
    if not required.issubset(src.columns):
        raise RuntimeError(f"В parquet нет нужных колонок: {required}")

    work = src[["input_text", "target_text", "top1_prediction"]].copy()
    work = work.rename(columns={"top1_prediction": "base_prediction"})

    work["input_text"] = work["input_text"].fillna("").astype(str).str.strip()
    work["target_text"] = work["target_text"].fillna("").astype(str).str.strip()
    work["base_prediction"] = work["base_prediction"].fillna("").astype(str).str.strip()

    rule_predictions = []
    rule_names = []
    rule_confidences = []
    rule_families = []

    for row in work.itertuples(index=False):
        pred, name, conf = rule_parse(row.input_text)
        family = infer_rule_family(row.input_text, pred, name)

        rule_predictions.append(pred)
        rule_names.append(name)
        rule_confidences.append(conf)
        rule_families.append(family)

    work["rule_prediction"] = rule_predictions
    work["rule_name"] = rule_names
    work["rule_confidence"] = rule_confidences
    work["rule_family"] = rule_families

    print(f"rows: {len(work)}")
    print()
    print(f"base raw exact:        {(work['base_prediction'] == work['target_text']).mean():.4f}")
    print(
        f"base normalized exact: "
        f"{pd.Series([normalized_exact_match(p, t) for p, t in zip(work['base_prediction'], work['target_text'])]).mean():.4f}"
    )
    print()

    print("detected rule families:")
    detected = work[work["rule_prediction"].notna()]["rule_family"].value_counts()
    if len(detected):
        print(detected.to_string())
    else:
        print("no rule hits")
    print()

    configs = [
        ("base_only", set(), 999.0),

        ("fraction_097", {"fraction"}, 0.97),
        ("fraction_log_097", {"fraction", "log_base", "natlog"}, 0.97),
        ("fraction_log_sqrt_097", {"fraction", "log_base", "natlog", "sqrt"}, 0.97),
        ("fraction_log_sqrt_trig_097", {"fraction", "log_base", "natlog", "sqrt", "trig"}, 0.97),

        ("fraction_log_sqrt_trig_095", {"fraction", "log_base", "natlog", "sqrt", "trig"}, 0.95),
        ("core_rules_095", {"fraction", "log_base", "natlog", "sqrt", "trig", "subset"}, 0.95),
        ("core_plus_derivative_095", {"fraction", "log_base", "natlog", "sqrt", "trig", "subset", "derivative"}, 0.95),

        ("core_rules_090", {"fraction", "log_base", "natlog", "sqrt", "trig", "subset"}, 0.90),
        ("all_safeish_090", {"fraction", "log_base", "natlog", "sqrt", "trig", "subset", "derivative", "limit", "sum"}, 0.90),
        ("everything_088", {"fraction", "log_base", "natlog", "sqrt", "trig", "subset", "derivative", "limit", "sum", "power_index", "relation"}, 0.88),
    ]

    summaries = []
    best_df = None
    best_norm = -1.0
    best_name = None

    for config_name, whitelist, threshold in configs:
        out_df, summary = evaluate_config(
            work,
            config_name=config_name,
            whitelist=whitelist,
            threshold=threshold,
        )
        summaries.append(summary)

        print(
            f"{config_name:28s} | "
            f"norm={summary['hybrid_normalized_exact']:.4f} | "
            f"raw={summary['hybrid_raw_exact']:.4f} | "
            f"rules_used={summary['rules_used']:3d} | "
            f"rule_hits_correct={summary['rule_hits_correct']:3d}"
        )

        if summary["hybrid_normalized_exact"] > best_norm:
            best_norm = summary["hybrid_normalized_exact"]
            best_df = out_df
            best_name = config_name

    summary_df = pd.DataFrame(summaries).sort_values(
        ["hybrid_normalized_exact", "hybrid_raw_exact"],
        ascending=False,
    ).reset_index(drop=True)

    SUMMARY_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_parquet(SUMMARY_OUT_PATH, index=False)

    if best_df is not None:
        best_df.to_parquet(BEST_OUT_PATH, index=False)

    print()
    print(f"saved summary: {SUMMARY_OUT_PATH}")
    print(f"saved best:    {BEST_OUT_PATH}")
    print()
    print("top configs:")
    print(summary_df.head(20).to_string(index=False))
    print()
    print(f"best config: {best_name} | best normalized exact = {best_norm:.4f}")


if __name__ == "__main__":
    main()