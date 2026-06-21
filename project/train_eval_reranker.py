from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


INPUT_PATH = Path(r"project\data\processed\reranker_candidate_dataset.parquet")
MODEL_PATH = Path(r"project\artifacts\learned_reranker\torch_logreg_reranker.pkl")
OUT_PATH = Path(r"project\artifacts\learned_reranker\test_sample_level_eval.parquet")


NON_FEATURE_COLS = {
    "sample_id",
    "input_text",
    "target_text",
    "prediction",
    "label",
    "split",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_EPOCHS = 80
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
RANDOM_STATE = 42


class TorchLogReg(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_best_by_score(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    idx = df.groupby("sample_id")[score_col].idxmax()
    out = df.loc[idx].copy().sort_values("sample_id").reset_index(drop=True)
    return out


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)

    pos = y_true.sum()
    neg = len(y_true) - pos
    if pos == 0 or neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)

    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


def standardize_fit(x: np.ndarray):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (x - mean) / std


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
):
    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    x_val_t = torch.tensor(x_val, dtype=torch.float32).to(DEVICE)

    ds = TensorDataset(x_train_t, y_train_t)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    model = TorchLogReg(x_train.shape[1]).to(DEVICE)

    pos = float(y_train.sum())
    neg = float(len(y_train) - y_train.sum())
    pos_weight_value = neg / max(pos, 1.0)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_state = None
    best_val_auc = -1.0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for xb, yb in dl:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.detach().cpu().item()) * len(xb)

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val_t).detach().cpu().numpy()

        val_auc = safe_auc(y_val, val_logits)

        if np.isfinite(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {
                "model_state_dict": model.state_dict(),
            }

        if epoch % 10 == 0 or epoch == 1 or epoch == NUM_EPOCHS:
            avg_loss = running_loss / len(x_train)
            print(f"epoch {epoch:03d} | train_loss={avg_loss:.4f} | val_auc={val_auc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state["model_state_dict"])

    return model


def predict_scores(model: nn.Module, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(x, dtype=torch.float32).to(DEVICE)
        logits = model(xt).detach().cpu().numpy()
    return logits.astype(np.float64)


def main():
    set_seed(RANDOM_STATE)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Файл не найден: {INPUT_PATH}")

    df = pd.read_parquet(INPUT_PATH)

    required = {"sample_id", "input_text", "target_text", "prediction", "label", "split"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"В parquet нет нужных колонок: {required}")

    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)

    if len(train_df) == 0 or len(test_df) == 0:
        raise RuntimeError("Пустой train или test split.")

    feature_cols = [
        c for c in df.columns
        if c not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]

    if len(feature_cols) == 0:
        raise RuntimeError("Не найдены числовые фичи для reranker.")

    x_train_raw = train_df[feature_cols].astype(float).to_numpy()
    y_train = train_df["label"].astype(int).to_numpy()

    x_test_raw = test_df[feature_cols].astype(float).to_numpy()
    y_test = test_df["label"].astype(int).to_numpy()

    mean, std = standardize_fit(x_train_raw)
    x_train = standardize_apply(x_train_raw, mean, std)
    x_test = standardize_apply(x_test_raw, mean, std)

    print(f"device: {DEVICE}")
    print(f"train candidate rows: {len(train_df)}")
    print(f"test candidate rows:  {len(test_df)}")
    print(f"num features:         {len(feature_cols)}")
    print()

    model = train_model(x_train, y_train, x_test, y_test)

    train_df["reranker_score"] = predict_scores(model, x_train)
    test_df["reranker_score"] = predict_scores(model, x_test)

    train_auc = safe_auc(y_train, train_df["reranker_score"].to_numpy())
    test_auc = safe_auc(y_test, test_df["reranker_score"].to_numpy())

    baseline_test = test_df[test_df["beam_rank"] == 1].copy().sort_values("sample_id").reset_index(drop=True)
    reranked_test = pick_best_by_score(test_df, "reranker_score")
    oracle_test = test_df.groupby("sample_id", as_index=False)["label"].max()

    compare = baseline_test[["sample_id", "input_text", "target_text", "prediction", "label"]].copy()
    compare = compare.rename(
        columns={
            "prediction": "top1_prediction",
            "label": "top1_normalized_exact_match",
        }
    )

    reranked_part = reranked_test[
        ["sample_id", "prediction", "label", "beam_rank", "reranker_score", "combined_score", "sequence_score"]
    ].copy()
    reranked_part = reranked_part.rename(
        columns={
            "prediction": "reranked_prediction",
            "label": "reranked_normalized_exact_match",
            "beam_rank": "reranked_candidate_rank",
        }
    )

    compare = compare.merge(reranked_part, on="sample_id", how="left")
    compare = compare.merge(oracle_test.rename(columns={"label": "oracle_at_5"}), on="sample_id", how="left")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    compare.to_parquet(OUT_PATH, index=False)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(
            {
                "model_state_dict": model.state_dict(),
                "feature_cols": feature_cols,
                "feature_mean": mean,
                "feature_std": std,
            },
            f,
        )

    coef = model.linear.weight.detach().cpu().numpy()[0]
    coef_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "coef": coef,
            "abs_coef": np.abs(coef),
        }
    ).sort_values("abs_coef", ascending=False).reset_index(drop=True)

    print()
    print(f"saved model: {MODEL_PATH}")
    print(f"saved eval:  {OUT_PATH}")
    print()
    print(f"train candidate AUC: {train_auc:.4f}")
    print(f"test candidate AUC:  {test_auc:.4f}")
    print()
    print(f"test top1 normalized exact:     {compare['top1_normalized_exact_match'].mean():.4f}")
    print(f"test reranked normalized exact: {compare['reranked_normalized_exact_match'].mean():.4f}")
    print(f"test oracle@5:                  {compare['oracle_at_5'].mean():.4f}")
    print()

    improved = compare[
        (compare["top1_normalized_exact_match"] == 0) &
        (compare["reranked_normalized_exact_match"] == 1)
    ].copy()
    print(f"reranker improvements over top1: {len(improved)}")
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
            ]
            .head(40)
            .to_string()
        )
        print()

    print("top coefficients:")
    print(coef_df.head(30).to_string(index=False))


if __name__ == "__main__":
    main()