import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.metrics import roc_auc_score


def load_task_data(data_dir):
    data_dir = Path(data_dir)

    y_train = np.load(data_dir / "y_train.npy", allow_pickle=True)
    # y_test = np.load(data_dir / "y_test.npy", allow_pickle=True)
    y_val = np.load(data_dir / "y_val.npy", allow_pickle=True)

    with open(data_dir / "mlb.pkl", "rb") as f:
        mlb = pickle.load(f)

    label_names = np.array(mlb.classes_)

    # return y_train, y_test, label_names
    return y_train, y_val, label_names


def calc_per_label_auc(y_true, y_pred, label_names):
    rows = []

    for i, label in enumerate(label_names):
        if len(np.unique(y_true[:, i])) < 2:
            auc = np.nan
        else:
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])

        rows.append({
            "label_index": i,
            "label": label,
            "auc": auc,
            # "test_positive_count": int(y_true[:, i].sum())
            "val_positive_count": int(y_true[:, i].sum())
        })

    return pd.DataFrame(rows)


def get_rare_labels_from_train(y_train, label_names, count_thr=50, rate_thr=0.01):
    n_train = y_train.shape[0]

    train_positive_count = y_train.sum(axis=0)
    train_positive_rate = train_positive_count / n_train

    rare_mask = (train_positive_count < count_thr) | (train_positive_rate < rate_thr)

    rare_df = pd.DataFrame({
        "label_index": np.arange(len(label_names)),
        "label": label_names,
        "train_positive_count": train_positive_count.astype(int),
        "train_positive_rate": train_positive_rate,
        "is_rare": rare_mask
    })

    return rare_df


def compare_bce_vs_cbl_rare_auc(
    data_dir,
    bce_pred_path,
    # asl_pred_path,
    cbl_pred_path,
    output_csv="rare_label_auc_comparison.csv"
):
    y_train, y_val, label_names = load_task_data(data_dir)
    # y_train, y_test, label_names = load_task_data(data_dir)

    y_pred_bce = np.load(bce_pred_path, allow_pickle=True)
    # y_pred_asl = np.load(asl_pred_path, allow_pickle=True)
    y_pred_cbl = np.load(cbl_pred_path, allow_pickle=True)

    rare_info_df = get_rare_labels_from_train(y_train, label_names)

    # bce_auc_df = calc_per_label_auc(y_test, y_pred_bce, label_names)
    bce_auc_df = calc_per_label_auc(y_val, y_pred_bce, label_names)
    # asl_auc_df = calc_per_label_auc(y_test, y_pred_asl, label_names)
    # cbl_auc_df = calc_per_label_auc(y_test, y_pred_cbl, label_names)
    cbl_auc_df = calc_per_label_auc(y_val, y_pred_cbl, label_names)

    compare_df = rare_info_df.copy()
    compare_df["bce_auc"] = bce_auc_df["auc"]
    # compare_df["asl_auc"] = asl_auc_df["auc"]
    compare_df["cbl_auc"] = cbl_auc_df["auc"]
    # compare_df["delta_auc"] = compare_df["asl_auc"] - compare_df["bce_auc"]
    compare_df["delta_auc"] = compare_df["cbl_auc"] - compare_df["bce_auc"]

    rare_df = compare_df[compare_df["is_rare"]].copy()

    rare_bce_mean_auc = rare_df["bce_auc"].mean()
    # rare_asl_mean_auc = rare_df["asl_auc"].mean()
    rare_cbl_mean_auc = rare_df["cbl_auc"].mean()
    # rare_delta_mean_auc = rare_asl_mean_auc - rare_bce_mean_auc
    rare_delta_mean_auc = rare_cbl_mean_auc - rare_bce_mean_auc

    n_rare = len(rare_df)
    n_rare_improved = (rare_df["delta_auc"] > 0).sum()
    n_rare_decreased = (rare_df["delta_auc"] < 0).sum()
    n_rare_same = (rare_df["delta_auc"] == 0).sum()

    # print("\n===== Rare-label mean AUC =====")
    # print(f"BCE rare-label mean AUC: {rare_bce_mean_auc:.6f}")
    # print(f"ASL rare-label mean AUC: {rare_asl_mean_auc:.6f}")
    # print(f"Delta ASL - BCE:         {rare_delta_mean_auc:.6f}")

    # print("\n===== Rare-label improvement count =====")
    # print(f"Number of rare labels:   {n_rare}")
    # print(f"Improved by ASL:         {n_rare_improved}/{n_rare}")
    # print(f"Decreased by ASL:        {n_rare_decreased}/{n_rare}")
    # print(f"Unchanged:               {n_rare_same}/{n_rare}")

    print("\n===== Rare-label mean AUC =====")
    print(f"BCE rare-label mean AUC: {rare_bce_mean_auc:.6f}")
    print(f"CBL rare-label mean AUC: {rare_cbl_mean_auc:.6f}")
    print(f"Delta CBL - BCE:         {rare_delta_mean_auc:.6f}")

    print("\n===== Rare-label improvement count =====")
    print(f"Number of rare labels:   {n_rare}")
    print(f"Improved by CBL:         {n_rare_improved}/{n_rare}")
    print(f"Decreased by CBL:        {n_rare_decreased}/{n_rare}")
    print(f"Unchanged:               {n_rare_same}/{n_rare}")

    compare_df.to_csv(output_csv, index=False)
    print(f"\nSaved comparison table to: {output_csv}")

    return compare_df, rare_df


if __name__ == "__main__":

    # For ALL task:
    data_dir = "../outputs_CBL/exp0/data"

    # Old BCE prediction --------------> BCE config
    # bce_pred_path = "../outputs/exp0/models/fastai_xresnet1d101/y_test_pred.npy"
    bce_pred_path = "../outputs/exp0/models/fastai_xresnet1d101/y_val_pred.npy"

    # New ASL prediction ------------> ASL config
    # asl_pred_path = "../outputs_asl/exp0/models/fastai_xresnet1d101_asl/y_test_pred.npy"

    # New CBL prediction ------------> CBL config
    # cbl_pred_path = "../outputs_CBL/exp0/models/fastai_xresnet1d101_cbl/y_test_pred.npy"
    cbl_pred_path = "../outputs_CBL/exp0/models/fastai_xresnet1d101_cbl/y_val_pred.npy"

    compare_df, rare_df = compare_bce_vs_cbl_rare_auc(
        data_dir=data_dir,
        bce_pred_path=bce_pred_path,
        cbl_pred_path=cbl_pred_path,
        output_csv="all_task_bce_vs_cbl_rare_auc_val.csv"
    )

    print("\n===== Most improved rare labels =====")
    print(
        rare_df.sort_values("delta_auc", ascending=False)
        [["label", "train_positive_count", "bce_auc", "cbl_auc", "delta_auc"]]
        .head(15)
    )

    print("\n===== Most decreased rare labels =====")
    print(
        rare_df.sort_values("delta_auc", ascending=True)
        [["label", "train_positive_count", "bce_auc", "cbl_auc", "delta_auc"]]
        .head(15)
    )