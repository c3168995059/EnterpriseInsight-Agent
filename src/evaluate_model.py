from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def evaluate_model(y_true, y_pred, model_name="Model"):
    """评估二分类模型，故障类别为1。"""

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true, y_pred, zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_true, y_pred, zero_division=0
    )

    cm = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    print(f"\n===== {model_name} 评估结果 =====")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\n混淆矩阵：")
    print(cm)

    print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn)
    }
