"""
Scholarship Prediction — полный пайплайн.

Предсказывает, получит ли студент стипендию в следующем семестре,
на основе его оценок и аттестационных баллов.

Пайплайн:
  1. Загрузка и фильтрация данных (очная, основная ведомость)
  2. Инженерия признаков (агрегаты по семестрам)
  3. Построение обучающих пар (сем N → сем N+1)
  4. Обучение CatBoost (бинарная: стипендия да/нет)
  5. Оценка + Variant B (4 класса) + сохранение результатов

Запуск:
    python scholarship_predict.py
    python scholarship_predict.py --data path/to/file.xlsx
    python scholarship_predict.py --data path/to/file.xlsx --output results/

Зависимости:
    pip install pandas numpy catboost scikit-learn openpyxl
"""

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

# =====================================================================
# CONSTANTS
# =====================================================================
SEMESTER_MAP = {
    "Первый семестр": 1, "Второй семестр": 2, "Третий семестр": 3,
    "Четвертый семестр": 4, "Пятый семестр": 5, "Шестой семестр": 6,
    "Седьмой семестр": 7, "Восьмой семестр": 8, "Девятый семестр": 9,
    "Десятый семестр": 10,
}
GRADE_MAP = {
    "Отлично": 5, "Хорошо": 4, "Удовлетворительно": 3,
    "Неудовлетворительно": 2, "Зачтено": 4, "Не зачтено": 2, "Неявка": 2,
}
PASS_GRADES = {"Отлично", "Хорошо", "Зачтено"}
FAIL_GRADES = {"Удовлетворительно", "Неудовлетворительно", "Не зачтено", "Неявка"}
VARIANT_B_LABELS = {
    1: "Была→сохранил", 2: "Была→потерял",
    3: "Не было→получил", 4: "Не было→нет",
}

DATA_FILE_PATTERNS = [
    "ГОСТ_Р_70946-2023-Приложение-8_sorted.xlsx",
    "ГОСТ Р 70946-2023-Приложение-8_sorted.xlsx",
    "ГОСТ*Приложение*.xlsx",
]


# =====================================================================
# 1. DATA LOADING
# =====================================================================
def find_data_file(directory: Path) -> Path:
    for name in DATA_FILE_PATTERNS:
        if "*" not in name:
            candidate = directory / name
            if candidate.exists():
                return candidate
    for pattern in DATA_FILE_PATTERNS:
        if "*" in pattern:
            matches = list(directory.glob(pattern))
            if matches:
                return matches[0]
    raise FileNotFoundError(
        f"Не нашёл xlsx с данными в {directory}.\n"
        f"Положите файл рядом со скриптом или укажите --data путь/к/файлу.xlsx"
    )


def load_data(path=None):
    if path is None:
        path = find_data_file(Path(__file__).parent)
    path = Path(path)
    print(f"[1/5] Загрузка данных из {path.name}...")

    df = pd.read_excel(path, sheet_name="Sheet1")
    df["sem_num"] = df["ПериодКонтроля"].map(SEMESTER_MAP)
    df = df[df["sem_num"].notna()].copy()
    df["sem_num"] = df["sem_num"].astype(int)
    df = df[(df["ФормаОбучения"] == "Очная") & (df["ТипВедомости"] == "Основная")]

    df["grade_num"] = df["ИтоговаяОтметка"].map(GRADE_MAP)
    df["is_fail"] = df["ИтоговаяОтметка"].isin(FAIL_GRADES).astype(int)
    df["has_retake"] = ((df["Пересдача"] > 0) | (df["Комиссия"] > 0)).astype(int)

    print(f"  → {len(df):,} записей, {df['ЗачетнаяКнижка'].nunique():,} студентов")
    return df


# =====================================================================
# 2. FEATURE ENGINEERING
# =====================================================================
def build_features(df):
    print(f"[2/5] Построение признаков...")

    # Dedup: keep highest-priority ВидКонтроля per (student, subject, semester)
    priority = {"Экзамен": 0, "Зачет с оценкой": 1, "Зачет": 2}
    df = df.copy()
    df["_prio"] = df["ВидКонтроля"].map(priority).fillna(3)
    df = df.sort_values(
        ["ЗачетнаяКнижка", "Дисциплина", "sem_num", "_prio", "grade_num"]
    )
    df = df.drop_duplicates(
        subset=["ЗачетнаяКнижка", "Дисциплина", "sem_num"], keep="first"
    )
    df = df.drop(columns=["_prio"])

    key = ["ЗачетнаяКнижка", "sem_num"]

    # Overall per-semester features
    overall = df.groupby(key).agg(
        n_subjects=("Дисциплина", "nunique"),
        gpa_overall=("grade_num", "mean"),
        min_grade=("grade_num", "min"),
        std_grade=("grade_num", "std"),
        n_fails=("is_fail", "sum"),
        n_retakes=("has_retake", "sum"),
        any_fail=("is_fail", "max"),
        any_retake=("has_retake", "max"),
        УчебныйПлан=("УчебныйПлан", "first"),
    ).reset_index()

    # Share of 5s and 3s
    total_per_key = df.groupby(key).size()
    share_5 = (
        df[df["ИтоговаяОтметка"] == "Отлично"].groupby(key).size() / total_per_key
    )
    share_3 = (
        df[df["ИтоговаяОтметка"] == "Удовлетворительно"].groupby(key).size()
        / total_per_key
    )
    overall = overall.set_index(key)
    overall["share_5"] = share_5.reindex(overall.index).fillna(0)
    overall["share_3"] = share_3.reindex(overall.index).fillna(0)
    overall = overall.reset_index()

    # Fill NaN
    for col in overall.columns:
        if col.endswith("_n") and col != "sem_num":
            overall[col] = overall[col].fillna(0).astype(int)
        if col.endswith(("_any_fail", "_any_retake", "_n_fails")):
            overall[col] = overall[col].fillna(0).astype(int)

    # Scholarship label
    overall["scholarship"] = 0
    overall.loc[
        (overall["any_fail"] == 0) & (overall["any_retake"] == 0), "scholarship"
    ] = 1
    overall.loc[overall["sem_num"] == 1, "scholarship"] = 1

    print(f"  → {len(overall):,} (студент, семестр) строк, {len(overall.columns)} колонок")
    return overall


# =====================================================================
# 3. BUILD TRANSITION PAIRS
# =====================================================================
def build_pairs(features_df):
    print(f"[3/5] Построение обучающих пар (сем N → сем N+1)...")

    features_df = features_df.sort_values(["ЗачетнаяКнижка", "sem_num"])
    pairs_list = []

    for sid, grp in features_df.groupby("ЗачетнаяКнижка"):
        grp = grp.sort_values("sem_num")
        sems = grp["sem_num"].values
        for i in range(len(sems) - 1):
            if sems[i + 1] - sems[i] == 1:
                current = grp[grp["sem_num"] == sems[i]].iloc[0].to_dict()
                next_sch = grp[grp["sem_num"] == sems[i + 1]].iloc[0]["scholarship"]
                current["target_scholarship"] = int(next_sch)
                current["target_sem"] = int(sems[i + 1])
                current["prev_scholarship"] = int(current["scholarship"])
                pairs_list.append(current)

    pairs_df = pd.DataFrame(pairs_list)

    # Variant B
    def variant_b(p, n):
        if p == 1 and n == 1:
            return 1
        if p == 1 and n == 0:
            return 2
        if p == 0 and n == 1:
            return 3
        return 4

    pairs_df["class_B"] = [
        variant_b(p, n)
        for p, n in zip(pairs_df["prev_scholarship"], pairs_df["target_scholarship"])
    ]

    print(f"  → {len(pairs_df):,} пар")
    print(f"  Стипендия в след. семестре: {pairs_df['target_scholarship'].mean():.1%}")
    for cls in [1, 2, 3, 4]:
        cnt = (pairs_df["class_B"] == cls).sum()
        print(f"    {cls} ({VARIANT_B_LABELS[cls]:<20}): {cnt:>5,} ({cnt / len(pairs_df) * 100:>5.1f}%)")

    return pairs_df


# =====================================================================
# 4. TRAIN MODEL
# =====================================================================
def train_model(pairs_df):
    print(f"[4/5] Обучение CatBoost...")

    EXCLUDE = {
        "ЗачетнаяКнижка", "target_scholarship", "target_sem",
        "class_B", "УчебныйПлан", "scholarship",
    }
    feature_cols = [c for c in pairs_df.columns if c not in EXCLUDE]

    X = pairs_df[feature_cols].copy()
    y = pairs_df["target_scholarship"].astype(int)

    # Fill NaN
    for col in X.columns:
        if X[col].isna().any():
            if col.endswith(("_gpa", "_min", "_share_5")):
                X[col] = X[col].fillna(-1)
            else:
                X[col] = X[col].fillna(0)

    # Split BY STUDENTS
    unique_students = pairs_df["ЗачетнаяКнижка"].unique()
    rng = np.random.RandomState(42)
    rng.shuffle(unique_students)
    split = int(len(unique_students) * 0.8)
    train_students = set(unique_students[:split])
    test_students = set(unique_students[split:])

    train_mask = pairs_df["ЗачетнаяКнижка"].isin(train_students)
    test_mask = pairs_df["ЗачетнаяКнижка"].isin(test_students)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print(f"  Train: {len(X_train):,} пар ({len(train_students):,} студентов)")
    print(f"  Test:  {len(X_test):,} пар ({len(test_students):,} студентов)")

    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=0,
        early_stopping_rounds=100,
        use_best_model=True,
        auto_class_weights="Balanced",
    )

    model.fit(
        Pool(X_train, y_train),
        eval_set=Pool(X_test, y_test),
    )
    best_iter = model.get_best_iteration()
    print(f"  Лучшая итерация: {best_iter}")

    return model, X_train, X_test, y_train, y_test, feature_cols, \
           test_mask, best_iter


# =====================================================================
# 5. EVALUATE AND SAVE
# =====================================================================
def evaluate_and_save(model, X_train, X_test, y_train, y_test, feature_cols,
                      test_mask, best_iter, pairs_df, output_dir):
    print(f"[5/5] Оценка и сохранение результатов...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test).flatten().astype(int)
    y_proba = model.predict_proba(X_test)[:, 1]

    # --- Binary metrics ---
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n{'=' * 70}")
    print(f"  РЕЗУЛЬТАТЫ ({len(X_test):,} пар)")
    print(f"{'=' * 70}")
    print(f"  Accuracy:   {acc:.1%}")
    print(f"  F1 (macro): {f1_macro:.3f}")
    print(f"  ROC-AUC:    {auc:.3f}")
    print()

    report = classification_report(
        y_test, y_pred,
        target_names=["Без стипендии", "Со стипендией"],
        digits=3,
    )
    print(report)

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Факт: без", "Факт: со"],
        columns=["Пред: без", "Пред: со"],
    )
    print("Confusion matrix (binary):")
    print(cm_df.to_string())

    # --- Variant B ---
    prev_sch = pairs_df.loc[test_mask, "prev_scholarship"].values
    class_b_true = pairs_df.loc[test_mask, "class_B"].values

    def to_class_b(p, n):
        if p == 1 and n == 1: return 1
        if p == 1 and n == 0: return 2
        if p == 0 and n == 1: return 3
        return 4

    class_b_pred = np.array([to_class_b(p, n) for p, n in zip(prev_sch, y_pred)])

    print(f"\nVariant B — 4 класса:")
    cm_b = confusion_matrix(class_b_true, class_b_pred, labels=[1, 2, 3, 4])
    cm_b_df = pd.DataFrame(
        cm_b,
        index=[f"Факт: {VARIANT_B_LABELS[i]}" for i in [1, 2, 3, 4]],
        columns=[f"Пред: {i}" for i in [1, 2, 3, 4]],
    )
    print(cm_b_df.to_string())

    print(f"\nPer-class accuracy:")
    for cls in [1, 2, 3, 4]:
        mask_cls = class_b_true == cls
        if mask_cls.sum() > 0:
            acc_cls = (class_b_pred[mask_cls] == cls).mean()
            print(f"  {cls} ({VARIANT_B_LABELS[cls]:<20}): "
                  f"{acc_cls:.1%} ({mask_cls.sum():,} cases)")

    # --- Feature importance ---
    importance = model.get_feature_importance()
    imp_df = pd.DataFrame({"feature": feature_cols, "importance": importance})
    imp_df = imp_df.sort_values("importance", ascending=False)
    print(f"\nTop-15 features:")
    for _, row in imp_df.head(15).iterrows():
        print(f"  {row['feature']:<40} {row['importance']:>6.1f}")

    # --- Per-semester ---
    test_sems = pairs_df.loc[test_mask, "sem_num"].values
    print(f"\nПо семестрам:")
    for from_s in sorted(set(test_sems)):
        mask_s = test_sems == from_s
        if mask_s.sum() < 10:
            continue
        acc_s = accuracy_score(y_test.values[mask_s], y_pred[mask_s])
        y_true_s = y_test.values[mask_s]
        if len(set(y_true_s)) > 1:
            auc_s = roc_auc_score(y_true_s, y_proba[mask_s])
        else:
            auc_s = 0
        print(f"  {from_s}→{from_s + 1}: acc={acc_s:.1%}, AUC={auc_s:.3f}, n={mask_s.sum()}")

    # --- Save predictions ---
    students_test = pairs_df.loc[test_mask, "ЗачетнаяКнижка"].values
    pred_df = pd.DataFrame({
        "ЗачетнаяКнижка": students_test,
        "sem_num": pairs_df.loc[test_mask, "sem_num"].values,
        "target_sem": pairs_df.loc[test_mask, "target_sem"].values,
        "prev_scholarship": prev_sch,
        "prob_scholarship": np.round(y_proba, 4),
        "pred_scholarship": y_pred,
        "actual_scholarship": y_test.values,
        "pred_class_B": class_b_pred,
        "actual_class_B": class_b_true,
        "risk_score": np.round(1 - y_proba, 4),
    })
    pred_csv = output_dir / "predictions.csv"
    pred_df.to_csv(pred_csv, index=False, encoding="utf-8-sig")
    print(f"\n  Сохранено: {pred_csv}")

    # --- Save model ---
    model_path = output_dir / "scholarship_model.cbm"
    model.save_model(str(model_path))
    print(f"  Сохранено: {model_path}")

    # --- Save summary ---
    summary_path = output_dir / "summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Scholarship Prediction — Grade-Level Model\n\n")
        f.write(f"**Дата:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"**Модель:** CatBoostClassifier, {best_iter} итераций\n\n")
        f.write(f"## Задача\n\n")
        f.write(f"Предсказание стипендии в следующем семестре (бинарная)\n")
        f.write(f"на основе оценок и аттестационных баллов текущего семестра.\n\n")
        f.write(f"## Данные\n\n")
        f.write(f"- Train: {len(X_train):,} пар\n")
        f.write(f"- Test: {len(X_test):,} пар\n")
        f.write(f"- Признаков: {len(feature_cols)}\n\n")
        f.write(f"## Метрики\n\n")
        f.write(f"| Метрика | Значение |\n|---|---|\n")
        f.write(f"| Accuracy | {acc:.1%} |\n")
        f.write(f"| F1 (macro) | {f1_macro:.3f} |\n")
        f.write(f"| ROC-AUC | {auc:.3f} |\n\n")
        f.write(f"## Classification report\n\n```\n{report}```\n\n")
        f.write(f"## Confusion matrix (binary)\n\n```\n{cm_df.to_string()}\n```\n\n")
        f.write(f"## Variant B — 4 класса\n\n```\n{cm_b_df.to_string()}\n```\n\n")
        f.write(f"## Top-15 features\n\n")
        for _, row in imp_df.head(15).iterrows():
            f.write(f"- `{row['feature']}`: {row['importance']:.1f}\n")
    print(f"  Сохранено: {summary_path}")

    # --- Save model card ---
    card = {
        "model_type": "CatBoostClassifier (binary)",
        "target": "scholarship next semester (yes/no)",
        "features": feature_cols,
        "n_features": len(feature_cols),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "best_iteration": best_iter,
        "metrics": {
            "accuracy": round(acc, 4),
            "f1_macro": round(f1_macro, 4),
            "roc_auc": round(auc, 4),
        },
        "variant_b_accuracy": {
            str(cls): round((class_b_pred[class_b_true == cls] == cls).mean(), 4)
            for cls in [1, 2, 3, 4]
            if (class_b_true == cls).sum() > 0
        },
        "random_seed": 42,
        "date": datetime.now().isoformat(),
    }
    card_path = output_dir / "model_card.json"
    with open(card_path, "w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=False, indent=2)
    print(f"  Сохранено: {card_path}")


# =====================================================================
# MAIN
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Scholarship Prediction — Grade-Level Model"
    )
    parser.add_argument("--data", type=str, default=None,
                        help="Путь к xlsx-файлу")
    parser.add_argument("--output", type=str, default="output",
                        help="Папка для результатов (default: output/)")
    args = parser.parse_args()

    # Pipeline
    df = load_data(args.data)
    features_df = build_features(df)
    pairs_df = build_pairs(features_df)

    model, X_train, X_test, y_train, y_test, feature_cols, \
        test_mask, best_iter = train_model(pairs_df)

    evaluate_and_save(
        model, X_train, X_test, y_train, y_test, feature_cols,
        test_mask, best_iter, pairs_df, Path(args.output)
    )

    print(f"\n{'=' * 70}")
    print(f"  Готово! Результаты в папке: {Path(args.output).resolve()}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
