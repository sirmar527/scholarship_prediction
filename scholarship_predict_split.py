"""
Scholarship Prediction — Split-Model Pipeline (Approach C).

Предсказывает, получит ли студент стипендию в следующем семестре,
используя ДВЕ отдельные модели:
  • Модель «Удержание»  — для студентов, у которых стипендия ЕСТЬ:
                           предсказывает, сохранят ли они её.
  • Модель «Получение»  — для студентов, у которых стипендии НЕТ:
                           предсказывает, получат ли они её.

Пайплайн:
  1. Загрузка и фильтрация данных (очная, основная ведомость)
  2. Инженерия признаков (агрегаты по семестрам)
  3. Построение обучающих пар (сем N → сем N+1)
  4. Обучение двух CatBoost-моделей (split by prev_scholarship)
  5. Объединение предсказаний + 4-классовая оценка + сохранение

Запуск:
    python scholarship_predict_split.py
    python scholarship_predict_split.py --data path/to/file.xlsx
    python scholarship_predict_split.py --data path/to/file.xlsx --output results/

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
CLASS_B_LABELS = {
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

    for col in overall.columns:
        if col.endswith("_n") and col != "sem_num":
            overall[col] = overall[col].fillna(0).astype(int)
        if col.endswith(("_any_fail", "_any_retake", "_n_fails")):
            overall[col] = overall[col].fillna(0).astype(int)

    overall["scholarship"] = 0
    overall.loc[
        (overall["any_fail"] == 0) & (overall["any_retake"] == 0), "scholarship"
    ] = 1
    overall.loc[overall["sem_num"] == 1, "scholarship"] = 1

    print(f"  → {len(overall):,} (студент, семестр) строк, "
          f"{len(overall.columns)} колонок")
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

    def variant_b(p, n):
        if p == 1 and n == 1: return 1
        if p == 1 and n == 0: return 2
        if p == 0 and n == 1: return 3
        return 4

    pairs_df["class_B"] = [
        variant_b(p, n)
        for p, n in zip(pairs_df["prev_scholarship"],
                        pairs_df["target_scholarship"])
    ]

    print(f"  → {len(pairs_df):,} пар")
    print(f"  Стипендия в след. семестре: "
          f"{pairs_df['target_scholarship'].mean():.1%}")
    for cls in [1, 2, 3, 4]:
        cnt = (pairs_df["class_B"] == cls).sum()
        print(f"    {cls} ({CLASS_B_LABELS[cls]:<20}): "
              f"{cnt:>5,} ({cnt / len(pairs_df) * 100:>5.1f}%)")

    return pairs_df


# =====================================================================
# 4. TRAIN SPLIT MODELS
# =====================================================================
def _prepare_features(pairs_df):
    """Return feature matrix, target vector, and feature column names."""
    EXCLUDE = {
        "ЗачетнаяКнижка", "target_scholarship", "target_sem",
        "class_B", "УчебныйПлан", "scholarship",
    }
    feature_cols = [c for c in pairs_df.columns if c not in EXCLUDE]

    X = pairs_df[feature_cols].copy()
    y = pairs_df["target_scholarship"].astype(int)

    for col in X.columns:
        if X[col].isna().any():
            if col.endswith(("_gpa", "_min", "_share_5")):
                X[col] = X[col].fillna(-1)
            else:
                X[col] = X[col].fillna(0)

    return X, y, feature_cols


def _split_by_students(pairs_df, seed=42):
    """80/20 split by student ID — same students never appear in both sets."""
    unique_students = pairs_df["ЗачетнаяКнижка"].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_students)
    split_idx = int(len(unique_students) * 0.8)
    train_students = set(unique_students[:split_idx])
    test_students = set(unique_students[split_idx:])

    train_mask = pairs_df["ЗачетнаяКнижка"].isin(train_students)
    test_mask = pairs_df["ЗачетнаяКнижка"].isin(test_students)
    return train_mask, test_mask, train_students, test_students


def _train_submodel(X_train, y_train, X_test, y_test, label):
    """Train a single CatBoost binary classifier and return it."""
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
    print(f"    Лучшая итерация: {best_iter}")
    return model, best_iter


def train_split_models(pairs_df):
    print(f"[4/5] Обучение двух CatBoost-моделей...")

    X, y, feature_cols = _prepare_features(pairs_df)
    train_mask, test_mask, train_students, test_students = _split_by_students(
        pairs_df
    )
    prev = pairs_df["prev_scholarship"].values

    models = {}

    # ── Model 1: students who HAVE a scholarship ──
    print(f"\n  Модель «Удержание» (prev_scholarship = 1):")
    has_sch = prev == 1
    tr1 = train_mask & has_sch
    te1 = test_mask & has_sch
    print(f"    Train: {tr1.sum():,} пар | Test: {te1.sum():,} пар")

    model_keep, iter_keep = _train_submodel(
        X[tr1], y[tr1], X[te1], y[te1], "keep"
    )
    models["keep"] = {
        "model": model_keep,
        "best_iter": iter_keep,
        "train_mask": tr1,
        "test_mask": te1,
    }

    # ── Model 2: students who DON'T have a scholarship ──
    print(f"\n  Модель «Получение» (prev_scholarship = 0):")
    no_sch = prev == 0
    tr2 = train_mask & no_sch
    te2 = test_mask & no_sch
    print(f"    Train: {tr2.sum():,} пар | Test: {te2.sum():,} пар")

    model_gain, iter_gain = _train_submodel(
        X[tr2], y[tr2], X[te2], y[te2], "gain"
    )
    models["gain"] = {
        "model": model_gain,
        "best_iter": iter_gain,
        "train_mask": tr2,
        "test_mask": te2,
    }

    return (models, X, y, feature_cols, train_mask, test_mask,
            train_students, test_students)


# =====================================================================
# 5. EVALUATE AND SAVE
# =====================================================================
def evaluate_and_save(models, X, y, feature_cols, train_mask, test_mask,
                      train_students, test_students, pairs_df, output_dir):
    print(f"\n[5/5] Оценка и сохранение результатов...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prev = pairs_df["prev_scholarship"].values
    true_b = pairs_df["class_B"].values

    # ── Assemble full test-set predictions ──
    test_idx = np.where(test_mask)[0]
    n_test = len(test_idx)

    y_pred_full = np.zeros(n_test, dtype=int)
    y_proba_full = np.zeros(n_test)
    pred_b_full = np.zeros(n_test, dtype=int)

    prev_test = prev[test_mask]
    y_test = y[test_mask].values
    true_b_test = true_b[test_mask]

    # Fill in predictions from the «Удержание» model (had scholarship)
    m_keep = models["keep"]
    keep_mask_in_test = prev_test == 1
    if keep_mask_in_test.sum() > 0:
        X_keep = X[m_keep["test_mask"]]
        pred_keep = m_keep["model"].predict(X_keep).flatten().astype(int)
        proba_keep = m_keep["model"].predict_proba(X_keep)[:, 1]
        y_pred_full[keep_mask_in_test] = pred_keep
        y_proba_full[keep_mask_in_test] = proba_keep
        # class_B: 1 = kept, 2 = lost
        pred_b_full[keep_mask_in_test] = np.where(pred_keep == 1, 1, 2)

    # Fill in predictions from the «Получение» model (no scholarship)
    m_gain = models["gain"]
    gain_mask_in_test = prev_test == 0
    if gain_mask_in_test.sum() > 0:
        X_gain = X[m_gain["test_mask"]]
        pred_gain = m_gain["model"].predict(X_gain).flatten().astype(int)
        proba_gain = m_gain["model"].predict_proba(X_gain)[:, 1]
        y_pred_full[gain_mask_in_test] = pred_gain
        y_proba_full[gain_mask_in_test] = proba_gain
        # class_B: 3 = gained, 4 = still no
        pred_b_full[gain_mask_in_test] = np.where(pred_gain == 1, 3, 4)

    # ── Binary metrics ──
    acc = accuracy_score(y_test, y_pred_full)
    f1_macro = f1_score(y_test, y_pred_full, average="macro")
    auc = roc_auc_score(y_test, y_proba_full)

    print(f"\n{'=' * 70}")
    print(f"  РЕЗУЛЬТАТЫ — ОБЪЕДИНЁННЫЕ ({n_test:,} пар)")
    print(f"{'=' * 70}")
    print(f"  Accuracy:   {acc:.1%}")
    print(f"  F1 (macro): {f1_macro:.3f}")
    print(f"  ROC-AUC:    {auc:.3f}")
    print()

    report = classification_report(
        y_test, y_pred_full,
        target_names=["Без стипендии", "Со стипендией"],
        digits=3,
    )
    print(report)

    cm = confusion_matrix(y_test, y_pred_full)
    cm_df = pd.DataFrame(
        cm,
        index=["Факт: без", "Факт: со"],
        columns=["Пред: без", "Пред: со"],
    )
    print("Confusion matrix (binary):")
    print(cm_df.to_string())

    # ── 4-class metrics ──
    print(f"\nVariant B — 4 класса:")
    cm_b = confusion_matrix(true_b_test, pred_b_full, labels=[1, 2, 3, 4])
    cm_b_df = pd.DataFrame(
        cm_b,
        index=[f"Факт: {CLASS_B_LABELS[i]}" for i in [1, 2, 3, 4]],
        columns=[f"Пред: {i}" for i in [1, 2, 3, 4]],
    )
    print(cm_b_df.to_string())

    print(f"\nPer-class accuracy:")
    class_b_accuracies = {}
    for cls in [1, 2, 3, 4]:
        mask_cls = true_b_test == cls
        if mask_cls.sum() > 0:
            acc_cls = (pred_b_full[mask_cls] == cls).mean()
            class_b_accuracies[cls] = round(acc_cls, 4)
            print(f"  {cls} ({CLASS_B_LABELS[cls]:<20}): "
                  f"{acc_cls:.1%} ({mask_cls.sum():,} cases)")

    # ── Per-submodel reports ──
    for sub_name, sub_label, group_val in [
        ("keep", "Модель «Удержание» (стипендия была)", 1),
        ("gain", "Модель «Получение» (стипендии не было)", 0),
    ]:
        m = models[sub_name]
        X_sub = X[m["test_mask"]]
        y_sub = y[m["test_mask"]].values
        pred_sub = m["model"].predict(X_sub).flatten().astype(int)
        proba_sub = m["model"].predict_proba(X_sub)[:, 1]

        sub_acc = accuracy_score(y_sub, pred_sub)
        sub_f1 = f1_score(y_sub, pred_sub, average="macro")
        sub_auc = roc_auc_score(y_sub, proba_sub) if len(set(y_sub)) > 1 else 0

        print(f"\n{'─' * 70}")
        print(f"  {sub_label}")
        print(f"{'─' * 70}")
        print(f"  Accuracy: {sub_acc:.1%} | F1 (macro): {sub_f1:.3f} | "
              f"AUC: {sub_auc:.3f}")
        print(classification_report(
            y_sub, pred_sub,
            target_names=["Нет", "Да"],
            digits=3,
        ))

    # ── Feature importance ──
    for sub_name, sub_label in [
        ("keep", "Модель «Удержание»"),
        ("gain", "Модель «Получение»"),
    ]:
        imp = models[sub_name]["model"].get_feature_importance()
        imp_df = pd.DataFrame({"feature": feature_cols, "importance": imp})
        imp_df = imp_df.sort_values("importance", ascending=False)
        print(f"\nTop-10 features — {sub_label}:")
        for _, row in imp_df.head(10).iterrows():
            print(f"  {row['feature']:<40} {row['importance']:>6.1f}")

    # ── Per-semester breakdown ──
    test_sems = pairs_df.loc[test_mask, "sem_num"].values
    print(f"\nПо семестрам (объединённые):")
    for from_s in sorted(set(test_sems)):
        mask_s = test_sems == from_s
        if mask_s.sum() < 10:
            continue
        acc_s = accuracy_score(y_test[mask_s], y_pred_full[mask_s])
        y_true_s = y_test[mask_s]
        if len(set(y_true_s)) > 1:
            auc_s = roc_auc_score(y_true_s, y_proba_full[mask_s])
        else:
            auc_s = 0
        print(f"  {from_s}→{from_s + 1}: "
              f"acc={acc_s:.1%}, AUC={auc_s:.3f}, n={mask_s.sum()}")

    # ── Save predictions ──
    students_test = pairs_df.loc[test_mask, "ЗачетнаяКнижка"].values
    pred_df = pd.DataFrame({
        "ЗачетнаяКнижка": students_test,
        "sem_num": pairs_df.loc[test_mask, "sem_num"].values,
        "target_sem": pairs_df.loc[test_mask, "target_sem"].values,
        "prev_scholarship": prev_test,
        "prob_scholarship": np.round(y_proba_full, 4),
        "pred_scholarship": y_pred_full,
        "actual_scholarship": y_test,
        "pred_class_B": pred_b_full,
        "actual_class_B": true_b_test,
        "risk_score": np.round(1 - y_proba_full, 4),
        "model_used": np.where(prev_test == 1, "keep", "gain"),
    })
    pred_csv = output_dir / "predictions.csv"
    pred_df.to_csv(pred_csv, index=False, encoding="utf-8-sig")
    print(f"\n  Сохранено: {pred_csv}")

    # ── Save models ──
    for sub_name in ["keep", "gain"]:
        model_path = output_dir / f"scholarship_model_{sub_name}.cbm"
        models[sub_name]["model"].save_model(str(model_path))
        print(f"  Сохранено: {model_path}")

    # ── Save summary ──
    summary_path = output_dir / "summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Scholarship Prediction — Split-Model Pipeline (Approach C)\n\n")
        f.write(f"**Дата:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("## Подход\n\n")
        f.write("Две отдельные модели CatBoostClassifier:\n\n")
        f.write(f"- **Модель «Удержание»** — для студентов со стипендией "
                f"({models['keep']['best_iter']} итераций)\n")
        f.write(f"- **Модель «Получение»** — для студентов без стипендии "
                f"({models['gain']['best_iter']} итераций)\n\n")
        f.write(f"## Данные\n\n")
        f.write(f"- Train: {train_mask.sum():,} пар "
                f"({len(train_students):,} студентов)\n")
        f.write(f"- Test: {test_mask.sum():,} пар "
                f"({len(test_students):,} студентов)\n")
        f.write(f"- Признаков: {len(feature_cols)}\n\n")
        f.write(f"## Объединённые метрики\n\n")
        f.write(f"| Метрика | Значение |\n|---|---|\n")
        f.write(f"| Accuracy | {acc:.1%} |\n")
        f.write(f"| F1 (macro) | {f1_macro:.3f} |\n")
        f.write(f"| ROC-AUC | {auc:.3f} |\n\n")
        f.write(f"## Classification report\n\n```\n{report}```\n\n")
        f.write(f"## Confusion matrix (binary)\n\n"
                f"```\n{cm_df.to_string()}\n```\n\n")
        f.write(f"## Variant B — 4 класса\n\n"
                f"```\n{cm_b_df.to_string()}\n```\n\n")
        f.write("## Per-class accuracy\n\n")
        for cls in [1, 2, 3, 4]:
            if cls in class_b_accuracies:
                f.write(f"- **{CLASS_B_LABELS[cls]}**: "
                        f"{class_b_accuracies[cls]:.1%}\n")

        f.write(f"\n## Top-10 features\n\n")
        for sub_name, sub_label in [
            ("keep", "Модель «Удержание»"),
            ("gain", "Модель «Получение»"),
        ]:
            f.write(f"\n### {sub_label}\n\n")
            imp = models[sub_name]["model"].get_feature_importance()
            imp_df = pd.DataFrame({"feature": feature_cols, "importance": imp})
            imp_df = imp_df.sort_values("importance", ascending=False)
            for _, row in imp_df.head(10).iterrows():
                f.write(f"- `{row['feature']}`: {row['importance']:.1f}\n")

    print(f"  Сохранено: {summary_path}")

    # ── Save model card ──
    card = {
        "model_type": "Split CatBoostClassifier (two binary models)",
        "target": "scholarship next semester (yes/no)",
        "approach": "Approach C — separate models per prev_scholarship group",
        "features": feature_cols,
        "n_features": len(feature_cols),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "submodels": {
            "keep": {
                "description": "Students WITH current scholarship — will they keep it?",
                "best_iteration": models["keep"]["best_iter"],
                "n_train": int(models["keep"]["train_mask"].sum()),
                "n_test": int(models["keep"]["test_mask"].sum()),
            },
            "gain": {
                "description": "Students WITHOUT current scholarship — will they gain it?",
                "best_iteration": models["gain"]["best_iter"],
                "n_train": int(models["gain"]["train_mask"].sum()),
                "n_test": int(models["gain"]["test_mask"].sum()),
            },
        },
        "combined_metrics": {
            "accuracy": round(acc, 4),
            "f1_macro": round(f1_macro, 4),
            "roc_auc": round(auc, 4),
        },
        "variant_b_accuracy": {
            str(cls): class_b_accuracies.get(cls, None)
            for cls in [1, 2, 3, 4]
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
        description="Scholarship Prediction — Split-Model Pipeline (Approach C)"
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

    (models, X, y, feature_cols, train_mask, test_mask,
     train_students, test_students) = train_split_models(pairs_df)

    evaluate_and_save(
        models, X, y, feature_cols, train_mask, test_mask,
        train_students, test_students, pairs_df, Path(args.output),
    )

    print(f"\n{'=' * 70}")
    print(f"  Готово! Результаты в папке: {Path(args.output).resolve()}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
