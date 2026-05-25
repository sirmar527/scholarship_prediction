"""
Scholarship Prediction — Scikit-Learn Multi-Model Pipeline.

Replaces the CatBoost-only approach with a multi-model comparison
using scikit-learn's unified API. Trains and evaluates several model
families on the same student-level split, selects the best one,
and produces the same outputs as the original script.

Models compared:
  • Logistic Regression (baseline linear)
  • Random Forest
  • Gradient Boosting (sklearn)
  • XGBoost (sklearn API)
  • LightGBM (sklearn API)

Запуск:
    python scholarship_predict_sklearn.py
    python scholarship_predict_sklearn.py --data path/to/file.xlsx
    python scholarship_predict_sklearn.py --data path/to/file.xlsx --output results/

Зависимости:
    pip install pandas numpy scikit-learn xgboost lightgbm openpyxl
"""

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier

    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

warnings.filterwarnings("ignore")

# =====================================================================
# CONSTANTS
# =====================================================================
SEMESTER_MAP = {
    "Первый семестр": 1,
    "Второй семестр": 2,
    "Третий семестр": 3,
    "Четвертый семестр": 4,
    "Пятый семестр": 5,
    "Шестой семестр": 6,
    "Седьмой семестр": 7,
    "Восьмой семестр": 8,
    "Девятый семестр": 9,
    "Десятый семестр": 10,
}
GRADE_MAP = {
    "Отлично": 5,
    "Хорошо": 4,
    "Удовлетворительно": 3,
    "Неудовлетворительно": 2,
    "Зачтено": 4,
    "Не зачтено": 2,
    "Неявка": 2,
}
SCHOLARSHIP_BLOCKING_GRADES = {
    "Удовлетворительно",
    "Неудовлетворительно",
    "Не зачтено",
    "Неявка",
}
CLASS_B_LABELS = {
    1: "Была→сохранил",
    2: "Была→потерял",
    3: "Не было→получил",
    4: "Не было→нет",
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
        path = find_data_file(Path(__file__).parent / "data")
    path = Path(path)
    print(f"[1/6] Загрузка данных из {path.name}...")

    df = pd.read_excel(path, sheet_name="Sheet1")
    df["sem_num"] = df["ПериодКонтроля"].map(SEMESTER_MAP)
    df = df[df["sem_num"].notna()].copy()
    df["sem_num"] = df["sem_num"].astype(int)
    df = df[(df["ФормаОбучения"] == "Очная") & (df["ТипВедомости"] == "Основная")]

    df["grade_num"] = df["ИтоговаяОтметка"].map(GRADE_MAP)
    df["is_scholarship_blocking"] = (
        df["ИтоговаяОтметка"].isin(SCHOLARSHIP_BLOCKING_GRADES).astype(int)
    )
    df["has_retake"] = ((df["Пересдача"] > 0) | (df["Комиссия"] > 0)).astype(int)

    print(f"  → {len(df):,} записей, {df['ЗачетнаяКнижка'].nunique():,} студентов")
    return df


# =====================================================================
# 2. FEATURE ENGINEERING
# =====================================================================
def build_features(df):
    print(f"[2/6] Построение признаков...")

    priority = {"Экзамен": 0, "Зачет с оценкой": 1, "Зачет": 2}
    df = df.copy()

    # ── Pre-aggregate fail/retake/min_grade across ALL assessment types
    #    per (student, discipline, semester) BEFORE dedup, so that
    #    Курсовая/Курсовой grades are not silently lost. ──
    disc_key = ["ЗачетнаяКнижка", "Дисциплина", "sem_num"]
    disc_flags = (
        df.groupby(disc_key)
        .agg(
            disc_any_block=("is_scholarship_blocking", "max"),
            disc_any_retake=("has_retake", "max"),
            disc_min_grade=("grade_num", "min"),
        )
        .reset_index()
    )

    # ── Dedup: keep primary assessment type row for GPA ──
    df["_prio"] = df["ВидКонтроля"].map(priority).fillna(3)
    df = df.sort_values(
        ["ЗачетнаяКнижка", "Дисциплина", "sem_num", "_prio", "grade_num"]
    )
    df = df.drop_duplicates(
        subset=["ЗачетнаяКнижка", "Дисциплина", "sem_num"], keep="first"
    )
    df = df.drop(columns=["_prio"])

    # ── Overlay pre-aggregated flags onto surviving rows ──
    df = df.merge(disc_flags, on=disc_key, how="left")
    df["is_scholarship_blocking"] = df["disc_any_block"]
    df["has_retake"] = df["disc_any_retake"]
    df["grade_num"] = df[["grade_num", "disc_min_grade"]].min(axis=1)
    df = df.drop(columns=["disc_any_block", "disc_any_retake", "disc_min_grade"])

    key = ["ЗачетнаяКнижка", "sem_num"]

    overall = (
        df.groupby(key)
        .agg(
            n_subjects=("Дисциплина", "nunique"),
            gpa_overall=("grade_num", "mean"),
            min_grade=("grade_num", "min"),
            std_grade=("grade_num", "std"),
            n_blocks=("is_scholarship_blocking", "sum"),
            n_retakes=("has_retake", "sum"),
            any_block=("is_scholarship_blocking", "max"),
            any_retake=("has_retake", "max"),
            УчебныйПлан=("УчебныйПлан", "first"),
        )
        .reset_index()
    )

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

    overall["scholarship"] = 0
    overall.loc[
        (overall["any_block"] == 0) & (overall["any_retake"] == 0), "scholarship"
    ] = 1
    overall.loc[overall["sem_num"] == 1, "scholarship"] = 1

    print(
        f"  → {len(overall):,} (студент, семестр) строк, "
        f"{len(overall.columns)} колонок"
    )
    return overall


# =====================================================================
# 3. BUILD TRANSITION PAIRS
# =====================================================================
def build_pairs(features_df):
    print(f"[3/6] Построение обучающих пар (сем N → сем N+1)...")

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
    print(
        f"  Стипендия в след. семестре: "
        f"{pairs_df['target_scholarship'].mean():.1%}"
    )
    for cls in [1, 2, 3, 4]:
        cnt = (pairs_df["class_B"] == cls).sum()
        print(
            f"    {cls} ({CLASS_B_LABELS[cls]:<20}): "
            f"{cnt:>5,} ({cnt / len(pairs_df) * 100:>5.1f}%)"
        )

    return pairs_df


# =====================================================================
# 4. PREPARE FEATURES & SPLIT
# =====================================================================
EXCLUDE_COLS = {
    "ЗачетнаяКнижка",
    "target_scholarship",
    "target_sem",
    "class_B",
    "УчебныйПлан",
    "scholarship",
}


def _prepare_features(pairs_df):
    """Return feature matrix, target vector, and feature column names."""
    feature_cols = [c for c in pairs_df.columns if c not in EXCLUDE_COLS]

    X = pairs_df[feature_cols].copy()
    y = pairs_df["target_scholarship"].astype(int)

    for col in X.columns:
        if X[col].isna().any():
            if col == "std_grade":
                # NaN from single-subject semesters — use -1 to distinguish
                # from genuine zero-variance (all same grade)
                X[col] = X[col].fillna(-1)
            elif col in ("gpa_overall", "min_grade", "share_5"):
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


# =====================================================================
# 5. MODEL DEFINITIONS
# =====================================================================
def _build_model_zoo():
    """Return a dict of {name: model} to compare."""
    models = {}

    # --- Linear baseline ---
    models["LogisticRegression"] = LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )

    # --- Tree ensembles (sklearn) ---
    models["RandomForest"] = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    models["GradientBoosting"] = GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=15,
        subsample=0.8,
        random_state=42,
        n_iter_no_change=50,
        validation_fraction=0.15,
    )

    # --- XGBoost (if available) ---
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=5,
            min_child_weight=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=5.0,
            scale_pos_weight=1.0,  # handled by sample_weight below
            random_state=42,
            early_stopping_rounds=50,
            eval_metric="auc",
            verbosity=0,
        )

    # --- LightGBM (if available) ---
    if HAS_LGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=5,
            min_child_samples=15,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=5.0,
            is_unbalance=True,
            random_state=42,
            verbosity=-1,
            n_jobs=-1,
        )

    return models


# =====================================================================
# 6. TRAIN ALL MODELS
# =====================================================================
def train_models(pairs_df):
    """Train all models, return results dict keyed by model name."""
    print(f"[4/6] Обучение моделей...")

    X, y, feature_cols = _prepare_features(pairs_df)
    train_mask, test_mask, train_students, test_students = _split_by_students(pairs_df)

    X_train, y_train = X[train_mask].values, y[train_mask].values
    X_test, y_test = X[test_mask].values, y[test_mask].values

    # ── Carve validation set from train for early-stopping models ──
    train_students_list = list(train_students)
    rng = np.random.RandomState(123)
    rng.shuffle(train_students_list)
    val_split = int(len(train_students_list) * 0.85)
    fit_students = set(train_students_list[:val_split])
    val_students = set(train_students_list[val_split:])

    fit_mask = pairs_df["ЗачетнаяКнижка"].isin(fit_students)
    val_mask = pairs_df["ЗачетнаяКнижка"].isin(val_students)

    X_fit, y_fit = X[fit_mask].values, y[fit_mask].values
    X_val, y_val = X[val_mask].values, y[val_mask].values

    print(f"  Fit:  {len(X_fit):,} пар ({len(fit_students):,} студентов)")
    print(f"  Val:  {len(X_val):,} пар ({len(val_students):,} студентов)")
    print(f"  Test: {len(X_test):,} пар ({len(test_students):,} студентов)")
    print(f"  Признаков: {len(feature_cols)}")
    print(f"  Feature list: {feature_cols}")

    # ── Scaler for LogisticRegression ──
    scaler = StandardScaler()
    X_fit_scaled = scaler.fit_transform(X_fit)
    X_val_scaled = scaler.transform(X_val)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ── Compute sample weights for balanced training ──
    from sklearn.utils.class_weight import compute_sample_weight

    fit_sample_weights = compute_sample_weight("balanced", y_fit)
    train_sample_weights = compute_sample_weight("balanced", y_train)

    models_zoo = _build_model_zoo()
    results = {}

    for name, model in models_zoo.items():
        print(f"\n  ── Training {name}...")

        needs_scaling = name == "LogisticRegression"
        needs_early_stopping = name in ("XGBoost", "LightGBM")

        # Choose data
        if needs_scaling:
            # LogReg: train on full train set (no early stopping), scaled
            model.fit(X_train_scaled, y_train, sample_weight=train_sample_weights)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        elif needs_early_stopping:
            # XGBoost / LightGBM: train on fit set, validate on val set
            if name == "XGBoost":
                model.fit(
                    X_fit,
                    y_fit,
                    eval_set=[(X_val, y_val)],
                    sample_weight=fit_sample_weights,
                    verbose=False,
                )
            else:
                # LightGBM
                lgbm_callbacks = [
                    __import__("lightgbm").early_stopping(50, verbose=False),
                    __import__("lightgbm").log_evaluation(period=0),
                ]
                model.fit(
                    X_fit,
                    y_fit,
                    eval_set=[(X_val, y_val)],
                    eval_metric="auc",
                    callbacks=lgbm_callbacks,
                    sample_weight=fit_sample_weights,
                )
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            # RF / sklearn GB: train on full train set, no early stopping needed
            # (sklearn GB has built-in n_iter_no_change using validation_fraction)
            if name == "GradientBoosting":
                # sklearn GB doesn't support sample_weight with n_iter_no_change
                # well — it uses its own internal validation_fraction split
                model.fit(X_train, y_train, sample_weight=train_sample_weights)
            else:
                model.fit(X_train, y_train, sample_weight=train_sample_weights)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

        # Evaluate
        acc = accuracy_score(y_test, y_pred)
        f1m = f1_score(y_test, y_pred, average="macro")
        try:
            auc = roc_auc_score(y_test, y_proba)
        except ValueError:
            auc = 0.0

        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "accuracy": acc,
            "f1_macro": f1m,
            "roc_auc": auc,
        }

        n_iters = "—"
        if hasattr(model, "best_iteration"):
            n_iters = model.best_iteration
        elif hasattr(model, "best_iteration_"):
            n_iters = model.best_iteration_
        elif hasattr(model, "n_iter_"):
            n_iters = model.n_iter_
        elif hasattr(model, "n_estimators_"):
            n_iters = model.n_estimators_

        print(
            f"     acc={acc:.1%}  F1m={f1m:.3f}  AUC={auc:.3f}"
            f"  iters={n_iters}"
        )

    return (
        results,
        X,
        y,
        feature_cols,
        train_mask,
        test_mask,
        train_students,
        test_students,
        scaler,
    )


# =====================================================================
# 7. BASELINES
# =====================================================================
def _run_baselines(pairs_df, test_mask):
    """Compute simple baselines on the test set for comparison."""
    y_test = pairs_df.loc[test_mask, "target_scholarship"].values
    prev_test = pairs_df.loc[test_mask, "prev_scholarship"].values
    gpa_test = pairs_df.loc[test_mask, "gpa_overall"].values
    any_block_test = pairs_df.loc[test_mask, "any_block"].values
    any_retake_test = pairs_df.loc[test_mask, "any_retake"].values

    baselines = {}

    # 1. Majority class
    majority_cls = int(pairs_df.loc[~test_mask, "target_scholarship"].mode()[0])
    pred_majority = np.full_like(y_test, majority_cls)
    baselines["majority"] = {
        "pred": pred_majority,
        "desc": f"Always predict {majority_cls}",
    }

    # 2. Persistence (next = current)
    baselines["persistence"] = {
        "pred": prev_test.copy(),
        "desc": "Predict next = prev_scholarship",
    }

    # 3. Rule: no blocking grades & no retakes → scholarship
    pred_rule = ((any_block_test == 0) & (any_retake_test == 0)).astype(int)
    baselines["rule_no_blocking"] = {
        "pred": pred_rule,
        "desc": "No blocking grades & no retakes → scholarship",
    }

    # 4. GPA thresholds
    for thr in [4.0, 4.2]:
        pred_gpa = (gpa_test >= thr).astype(int)
        baselines[f"gpa_{thr}"] = {
            "pred": pred_gpa,
            "desc": f"GPA ≥ {thr} → scholarship",
        }

    return baselines, y_test


# =====================================================================
# 8. EVALUATE AND SAVE
# =====================================================================
def evaluate_and_save(
    results,
    X,
    y,
    feature_cols,
    train_mask,
    test_mask,
    train_students,
    test_students,
    pairs_df,
    output_dir,
    scaler,
):
    print(f"\n[5/6] Оценка и сохранение результатов...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Select best model by AUC ──
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best = results[best_name]
    model = best["model"]
    y_pred = best["y_pred"]
    y_proba = best["y_proba"]

    print(f"\n  Лучшая модель: {best_name} (AUC={best['roc_auc']:.3f})")

    prev = pairs_df["prev_scholarship"].values
    true_b = pairs_df["class_B"].values

    y_test = y[test_mask].values
    prev_test = prev[test_mask]
    true_b_test = true_b[test_mask]

    # Derive 4-class predictions
    pred_b = np.zeros(len(y_pred), dtype=int)
    had = prev_test == 1
    pred_b[had & (y_pred == 1)] = 1
    pred_b[had & (y_pred == 0)] = 2
    no = prev_test == 0
    pred_b[no & (y_pred == 1)] = 3
    pred_b[no & (y_pred == 0)] = 4

    # ── Baselines ──
    baselines, _ = _run_baselines(pairs_df, test_mask)

    print(f"\n{'=' * 70}")
    print(f"  BASELINES (test set, {len(y_test):,} пар)")
    print(f"{'=' * 70}")
    for name, bl in baselines.items():
        bl_acc = accuracy_score(y_test, bl["pred"])
        bl_f1 = f1_score(y_test, bl["pred"], average="macro")
        try:
            bl_auc = roc_auc_score(y_test, bl["pred"])
        except ValueError:
            bl_auc = 0.0
        print(
            f"  {name:<20} acc={bl_acc:.1%}  F1m={bl_f1:.3f}  "
            f"AUC={bl_auc:.3f}  | {bl['desc']}"
        )

    # ── Model comparison table ──
    print(f"\n{'=' * 70}")
    print(f"  MODEL COMPARISON (test set)")
    print(f"{'=' * 70}")
    print(f"  {'Model':<25} {'Accuracy':>8}  {'F1 macro':>8}  {'ROC-AUC':>8}")
    print(f"  {'─' * 25} {'─' * 8}  {'─' * 8}  {'─' * 8}")
    for name in results:
        r = results[name]
        marker = " ◀ best" if name == best_name else ""
        print(
            f"  {name:<25} {r['accuracy']:>7.1%}  {r['f1_macro']:>8.3f}"
            f"  {r['roc_auc']:>8.3f}{marker}"
        )

    # ── Best model detailed metrics ──
    acc = best["accuracy"]
    f1_macro = best["f1_macro"]
    auc = best["roc_auc"]

    print(f"\n{'=' * 70}")
    print(f"  BEST MODEL: {best_name} ({len(y_test):,} пар)")
    print(f"{'=' * 70}")
    print(f"  Accuracy:   {acc:.1%}")
    print(f"  F1 (macro): {f1_macro:.3f}")
    print(f"  ROC-AUC:    {auc:.3f}")
    print()

    report = classification_report(
        y_test,
        y_pred,
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

    # ── 4-class metrics ──
    print(f"\nVariant B — 4 класса:")
    cm_b = confusion_matrix(true_b_test, pred_b, labels=[1, 2, 3, 4])
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
            acc_cls = (pred_b[mask_cls] == cls).mean()
            class_b_accuracies[cls] = round(acc_cls, 4)
            print(
                f"  {cls} ({CLASS_B_LABELS[cls]:<20}): "
                f"{acc_cls:.1%} ({mask_cls.sum():,} cases)"
            )

    # ── Per-subgroup report ──
    for label, group_val in [
        ("Подгруппа «стипендия была» (prev=1)", 1),
        ("Подгруппа «стипендии не было» (prev=0)", 0),
    ]:
        sub_mask = prev_test == group_val
        if sub_mask.sum() == 0:
            continue
        y_sub = y_test[sub_mask]
        pred_sub = y_pred[sub_mask]
        proba_sub = y_proba[sub_mask]

        sub_acc = accuracy_score(y_sub, pred_sub)
        sub_f1 = f1_score(y_sub, pred_sub, average="macro")
        try:
            sub_auc = roc_auc_score(y_sub, proba_sub)
        except ValueError:
            sub_auc = 0.0

        print(f"\n{'─' * 70}")
        print(f"  {label}")
        print(f"{'─' * 70}")
        print(
            f"  Accuracy: {sub_acc:.1%} | F1 (macro): {sub_f1:.3f} | "
            f"AUC: {sub_auc:.3f}  (n={sub_mask.sum():,})"
        )
        print(
            classification_report(
                y_sub,
                pred_sub,
                target_names=["Нет", "Да"],
                digits=3,
            )
        )

    # ── Feature importance ──
    print(f"\nFeature importances (best model: {best_name}):")
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        imp_df = pd.DataFrame({"feature": feature_cols, "importance": imp})
        imp_df = imp_df.sort_values("importance", ascending=False)
        for _, row in imp_df.iterrows():
            print(f"  {row['feature']:<40} {row['importance']:>8.4f}")
    elif hasattr(model, "coef_"):
        # LogReg: show absolute coefficients
        coef = np.abs(model.coef_[0])
        imp_df = pd.DataFrame({"feature": feature_cols, "importance": coef})
        imp_df = imp_df.sort_values("importance", ascending=False)
        print("  (absolute coefficients on scaled features)")
        for _, row in imp_df.iterrows():
            print(f"  {row['feature']:<40} {row['importance']:>8.4f}")
    else:
        imp_df = pd.DataFrame({"feature": feature_cols, "importance": [0] * len(feature_cols)})
        print("  (model does not expose feature importances)")

    # ── Per-semester breakdown ──
    test_sems = pairs_df.loc[test_mask, "sem_num"].values
    print(f"\nПо семестрам:")
    for from_s in sorted(set(test_sems)):
        mask_s = test_sems == from_s
        if mask_s.sum() < 10:
            continue
        acc_s = accuracy_score(y_test[mask_s], y_pred[mask_s])
        y_true_s = y_test[mask_s]
        if len(set(y_true_s)) > 1:
            auc_s = roc_auc_score(y_true_s, y_proba[mask_s])
        else:
            auc_s = 0
        print(
            f"  {int(from_s)}→{int(from_s) + 1}: "
            f"acc={acc_s:.1%}, AUC={auc_s:.3f}, n={mask_s.sum()}"
        )

    # ── Save predictions ──
    pred_df = pd.DataFrame(
        {
            "ЗачетнаяКнижка": pairs_df.loc[test_mask, "ЗачетнаяКнижка"].values,
            "sem_num": pairs_df.loc[test_mask, "sem_num"].values,
            "target_sem": pairs_df.loc[test_mask, "target_sem"].values,
            "prev_scholarship": prev_test,
            "prob_scholarship": np.round(y_proba, 4),
            "pred_scholarship": y_pred,
            "actual_scholarship": y_test,
            "pred_class_B": pred_b,
            "actual_class_B": true_b_test,
            "risk_score": np.round(1 - y_proba, 4),
            "model_used": best_name,
        }
    )
    pred_csv = output_dir / "predictions_sklearn.csv"
    pred_df.to_csv(pred_csv, index=False, encoding="utf-8-sig")
    print(f"\n  Сохранено: {pred_csv}")

    # ── Save summary ──
    summary_path = output_dir / "summary_sklearn.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Scholarship Prediction — Scikit-Learn Multi-Model Pipeline\n\n")
        f.write(f"**Дата:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        f.write("## Подход\n\n")
        f.write(
            "Сравнение нескольких моделей через scikit-learn API, "
            "`prev_scholarship` как обычный признак.\n\n"
        )

        f.write("## Данные\n\n")
        f.write(
            f"- Train: {train_mask.sum():,} пар "
            f"({len(train_students):,} студентов)\n"
        )
        f.write(
            f"- Test: {test_mask.sum():,} пар "
            f"({len(test_students):,} студентов)\n"
        )
        f.write(f"- Признаков: {len(feature_cols)}\n\n")

        f.write("## Model Comparison\n\n")
        f.write("| Model | Accuracy | F1 macro | ROC-AUC | Best? |\n")
        f.write("|---|---|---|---|---|\n")
        for name in results:
            r = results[name]
            star = "✓" if name == best_name else ""
            f.write(
                f"| {name} | {r['accuracy']:.1%} | {r['f1_macro']:.3f} "
                f"| {r['roc_auc']:.3f} | {star} |\n"
            )

        f.write(f"\n## Baselines (test set)\n\n")
        f.write("| Baseline | Accuracy | F1 macro | Описание |\n|---|---|---|---|\n")
        for name, bl in baselines.items():
            bl_acc = accuracy_score(y_test, bl["pred"])
            bl_f1 = f1_score(y_test, bl["pred"], average="macro")
            f.write(
                f"| {name} | {bl_acc:.1%} | {bl_f1:.3f} | " f"{bl['desc']} |\n"
            )

        f.write(f"\n## Метрики лучшей модели ({best_name})\n\n")
        f.write("| Метрика | Значение |\n|---|---|\n")
        f.write(f"| Accuracy | {acc:.1%} |\n")
        f.write(f"| F1 (macro) | {f1_macro:.3f} |\n")
        f.write(f"| ROC-AUC | {auc:.3f} |\n\n")
        f.write(f"## Classification report\n\n```\n{report}```\n\n")
        f.write(
            f"## Confusion matrix (binary)\n\n" f"```\n{cm_df.to_string()}\n```\n\n"
        )
        f.write(
            f"## Variant B — 4 класса\n\n" f"```\n{cm_b_df.to_string()}\n```\n\n"
        )
        f.write("## Per-class accuracy\n\n")
        for cls in [1, 2, 3, 4]:
            if cls in class_b_accuracies:
                f.write(
                    f"- **{CLASS_B_LABELS[cls]}**: " f"{class_b_accuracies[cls]:.1%}\n"
                )

        f.write(f"\n## Feature importance\n\n")
        for _, row in imp_df.iterrows():
            f.write(f"- `{row['feature']}`: {row['importance']:.4f}\n")

    print(f"  Сохранено: {summary_path}")

    # ── Save model card ──
    card = {
        "pipeline": "Scikit-Learn Multi-Model Comparison",
        "best_model": best_name,
        "target": "scholarship next semester (yes/no)",
        "features": feature_cols,
        "n_features": len(feature_cols),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "all_models": {
            name: {
                "accuracy": round(r["accuracy"], 4),
                "f1_macro": round(r["f1_macro"], 4),
                "roc_auc": round(r["roc_auc"], 4),
            }
            for name, r in results.items()
        },
        "best_model_metrics": {
            "accuracy": round(acc, 4),
            "f1_macro": round(f1_macro, 4),
            "roc_auc": round(auc, 4),
        },
        "baselines": {
            name: {
                "accuracy": round(accuracy_score(y_test, bl["pred"]), 4),
                "description": bl["desc"],
            }
            for name, bl in baselines.items()
        },
        "variant_b_accuracy": {
            str(cls): class_b_accuracies.get(cls, None) for cls in [1, 2, 3, 4]
        },
        "random_seed": 42,
        "date": datetime.now().isoformat(),
    }
    card_path = output_dir / "model_card_sklearn.json"
    with open(card_path, "w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=False, indent=2)
    print(f"  Сохранено: {card_path}")

    return best_name


# =====================================================================
# MAIN
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Scholarship Prediction — Scikit-Learn Multi-Model Pipeline"
    )
    parser.add_argument("--data", type=str, default=None, help="Путь к xlsx-файлу")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/sklearn",
        help="Папка для результатов (default: outputs/sklearn/)",
    )
    args = parser.parse_args()

    # Pipeline
    print(f"{'=' * 70}")
    print(f"  Scholarship Prediction — Scikit-Learn Multi-Model Pipeline")
    print(f"{'=' * 70}\n")

    df = load_data(args.data)
    features_df = build_features(df)
    pairs_df = build_pairs(features_df)

    (
        results,
        X,
        y,
        feature_cols,
        train_mask,
        test_mask,
        train_students,
        test_students,
        scaler,
    ) = train_models(pairs_df)

    best_name = evaluate_and_save(
        results,
        X,
        y,
        feature_cols,
        train_mask,
        test_mask,
        train_students,
        test_students,
        pairs_df,
        Path(args.output),
        scaler,
    )

    print(f"\n{'=' * 70}")
    print(f"  Готово! Лучшая модель: {best_name}")
    print(f"  Результаты в папке: {Path(args.output).resolve()}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
