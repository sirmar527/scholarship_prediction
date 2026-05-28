"""
same thing but only 1 pair
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight

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
PASS_FAIL_CONTROL_TYPES = {"Зачет"}
PASS_FAIL_GRADES = {"Зачтено", "Не зачтено"}

DEDUP_PRIORITY = {
    "Экзамен": 0,
    "Зачет с оценкой": 1,
    "Курсовая работа": 2,
    "Курсовой проект": 2,
    "ГАК": 2,
    "Защита ВКР": 2,
    "Зачет": 3,
}
DEDUP_PRIORITY_DEFAULT = 4

KEPT_VEDOMOST_TYPES = {"Основная", "Перезачет", "Пересдача", "Пересдача с комиссией"}
RETAKE_VEDOMOST_TYPES = {"Пересдача", "Пересдача с комиссией"}

CLASS_B_LABELS = {
    1: "Чисто->чисто",
    2: "Чисто->провал",
    3: "Провал->чисто",
    4: "Провал->провал",
}

DATA_FILE_PATTERNS = [
    "output_bachelors_both_sem2_3_ochnaya.xlsx",
]


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
        f"Нет xlsx с данными в {directory}.\n"
        f"Укажите путь к файлу"
    )


def load_data(path=None):
    if path is None:
        path = find_data_file(Path(__file__).parent / "data")
    path = Path(path)
    print(f"[1/5] Загрузка данных из {path.name}...")

    df = pd.read_excel(path, sheet_name="Sheet1")
    df["sem_num"] = df["ПериодКонтроля"].map(SEMESTER_MAP)
    df = df[df["sem_num"].notna()].copy()
    df["sem_num"] = df["sem_num"].astype(int)
    df = df[df["ФормаОбучения"] == "Очная"]

    df = df[df["ТипВедомости"].isin(KEPT_VEDOMOST_TYPES)].copy()

    osn_keys = set(
        zip(
            df.loc[df["ТипВедомости"] == "Основная", "ЗачетнаяКнижка"],
            df.loc[df["ТипВедомости"] == "Основная", "sem_num"],
            df.loc[df["ТипВедомости"] == "Основная", "Дисциплина"],
        )
    )
    is_retake_row = df["ТипВедомости"].isin(RETAKE_VEDOMOST_TYPES)
    keys = list(zip(df["ЗачетнаяКнижка"], df["sem_num"], df["Дисциплина"]))
    has_osn_partner = pd.Series([k in osn_keys for k in keys], index=df.index)
    drop_mask = is_retake_row & has_osn_partner
    if drop_mask.any():
        df = df[~drop_mask].copy()

    df["grade_num"] = df["ИтоговаяОтметка"].map(GRADE_MAP)
    df["is_scholarship_blocking"] = (
        df["ИтоговаяОтметка"].isin(SCHOLARSHIP_BLOCKING_GRADES).astype(int)
    )
    df["has_retake"] = (
        (df["Пересдача"] > 0)
        | (df["Комиссия"] > 0)
        | (df["ИтоговаяОтметка"] == "Неявка")
    ).astype(int)
    df.loc[df["ТипВедомости"].isin(RETAKE_VEDOMOST_TYPES), "has_retake"] = 1

    perezachet_mask = df["ТипВедомости"] == "Перезачет"
    df.loc[perezachet_mask, "is_scholarship_blocking"] = 0
    df.loc[perezachet_mask, "has_retake"] = 0

    print(f"  -> {len(df):,} записей, {df['ЗачетнаяКнижка'].nunique():,} студентов")
    return df


def build_features(df):
    print(f"[2/5] Построение признаков...")

    df = df.copy()

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

    df["_prio"] = df["ВидКонтроля"].map(DEDUP_PRIORITY).fillna(DEDUP_PRIORITY_DEFAULT)
    df = df.sort_values(
        ["ЗачетнаяКнижка", "Дисциплина", "sem_num", "_prio", "grade_num"]
    )
    df = df.drop_duplicates(
        subset=["ЗачетнаяКнижка", "Дисциплина", "sem_num"], keep="first"
    )
    df = df.drop(columns=["_prio"])

    df = df.merge(disc_flags, on=disc_key, how="left")
    df["is_scholarship_blocking"] = df["disc_any_block"]
    df["has_retake"] = df["disc_any_retake"]
    df["grade_num"] = df[["grade_num", "disc_min_grade"]].min(axis=1)
    df = df.drop(columns=["disc_any_block", "disc_any_retake", "disc_min_grade"])

    key = ["ЗачетнаяКнижка", "sem_num"]

    is_pass_fail = (
        df["ВидКонтроля"].isin(PASS_FAIL_CONTROL_TYPES)
        | df["ИтоговаяОтметка"].isin(PASS_FAIL_GRADES)
    )
    df_graded = df[~is_pass_fail]

    overall = (
        df.groupby(key)
        .agg(
            n_subjects=("Дисциплина", "nunique"),
            n_blocks=("is_scholarship_blocking", "sum"),
            n_retakes=("has_retake", "sum"),
            any_block=("is_scholarship_blocking", "max"),
            any_retake=("has_retake", "max"),
        )
        .reset_index()
    )

    gpa_stats = (
        df_graded.groupby(key)
        .agg(
            gpa_overall=("grade_num", "mean"),
            min_grade=("grade_num", "min"),
            std_grade=("grade_num", "std"),
        )
        .reset_index()
    )
    overall = overall.merge(gpa_stats, on=key, how="left")

    graded_per_key = df_graded.groupby(key).size()
    share_5 = df_graded[df_graded["grade_num"] == 5].groupby(key).size() / graded_per_key
    share_3 = df_graded[df_graded["grade_num"] == 3].groupby(key).size() / graded_per_key

    total_per_key = df.groupby(key).size()
    zachet_per_key = df[is_pass_fail].groupby(key).size()
    share_zachet = zachet_per_key / total_per_key

    overall = overall.set_index(key)
    overall["share_5"] = share_5.reindex(overall.index).fillna(0)
    overall["share_3"] = share_3.reindex(overall.index).fillna(0)
    overall["share_zachet"] = share_zachet.reindex(overall.index).fillna(0)
    overall = overall.reset_index()

    overall["clean_record"] = 0
    overall.loc[
        (overall["any_block"] == 0) & (overall["any_retake"] == 0), "clean_record"
    ] = 1

    print(
        f"  -> {len(overall):,} (студент, семестр) строк, "
        f"{len(overall.columns)} колонок"
    )
    return overall


def build_pairs(features_df):
    """ class_B
    """
    print(f"[3/5] Построение обучающих пар (сем N -> сем N+1)...")

    features_df = features_df.sort_values(["ЗачетнаяКнижка", "sem_num"])
    pairs_list = []

    for sid, grp in features_df.groupby("ЗачетнаяКнижка"):
        grp = grp.sort_values("sem_num")
        sems = grp["sem_num"].values
        for i in range(len(sems) - 1):
            if sems[i + 1] - sems[i] == 1:
                current = grp[grp["sem_num"] == sems[i]].iloc[0].to_dict()
                next_clean = grp[grp["sem_num"] == sems[i + 1]].iloc[0]["clean_record"]
                current["target_clean_next_sem"] = int(next_clean)
                current["target_sem"] = int(sems[i + 1])
                current["had_clean_current_sem"] = int(current["clean_record"])
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
        for p, n in zip(
            pairs_df["had_clean_current_sem"], pairs_df["target_clean_next_sem"]
        )
    ]

    print(f"  -> {len(pairs_df):,} пар")
    print(
        f"  Чистый академический результат в след. семестре: "
        f"{pairs_df['target_clean_next_sem'].mean():.1%}"
    )
    for cls in [1, 2, 3, 4]:
        cnt = (pairs_df["class_B"] == cls).sum()
        print(
            f"    {cls} ({CLASS_B_LABELS[cls]:<20}): "
            f"{cnt:>5,} ({cnt / len(pairs_df) * 100:>5.1f}%)"
        )

    return pairs_df


EXCLUDE_COLS = {
    "ЗачетнаяКнижка",
    "target_clean_next_sem",
    "target_sem",
    "class_B",
    "clean_record",
}


def _prepare_features(pairs_df):
    feature_cols = [c for c in pairs_df.columns if c not in EXCLUDE_COLS]

    X = pairs_df[feature_cols].copy()
    y = pairs_df["target_clean_next_sem"].astype(int)

    if "std_grade" in X.columns:
        X["std_grade"] = X["std_grade"].fillna(0)

    return X, y, feature_cols


def _split_by_students(pairs_df, seed=42):
    """Random 80/20 split at the student level - same person never on both sides."""
    unique_students = pairs_df["ЗачетнаяКнижка"].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_students)
    split_idx = int(len(unique_students) * 0.8)
    train_students = set(unique_students[:split_idx])
    test_students = set(unique_students[split_idx:])

    train_mask = pairs_df["ЗачетнаяКнижка"].isin(train_students)
    test_mask = pairs_df["ЗачетнаяКнижка"].isin(test_students)
    return train_mask, test_mask, train_students, test_students


def train_model(pairs_df):
    print(f"[4/5] Обучение HistGradientBoosting...")

    X, y, feature_cols = _prepare_features(pairs_df)
    train_mask, test_mask, train_students, test_students = _split_by_students(pairs_df)

    X_train, y_train = X[train_mask].values, y[train_mask].values
    sample_weights = compute_sample_weight("balanced", y_train)

    print(f"  Train: {len(X_train):,} пар ({len(train_students):,} студентов)")
    print(f"  Test:  {test_mask.sum():,} пар ({len(test_students):,} студентов)")
    print(f"  Признаков: {len(feature_cols)}")
    print(f"  Feature list: {feature_cols}")

    model = HistGradientBoostingClassifier(
        max_iter=1000,
        learning_rate=0.05,
        max_depth=5,
        min_samples_leaf=10,
        l2_regularization=5.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=50,
        scoring="roc_auc",
        random_state=42,
        verbose=0,
    )

    model.fit(X_train, y_train, sample_weight=sample_weights)

    best_iter = int(model.n_iter_)
    print(f"  Итераций до остановки: {best_iter}")

    return model, best_iter, X, y, feature_cols, train_mask, test_mask, \
        train_students, test_students


def _run_baselines(pairs_df, test_mask):
    """A few dead-simple rules to compare the model against on the test set."""
    y_test = pairs_df.loc[test_mask, "target_clean_next_sem"].values
    prev_test = pairs_df.loc[test_mask, "had_clean_current_sem"].values
    gpa_test = pairs_df.loc[test_mask, "gpa_overall"].values
    any_block_test = pairs_df.loc[test_mask, "any_block"].values
    any_retake_test = pairs_df.loc[test_mask, "any_retake"].values

    baselines = {}

    majority_cls = int(pairs_df.loc[~test_mask, "target_clean_next_sem"].mode()[0])
    pred_majority = np.full_like(y_test, majority_cls)
    baselines["majority"] = {
        "pred": pred_majority,
        "desc": f"Always predict {majority_cls}",
    }

    baselines["persistence"] = {
        "pred": prev_test.copy(),
        "desc": "Predict next clean = current clean (had_clean_current_sem)",
    }

    pred_rule = ((any_block_test == 0) & (any_retake_test == 0)).astype(int)
    baselines["rule_no_blocking"] = {
        "pred": pred_rule,
        "desc": "No blocking grades & no retakes -> clean next sem",
    }

    for thr in [4.0, 4.2]:
        pred_gpa = ((~np.isnan(gpa_test)) & (gpa_test >= thr)).astype(int)
        baselines[f"gpa_{thr}"] = {
            "pred": pred_gpa,
            "desc": f"GPA >= {thr} -> clean next sem",
        }

    return baselines


def evaluate_and_save(
    model, best_iter, X, y, feature_cols,
    train_mask, test_mask, train_students, test_students,
    pairs_df, output_dir,
):
    print(f"\n[5/5] Оценка и сохранение результатов...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prev = pairs_df["had_clean_current_sem"].values
    true_b = pairs_df["class_B"].values

    X_test = X[test_mask].values
    y_test = y[test_mask].values
    prev_test = prev[test_mask]
    true_b_test = true_b[test_mask]

    y_pred = model.predict(X_test).astype(int)
    y_proba = model.predict_proba(X_test)[:, 1]

    pred_b = np.zeros(len(y_pred), dtype=int)
    had = prev_test == 1
    pred_b[had & (y_pred == 1)] = 1
    pred_b[had & (y_pred == 0)] = 2
    no = prev_test == 0
    pred_b[no & (y_pred == 1)] = 3
    pred_b[no & (y_pred == 0)] = 4

    baselines = _run_baselines(pairs_df, test_mask)

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

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n{'=' * 70}")
    print(f"  HistGradientBoosting ({len(y_test):,} пар, n_iter={best_iter})")
    print(f"{'=' * 70}")
    print(f"  Accuracy:   {acc:.1%}")
    print(f"  F1 (macro): {f1_macro:.3f}")
    print(f"  ROC-AUC:    {auc:.3f}")
    print()

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Не чисто (N+1)", "Чисто (N+1)"],
        digits=3,
    )
    print(report)

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Факт: не чисто", "Факт: чисто"],
        columns=["Пред: не чисто", "Пред: чисто"],
    )
    print("Confusion matrix (binary):")
    print(cm_df.to_string())

    print(f"\nVariant B - 4 класса:")
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

    for label, group_val in [
        ("Подгруппа «чистый результат сейчас» (had_clean=1)", 1),
        ("Подгруппа «не чистый сейчас» (had_clean=0)", 0),
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

        print(f"\n{'-' * 70}")
        print(f"  {label}")
        print(f"{'-' * 70}")
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

    print(f"\nFeature importances (permutation, n_repeats=5)...")
    perm = permutation_importance(
        model, X_test, y_test,
        scoring="roc_auc", n_repeats=5, random_state=42, n_jobs=-1,
    )
    raw_imp = np.clip(perm.importances_mean, 0, None)
    total = raw_imp.sum() if raw_imp.sum() > 0 else 1.0
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": raw_imp / total,
    })
    imp_df = imp_df.sort_values("importance", ascending=False)
    for _, row in imp_df.iterrows():
        print(f"  {row['feature']:<40} {row['importance']:>8.4f}")

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
            f"  {int(from_s)}->{int(from_s) + 1}: "
            f"acc={acc_s:.1%}, AUC={auc_s:.3f}, n={mask_s.sum()}"
        )

    pred_df = pd.DataFrame(
        {
            "ЗачетнаяКнижка": pairs_df.loc[test_mask, "ЗачетнаяКнижка"].values,
            "sem_num": pairs_df.loc[test_mask, "sem_num"].values,
            "target_sem": pairs_df.loc[test_mask, "target_sem"].values,
            "had_clean_current_sem": prev_test,
            "prob_clean_next_sem": np.round(y_proba, 4),
            "pred_clean_next_sem": y_pred,
            "actual_clean_next_sem": y_test,
            "pred_class_B": pred_b,
            "actual_class_B": true_b_test,
            "risk_score": np.round(1 - y_proba, 4),
        }
    )
    pred_csv = output_dir / "predictions.csv"
    pred_df.to_csv(pred_csv, index=False, encoding="utf-8-sig")
    print(f"\n  Сохранено: {pred_csv}")

    model_path = output_dir / "scholarship_model.joblib"
    joblib.dump(model, model_path)
    print(f"  Сохранено: {model_path}")

    summary_path = output_dir / "summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Scholarship Prediction - HistGradientBoosting (bachelor sem 2 -> 3)\n\n")
        f.write(f"**Дата:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        f.write("## Данные\n\n")
        f.write(
            f"- Train: {train_mask.sum():,} пар "
            f"({len(train_students):,} студентов)\n"
        )
        f.write(
            f"- Test: {test_mask.sum():,} пар "
            f"({len(test_students):,} студентов)\n"
        )
        f.write(f"- Признаков: {len(feature_cols)}\n")
        f.write(f"- Итераций до остановки: {best_iter}\n\n")

        f.write("## Baselines (test set)\n\n")
        f.write("| Baseline | Accuracy | F1 macro | Описание |\n|---|---|---|---|\n")
        for name, bl in baselines.items():
            bl_acc = accuracy_score(y_test, bl["pred"])
            bl_f1 = f1_score(y_test, bl["pred"], average="macro")
            f.write(
                f"| {name} | {bl_acc:.1%} | {bl_f1:.3f} | {bl['desc']} |\n"
            )

        f.write(f"\n## Метрики HistGradientBoosting\n\n")
        f.write("| Метрика | Значение |\n|---|---|\n")
        f.write(f"| Accuracy | {acc:.1%} |\n")
        f.write(f"| F1 (macro) | {f1_macro:.3f} |\n")
        f.write(f"| ROC-AUC | {auc:.3f} |\n\n")
        f.write(f"## Classification report\n\n```\n{report}```\n\n")
        f.write(
            f"## Confusion matrix (binary)\n\n"
            f"```\n{cm_df.to_string()}\n```\n\n"
        )
        f.write(
            f"## Variant B - 4 класса\n\n"
            f"```\n{cm_b_df.to_string()}\n```\n\n"
        )
        f.write("## Per-class accuracy\n\n")
        for cls in [1, 2, 3, 4]:
            if cls in class_b_accuracies:
                f.write(
                    f"- **{CLASS_B_LABELS[cls]}**: "
                    f"{class_b_accuracies[cls]:.1%}\n"
                )

        f.write(f"\n## Feature importance (permutation, ROC-AUC)\n\n")
        for _, row in imp_df.iterrows():
            f.write(f"- `{row['feature']}`: {row['importance']:.4f}\n")

    print(f"  Сохранено: {summary_path}")

    card = {
        "model_type": "HistGradientBoostingClassifier",
        "slice": "bachelor sem 2 -> 3 (output_bachelors_both_sem2_3_ochnaya)",
        "target": (
            "Clean academic record in next semester (no blocking grades, no "
            "retakes). Under Russian academic rules, a clean record in sem "
            "N+1 entitles the student to the stipend paid in N+2."
        ),
        "features": feature_cols,
        "n_features": len(feature_cols),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "n_iter": best_iter,
        "metrics": {
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
    card_path = output_dir / "model_card.json"
    with open(card_path, "w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=False, indent=2)
    print(f"  Сохранено: {card_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Scholarship Prediction - HistGradientBoosting on the bachelor "
            "sem 2 -> 3 slice"
        )
    )
    parser.add_argument("--data", type=str, default=None, help="Путь к xlsx-файлу")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/histgb_bachelors_sem2_3",
        help="Папка для результатов (default: outputs/histgb_bachelors_sem2_3/)",
    )
    args = parser.parse_args()

    print(f"{'=' * 70}")
    print(f"  Scholarship Prediction - Bachelor's sem 2 -> sem 3 slice")
    print(f"{'=' * 70}\n")

    df = load_data(args.data)
    features_df = build_features(df)
    pairs_df = build_pairs(features_df)

    model, best_iter, X, y, feature_cols, train_mask, test_mask, \
        train_students, test_students = train_model(pairs_df)

    evaluate_and_save(
        model, best_iter, X, y, feature_cols,
        train_mask, test_mask, train_students, test_students,
        pairs_df, Path(args.output),
    )

    print(f"\n{'=' * 70}")
    print(f"  Готово! Результаты в папке: {Path(args.output).resolve()}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
