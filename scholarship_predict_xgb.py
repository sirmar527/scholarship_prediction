"""
Scholarship Prediction - XGBoost Pipeline.

Predicts the probability that a student will maintain a CLEAN ACADEMIC
RECORD in the next semester (no blocking grades, no retakes), based on
their performance in the current semester.

Note on terminology: under standard Russian academic rules the scholarship
a student receives IN semester N+1 is determined by their performance IN
semester N - so a clean record in N+1 entitles the student to the stipend
paid in N+2. This pipeline is therefore a ONE-SEMESTER-AHEAD early-warning
forecast: features come from sem N, the target is "clean record in N+1"
(which would entitle the student to a stipend in N+2).

Uses XGBoost with student-level train/test split and early stopping.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight

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
# Pass/fail subjects do NOT contribute to GPA. They are identified by
# ВидКонтроля == "Зачет" (universally pass/fail in Russian higher ed).
# PASS_FAIL_GRADES is kept as a defensive secondary check in case future
# data has anomalous control-type / grade combinations.
PASS_FAIL_CONTROL_TYPES = {"Зачет"}
PASS_FAIL_GRADES = {"Зачтено", "Не зачтено"}

# Dedup priority: when one discipline has multiple component rows
# (e.g. an Экзамен + a Курсовая работа), the surviving row is the one
# with the lowest priority value. Graded coursework outranks pure pass/fail
# Зачет; this matters because the surviving row's text label feeds into
# share_5 / share_3 computations.
DEDUP_PRIORITY = {
    "Экзамен": 0,
    "Зачет с оценкой": 1,
    "Курсовая работа": 2,
    "Курсовой проект": 2,
    "ГАК": 2,
    "Защита ВКР": 2,
    "Зачет": 3,
}
DEDUP_PRIORITY_DEFAULT = 4  # for any ВидКонтроля not listed above

# Types of ведомость we keep alongside Основная:
#   - Перезачет: transferred credits from prior education; treated as
#     passed prior work - counted in subject load and GPA but NEVER
#     scholarship-blocking and NEVER a retake event.
#   - Пересдача / Пересдача с комиссией: in this dataset, retake events
#     are usually duplicated as separate rows alongside the Основная row
#     they update. We drop those duplicates and KEEP only orphan retake
#     rows (no Основная partner for that student/sem/discipline).
KEPT_VEDOMOST_TYPES = {"Основная", "Перезачет", "Пересдача", "Пересдача с комиссией"}
RETAKE_VEDOMOST_TYPES = {"Пересдача", "Пересдача с комиссией"}

CLASS_B_LABELS = {
    1: "Чисто→чисто",
    2: "Чисто→провал",
    3: "Провал→чисто",
    4: "Провал→провал",
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
        f"Положите файл в data/ или укажите --data путь/к/файлу.xlsx"
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

    # Keep Основная, Перезачет, and retake-type rows (filtered further below).
    df = df[df["ТипВедомости"].isin(KEPT_VEDOMOST_TYPES)].copy()

    # Drop retake rows that DUPLICATE an Основная row for the same
    # (student, sem, discipline). In this dataset such duplicates carry the
    # same final mark and score columns; keeping them would double-count.
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
    # Score columns (Пересдача, Комиссия, Экзамен): 100-point scale.
    # For retake columns, value > 0 means the retake was taken and scored;
    # value == 0 is AMBIGUOUS - either no retake was scheduled, OR a retake
    # was scheduled but the student did not show up. The no-show case is
    # identified by ИтоговаяОтметка == "Неявка" (in this dataset, Неявка as
    # a final mark only occurs at Пересдача/Комиссия stage, never at the
    # original exam). Therefore retake events must be detected as:
    df["has_retake"] = (
        (df["Пересдача"] > 0)
        | (df["Комиссия"] > 0)
        | (df["ИтоговаяОтметка"] == "Неявка")
    ).astype(int)
    # Orphan retake-type rows are themselves a retake event by construction.
    df.loc[df["ТипВедомости"].isin(RETAKE_VEDOMOST_TYPES), "has_retake"] = 1

    # Перезачет = transferred prior credits. By Option 1 convention,
    # they NEVER block scholarship and NEVER count as a retake event,
    # but their grades DO contribute to GPA and they count in n_subjects.
    perezachet_mask = df["ТипВедомости"] == "Перезачет"
    df.loc[perezachet_mask, "is_scholarship_blocking"] = 0
    df.loc[perezachet_mask, "has_retake"] = 0

    print(f"  → {len(df):,} записей, {df['ЗачетнаяКнижка'].nunique():,} студентов")
    return df


# =====================================================================
# 2. FEATURE ENGINEERING
# =====================================================================
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

    # ── Split: graded subjects vs pass/fail ──
    # Classify by ВидКонтроля (universal rule); PASS_FAIL_GRADES kept as a
    # defensive secondary check.
    is_pass_fail = (
        df["ВидКонтроля"].isin(PASS_FAIL_CONTROL_TYPES)
        | df["ИтоговаяОтметка"].isin(PASS_FAIL_GRADES)
    )
    df_graded = df[~is_pass_fail]

    # ── Aggregate over ALL subjects (counts, blocking, retakes) ──
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

    # ── GPA stats from graded subjects only (excluding pass/fail Зачет) ──
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

    # ── share_5, share_3: computed from the numeric grade after dedup ──
    # (NOT from the surviving row's text label - those can disagree because
    # disc_min_grade overwrites grade_num while ИтоговаяОтметка is left
    # alone.)
    graded_per_key = df_graded.groupby(key).size()
    share_5 = df_graded[df_graded["grade_num"] == 5].groupby(key).size() / graded_per_key
    share_3 = df_graded[df_graded["grade_num"] == 3].groupby(key).size() / graded_per_key

    # ── share_zachet: fraction of pass/fail subjects in total load ──
    total_per_key = df.groupby(key).size()
    zachet_per_key = df[is_pass_fail].groupby(key).size()
    share_zachet = zachet_per_key / total_per_key

    overall = overall.set_index(key)
    overall["share_5"] = share_5.reindex(overall.index).fillna(0)
    overall["share_3"] = share_3.reindex(overall.index).fillna(0)
    overall["share_zachet"] = share_zachet.reindex(overall.index).fillna(0)
    overall = overall.reset_index()

    # clean_record = 1 iff the student had no blocking grades and no retakes
    # in this semester. Under standard Russian academic rules, a clean record
    # in semester N entitles the student to the stipend paid in semester N+1.
    # NOTE: no sem_1 override - first-semester students with blocking grades
    # genuinely did not have a clean record.
    overall["clean_record"] = 0
    overall.loc[
        (overall["any_block"] == 0) & (overall["any_retake"] == 0), "clean_record"
    ] = 1

    print(
        f"  → {len(overall):,} (студент, семестр) строк, "
        f"{len(overall.columns)} колонок"
    )
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
                next_clean = grp[grp["sem_num"] == sems[i + 1]].iloc[0]["clean_record"]
                current["target_clean_next_sem"] = int(next_clean)
                current["target_sem"] = int(sems[i + 1])
                current["had_clean_current_sem"] = int(current["clean_record"])
                pairs_list.append(current)

    pairs_df = pd.DataFrame(pairs_list)

    def variant_b(p, n):
        # p = had clean record in current sem, n = clean record in next sem
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

    print(f"  → {len(pairs_df):,} пар")
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


# =====================================================================
# 4. PREPARE, SPLIT, TRAIN
# =====================================================================
EXCLUDE_COLS = {
    "ЗачетнаяКнижка",
    "target_clean_next_sem",
    "target_sem",
    "class_B",
    "clean_record",
}


def _prepare_features(pairs_df):
    """Return feature matrix, target vector, and feature column names.

    NaN values are LEFT IN PLACE for numeric features: XGBoost handles them
    natively by learning the optimal split direction at each node. Do not
    fillna(0) for GPA / min_grade - 0 is not a valid value on the 2-5 grade
    scale, and student-semesters that only contain pass/fail Зачет subjects
    legitimately have NaN here.
    """
    feature_cols = [c for c in pairs_df.columns if c not in EXCLUDE_COLS]

    X = pairs_df[feature_cols].copy()
    y = pairs_df["target_clean_next_sem"].astype(int)

    # std_grade is genuinely undefined for single-graded-subject semesters
    # (n=1 → std is mathematically undefined). Filling with 0 is fine here:
    # a single value has zero variance.
    if "std_grade" in X.columns:
        X["std_grade"] = X["std_grade"].fillna(0)

    return X, y, feature_cols


def _split_by_students(pairs_df, seed=42):
    """80/20 split by student ID - same students never appear in both sets."""
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
    """Train XGBoost with early stopping on a student-level split."""
    print(f"[4/5] Обучение XGBoost...")

    X, y, feature_cols = _prepare_features(pairs_df)
    train_mask, test_mask, train_students, test_students = _split_by_students(pairs_df)

    # ── Carve validation set from train for early stopping ──
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

    sample_weights = compute_sample_weight("balanced", y_fit)

    print(f"  Fit:  {len(X_fit):,} пар ({len(fit_students):,} студентов)")
    print(f"  Val:  {len(X_val):,} пар ({len(val_students):,} студентов)")
    print(f"  Test: {test_mask.sum():,} пар ({len(test_students):,} студентов)")
    print(f"  Признаков: {len(feature_cols)}")
    print(f"  Feature list: {feature_cols}")

    model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        random_state=42,
        early_stopping_rounds=50,
        eval_metric="auc",
        verbosity=0,
    )

    model.fit(
        X_fit,
        y_fit,
        eval_set=[(X_val, y_val)],
        sample_weight=sample_weights,
        verbose=False,
    )

    best_iter = model.best_iteration
    print(f"  Лучшая итерация: {best_iter}")

    return model, best_iter, X, y, feature_cols, train_mask, test_mask, \
        train_students, test_students


# =====================================================================
# 5. BASELINES
# =====================================================================
def _run_baselines(pairs_df, test_mask):
    """Compute simple baselines on the test set for comparison."""
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
        "desc": "No blocking grades & no retakes → clean next sem",
    }

    for thr in [4.0, 4.2]:
        # gpa NaN means "only-pass/fail semester"; treat as not meeting bar.
        pred_gpa = ((~np.isnan(gpa_test)) & (gpa_test >= thr)).astype(int)
        baselines[f"gpa_{thr}"] = {
            "pred": pred_gpa,
            "desc": f"GPA ≥ {thr} → clean next sem",
        }

    return baselines


# =====================================================================
# 6. EVALUATE AND SAVE
# =====================================================================
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

    # ── Predictions ──
    X_test = X[test_mask].values
    y_test = y[test_mask].values
    prev_test = prev[test_mask]
    true_b_test = true_b[test_mask]

    y_pred = model.predict(X_test).astype(int)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Derive 4-class predictions
    pred_b = np.zeros(len(y_pred), dtype=int)
    had = prev_test == 1
    pred_b[had & (y_pred == 1)] = 1
    pred_b[had & (y_pred == 0)] = 2
    no = prev_test == 0
    pred_b[no & (y_pred == 1)] = 3
    pred_b[no & (y_pred == 0)] = 4

    # ── Baselines ──
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

    # ── Model metrics ──
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n{'=' * 70}")
    print(f"  XGBOOST ({len(y_test):,} пар, best_iteration={best_iter})")
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

    # ── 4-class metrics ──
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

    # ── Per-subgroup report ──
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
    imp = model.feature_importances_
    imp_df = pd.DataFrame({"feature": feature_cols, "importance": imp})
    imp_df = imp_df.sort_values("importance", ascending=False)
    print(f"\nFeature importances:")
    for _, row in imp_df.iterrows():
        print(f"  {row['feature']:<40} {row['importance']:>8.4f}")

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
    # NOTE: prob_clean_next_sem is the model's probability that the student
    # will have a clean academic record in sem N+1 (which, by Russian rules,
    # entitles them to the stipend paid in N+2). The 'risk_score' column is
    # the complement (probability of failing N+1). For students who already
    # had a clean current sem (had_clean=1), this is a "loss" risk; for those
    # who didn't, it is a "won't recover" probability.
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

    # ── Save model ──
    model_path = output_dir / "scholarship_model.json"
    model.save_model(str(model_path))
    print(f"  Сохранено: {model_path}")

    # ── Save summary ──
    summary_path = output_dir / "summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Scholarship Prediction - XGBoost\n\n")
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
        f.write(f"- Лучшая итерация: {best_iter}\n\n")

        f.write("## Baselines (test set)\n\n")
        f.write("| Baseline | Accuracy | F1 macro | Описание |\n|---|---|---|---|\n")
        for name, bl in baselines.items():
            bl_acc = accuracy_score(y_test, bl["pred"])
            bl_f1 = f1_score(y_test, bl["pred"], average="macro")
            f.write(
                f"| {name} | {bl_acc:.1%} | {bl_f1:.3f} | {bl['desc']} |\n"
            )

        f.write(f"\n## Метрики XGBoost\n\n")
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

        f.write(f"\n## Feature importance\n\n")
        for _, row in imp_df.iterrows():
            f.write(f"- `{row['feature']}`: {row['importance']:.4f}\n")

    print(f"  Сохранено: {summary_path}")

    # ── Save model card ──
    card = {
        "model_type": "XGBClassifier",
        "target": (
            "Clean academic record in next semester (no blocking grades, no "
            "retakes). Under Russian academic rules, a clean record in sem "
            "N+1 entitles the student to the stipend paid in N+2."
        ),
        "features": feature_cols,
        "n_features": len(feature_cols),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "best_iteration": best_iter,
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

    # ── Save human-readable report ──
    _save_human_report(
        output_dir=output_dir,
        acc=acc, f1_macro=f1_macro, auc=auc,
        baselines=baselines, y_test=y_test,
        train_mask=train_mask, test_mask=test_mask,
        train_students=train_students, test_students=test_students,
        imp_df=imp_df,
        class_b_accuracies=class_b_accuracies,
        pred_b=pred_b, true_b_test=true_b_test,
        y_pred=y_pred, y_proba=y_proba,
    )


# =====================================================================
# HUMAN-READABLE REPORT
# =====================================================================
FEATURE_DESCRIPTIONS = {
    "sem_num": "номер текущего семестра",
    "n_subjects": "количество дисциплин в семестре",
    "n_blocks": "количество предметов с блокирующими оценками",
    "n_retakes": "количество пересдач",
    "any_block": "наличие хотя бы одной блокирующей оценки",
    "any_retake": "наличие хотя бы одной пересдачи",
    "gpa_overall": "средний балл за семестр",
    "min_grade": "минимальная полученная оценка",
    "std_grade": "разброс (стабильность) оценок",
    "share_5": "доля «отлично» среди оцениваемых дисциплин",
    "share_3": "доля «удовлетворительно» среди оцениваемых дисциплин",
    "share_zachet": "доля зачётных (без оценки) дисциплин в нагрузке",
    "had_clean_current_sem": "был ли чистый академический результат в текущем семестре",
}


def _save_human_report(
    output_dir,
    acc, f1_macro, auc,
    baselines, y_test,
    train_mask, test_mask,
    train_students, test_students,
    imp_df,
    class_b_accuracies,
    pred_b, true_b_test,
    y_pred, y_proba,
):
    """Save a plain-language report aimed at non-technical stakeholders."""

    n_test = len(y_test)
    n_correct = int((y_test == y_pred).sum())
    n_wrong = n_test - n_correct

    # Scholarship-loss detection stats
    actual_lost = (true_b_test == 2).sum()
    predicted_lost = (pred_b == 2).sum()
    caught_lost = int(((pred_b == 2) & (true_b_test == 2)).sum())

    # Scholarship-gain detection stats
    actual_gained = (true_b_test == 3).sum()
    predicted_gained = (pred_b == 3).sum()
    caught_gained = int(((pred_b == 3) & (true_b_test == 3)).sum())

    # Best baseline
    best_bl_name, best_bl_acc = None, 0.0
    for name, bl in baselines.items():
        bl_acc = accuracy_score(y_test, bl["pred"])
        if bl_acc > best_bl_acc:
            best_bl_acc = bl_acc
            best_bl_name = name

    # Top features
    top_features = imp_df.head(5)

    # Risk tiers
    high_risk = (y_proba < 0.3).sum()
    medium_risk = ((y_proba >= 0.3) & (y_proba < 0.6)).sum()
    low_risk = (y_proba >= 0.6).sum()

    lines = []
    w = lines.append

    w("# Прогноз академической успеваемости (стипендиальный риск)\n")
    w(f"**Дата формирования:** {datetime.now().strftime('%d.%m.%Y %H:%M')}\n")

    w("> **Что предсказывает модель.** Модель оценивает вероятность того, ")
    w("> что студент **сохранит чистый академический результат в следующем ")
    w("> семестре** (без блокирующих оценок и пересдач). По правилам ")
    w("> большинства российских вузов чистый результат в семестре N+1 даёт ")
    w("> право на стипендию, выплачиваемую в семестре N+2 (стипендия в N+1 ")
    w("> уже определяется результатами текущего семестра N и не нуждается ")
    w("> в прогнозе). Поэтому модель - это раннее предупреждение на ")
    w("> один семестр вперёд.\n")

    # ── Section 2: Data scope ──
    w("## На каких данных обучена модель\n")
    w(
        f"Модель обучена на данных **{train_mask.sum():,}** переходов между семестрами "
        f"(**{len(train_students):,}** студентов). "
        f"Для проверки качества использовались данные **{test_mask.sum():,}** переходов "
        f"(**{len(test_students):,}** студентов) тестовые. "
    )

    # ── Section 3: How well does it work? ──
    w("## Насколько точна модель\n")
    w(
        f"На проверочной выборке из {n_test:,} переходов модель дала правильный ответ "
        f"в **{n_correct:,}** случаях и ошиблась в **{n_wrong:,}** "
        f"(точность **{acc:.1%}**).\n"
    )
    w(
        f"Для сравнения: лучший простой метод («{best_bl_name}») даёт точность "
        f"**{best_bl_acc:.1%}**. Модель превосходит простые правила "
        f"на **{(acc - best_bl_acc) * 100:.1f}** п.п.\n"
    )
    w(
        f"Дополнительные метрики качества: ROC-AUC = **{auc:.3f}** "
        f"F1 (macro) = **{f1_macro:.3f}**.\n"
    )

    # ── Section 4: Transition analysis ──
    w("## Как модель работает по категориям студентов\n")
    w(
        "Каждый студент относится к одной из четырёх категорий по тому, был ли "
        "у него чистый академический результат в текущем семестре и будет ли в следующем:\n"
    )
    for cls in [1, 2, 3, 4]:
        if cls in class_b_accuracies:
            w(
                f"- {cls}. **{CLASS_B_LABELS[cls]}** - модель угадывает "
                f"**{class_b_accuracies[cls]:.0%}** таких случаев"
            )
    w("")

    if actual_lost > 0:
        w(
            f"**Срыв успеваемости (потеря права на стипендию N+2):** из {actual_lost:,} "
            f"студентов, у которых был чистый текущий результат, но следующий семестр "
            f"оказался неуспешным, модель верно предупредила о {caught_lost:,} "
            f"({caught_lost / actual_lost:.0%}). "
        )
    if actual_gained > 0:
        w(
            f"**Восстановление успеваемости (получение права на стипендию N+2):** "
            f"из {actual_gained:,} студентов, у которых текущий семестр был не чистый, "
            f"но следующий стал чистым, модель верно предсказала {caught_gained:,} "
            f"({caught_gained / actual_gained:.0%}).\n"
        )

    # ── Section 5: Risk tiers ──
    w("## Распределение по уровням риска\n")
    w(
        "Модель присваивает каждому переходу вероятность того, что следующий семестр "
        "окажется чистым (от 0% до 100%). На основе этой вероятности студенты "
        "делятся на группы. **Важно:** интерпретация риска зависит от того, "
        "был ли чистым текущий семестр - для тех, кто и сейчас не имеет права "
        "на стипендию (had_clean=0), низкая вероятность означает «не восстановится», "
        "а не «потеряет».\n"
    )
    w(
        f"- **Высокий риск** (вероятность < 30%): **{high_risk:,}** студентов "
        f"({high_risk / n_test:.0%}) - рекомендуется обратить внимание"
    )
    w(
        f"- **Средний риск** (30%-60%): **{medium_risk:,}** студентов "
        f"({medium_risk / n_test:.0%}) - стоит мониторить"
    )
    w(
        f"- **Низкий риск** (> 60%): **{low_risk:,}** студентов "
        f"({low_risk / n_test:.0%}) - ситуация стабильная"
    )
    w("")

    # ── Section 6: What matters most ──
    w("## Что больше всего влияет на прогноз\n")
    w(
        "Ниже перечислены пять факторов, которые модель считает наиболее "
        "важными при принятии решения:\n"
    )
    for rank, (_, row) in enumerate(top_features.iterrows(), 1):
        feat = row["feature"]
        desc = FEATURE_DESCRIPTIONS.get(feat, feat)
        pct = row["importance"] * 100
        w(f"{rank}. **{desc}** - вес {pct:.1f}%")
    w("")

    report_path = output_dir / "report_human.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  Сохранено: {report_path}")
    return report_path


# =====================================================================
# MAIN
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Scholarship Prediction - XGBoost Pipeline"
    )
    parser.add_argument("--data", type=str, default=None, help="Путь к xlsx-файлу")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/xgb",
        help="Папка для результатов (default: outputs/xgb/)",
    )
    args = parser.parse_args()

    print(f"{'=' * 70}")
    print(f"  Scholarship Prediction - XGBoost Pipeline")
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
