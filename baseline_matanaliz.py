"""
Baseline: предсказание оценки по Математическому анализу (семестр 1 → 2).

Модель смотрит ТОЛЬКО на данные по Мат анализу — никаких других предметов.
Признаки: оценка MA в сем.1, вид контроля, пересдача, комиссия, учебный план.

Запуск:
    python baseline_matanaliz.py
    # или с явным путём:
    python baseline_matanaliz.py --data path/to/file.xlsx
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
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------
TARGET_SUBJECT = "Математический анализ"
GRADE_MAP = {
    "Отлично": 5, "Хорошо": 4, "Удовлетворительно": 3,
    "Неудовлетворительно": 2, "Зачтено": 4, "Не зачтено": 2, "Неявка": 2,
}
TARGET_CLASSES = ["5", "4", "3", "fail"]
SEMESTER_MAP = {
    "Первый семестр": 1, "Второй семестр": 2, "Третий семестр": 3,
    "Четвертый семестр": 4, "Пятый семестр": 5, "Шестой семестр": 6,
    "Седьмой семестр": 7, "Восьмой семестр": 8, "Девятый семестр": 9,
    "Десятый семестр": 10,
}

DATA_FILE_PATTERNS = [
    "ГОСТ Р 70946-2023-Приложение-8_sorted.xlsx",
]


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------
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
        f"There is no file in the {directory}.\n"
    )


def grade_to_target(g: str) -> str:
    if g == "Отлично":
        return "5"
    if g == "Хорошо":
        return "4"
    if g == "Удовлетворительно":
        return "3"
    return "fail"


# ---------------------------------------------------------------------------
# Загрузка и подготовка
# ---------------------------------------------------------------------------
def load_data(path: str | Path | None = None) -> pd.DataFrame:
    if path is None:
        path = find_data_file(Path(__file__).parent / "data")
    path = Path(path)
    print(f"Загрузка данных из {path.name}...")
    df = pd.read_excel(path, sheet_name="Sheet1")
    df["sem_num"] = df["ПериодКонтроля"].map(SEMESTER_MAP)
    df = df[df["sem_num"].notna()].copy()
    df["sem_num"] = df["sem_num"].astype(int)
    df = df[(df["ФормаОбучения"] == "Очная") & (df["ТипВедомости"] == "Основная")]
    print(f"  → {len(df):,} записей, {df['ЗачетнаяКнижка'].nunique():,} студентов")
    return df


def prepare_ma_data(df: pd.DataFrame) -> pd.DataFrame:
    """Фильтрует только Мат анализ, дедуплицирует, мержит сем.1 и сем.2."""

    ma = df[df["Дисциплина"] == TARGET_SUBJECT].copy()
    ma["grade_num"] = ma["ИтоговаяОтметка"].map(GRADE_MAP)
    print(f"  → {len(ma):,} записей по '{TARGET_SUBJECT}'")

    # Дедупликация: одна запись на (студент, семестр).
    # Приоритет: Экзамен > Зачет с оценкой. При равном — худшая оценка.
    priority = {"Экзамен": 0, "Зачет с оценкой": 1}
    ma["_prio"] = ma["ВидКонтроля"].map(priority).fillna(2)
    ma = ma.sort_values(["ЗачетнаяКнижка", "sem_num", "_prio", "grade_num"])
    ma = ma.drop_duplicates(subset=["ЗачетнаяКнижка", "sem_num"], keep="first")
    ma = ma.drop(columns=["_prio"])

    # Семестр 1 и 2 отдельно, inner join = только студенты с обоими
    sem1 = ma[ma["sem_num"] == 1].copy()
    sem2 = ma[ma["sem_num"] == 2][["ЗачетнаяКнижка", "ИтоговаяОтметка"]].copy()

    merged = sem1.merge(sem2, on="ЗачетнаяКнижка", how="inner",
                        suffixes=("_s1", "_s2"))
    merged["target"] = merged["ИтоговаяОтметка_s2"].apply(grade_to_target)
    print(f"  → {len(merged)} студентов с MA в обоих семестрах")

    return merged


# ---------------------------------------------------------------------------
# Обучение и оценка
# ---------------------------------------------------------------------------
def train_and_evaluate(merged: pd.DataFrame, output_dir: Path):

    # Признаки: ТОЛЬКО из MA семестра 1
    X = pd.DataFrame({
        "ma_grade_sem1": merged["grade_num"].values,
        "ma_control_type": merged["ВидКонтроля"].values,
        "ma_retake": (merged["Пересдача"].values > 0).astype(int),
        "ma_commission": (merged["Комиссия"].values > 0).astype(int),
        "plan": merged["УчебныйПлан"].astype(str).values,
    })
    y = merged["target"].reset_index(drop=True)
    students = merged["ЗачетнаяКнижка"].values

    print(f"\nПризнаки: {list(X.columns)}")
    print(f"\nЦелевая (оценка MA сем.2):")
    print(y.value_counts().sort_index().to_string())

    # Train/test split
    cat_idx = [1, 4]  # ma_control_type, plan
    X_train, X_test, y_train, y_test, _, students_test = train_test_split(
        X, y, students, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)} / Test: {len(X_test)}")

    # CatBoost
    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.05,
        depth=4,
        loss_function="MultiClass",
        cat_features=cat_idx,
        random_seed=42,
        verbose=0,
        early_stopping_rounds=100,
        use_best_model=True,
    )

    print("Обучение CatBoost...")
    model.fit(
        Pool(X_train, y_train, cat_features=cat_idx),
        eval_set=Pool(X_test, y_test, cat_features=cat_idx),
    )
    best_iter = model.get_best_iteration()
    print(f"  Лучшая итерация: {best_iter}")

    # Предсказания
    y_pred = model.predict(X_test).flatten()
    y_proba = model.predict_proba(X_test)
    classes = model.classes_

    # Метрики
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    grade_order = {"5": 5, "4": 4, "3": 3, "fail": 1}
    y_t_ord = np.array([grade_order[g] for g in y_test])
    y_p_ord = np.array([grade_order[g] for g in y_pred])
    within_1 = np.mean(np.abs(y_t_ord - y_p_ord) <= 1)

    print(f"\n{'='*60}")
    print(f"  РЕЗУЛЬТАТЫ ({len(X_test)} студентов)")
    print(f"{'='*60}")
    print(f"  Accuracy:       {acc:.1%}")
    print(f"  Macro F1:       {macro_f1:.3f}")
    print(f"  Within-1-grade: {within_1:.1%}")
    print()

    report = classification_report(
        y_test, y_pred, labels=TARGET_CLASSES, digits=3, zero_division=0
    )
    print(report)

    cm = confusion_matrix(y_test, y_pred, labels=TARGET_CLASSES)
    cm_df = pd.DataFrame(
        cm,
        index=[f"факт_{c}" for c in TARGET_CLASSES],
        columns=[f"пред_{c}" for c in TARGET_CLASSES],
    )
    print("Confusion matrix:")
    print(cm_df.to_string())

    # Важность признаков
    print(f"\nВажность признаков:")
    importance = model.get_feature_importance()
    imp_pairs = sorted(zip(X.columns, importance), key=lambda x: -x[1])
    for feat, imp in imp_pairs:
        print(f"  {feat:<25} {imp:.1f}")

    # Примеры
    print(f"\nПримеры (10 случайных):")
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(X_test), size=min(10, len(X_test)), replace=False)
    for i in sample_idx:
        sid = students_test[i]
        g1 = X_test.iloc[i]["ma_grade_sem1"]
        rt = "пересд" if X_test.iloc[i]["ma_retake"] else ""
        t = y_test.iloc[i]
        p = y_pred[i]
        mark = "✓" if t == p else "✗"
        print(f"  Студент {sid}: MA сем.1={g1:.0f} {rt:<7} → пред={p}, факт={t} {mark}")

    # --- Сохранение ---
    output_dir.mkdir(parents=True, exist_ok=True)

    # predictions.csv
    pred_df = pd.DataFrame({
        "ЗачетнаяКнижка": students_test,
        "ma_grade_sem1": X_test["ma_grade_sem1"].values,
        "ma_retake": X_test["ma_retake"].values,
        "plan": X_test["plan"].values,
        "predicted_grade": y_pred,
        "actual_grade": y_test.values,
        "correct": (y_pred == y_test.values).astype(int),
    })
    for ci, cls_name in enumerate(classes):
        pred_df[f"prob_{cls_name}"] = y_proba[:, ci]
    pred_csv = output_dir / "predictions.csv"
    pred_df.to_csv(pred_csv, index=False, encoding="utf-8-sig")
    print(f"\n  Сохранено: {pred_csv}")

    # model.cbm
    model_path = output_dir / "model.cbm"
    model.save_model(str(model_path))
    print(f"  Сохранено: {model_path}")

    # summary.md
    summary_path = output_dir / "summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"# Baseline: {TARGET_SUBJECT} (только этот предмет)\n\n")
        f.write(f"**Дата:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"**Горизонт:** семестр 1 → 2\n\n")
        f.write(f"**Модель:** CatBoostClassifier, {best_iter} итераций\n\n")
        f.write(f"## Признаки (5 штук)\n\n")
        f.write(f"Все признаки — только из записей по Мат анализу семестра 1:\n\n")
        f.write(f"1. `ma_grade_sem1` — оценка (2–5)\n")
        f.write(f"2. `ma_control_type` — вид контроля (Экзамен / Зачет с оценкой)\n")
        f.write(f"3. `ma_retake` — была ли пересдача (0/1)\n")
        f.write(f"4. `ma_commission` — была ли комиссия (0/1)\n")
        f.write(f"5. `plan` — учебный план (категориальный)\n\n")
        f.write(f"## Данные\n\n")
        f.write(f"- Студентов с MA в обоих семестрах: {len(merged)}\n")
        f.write(f"- Train: {len(X_train)} / Test: {len(X_test)}\n\n")
        f.write(f"## Метрики\n\n")
        f.write(f"| Метрика | Значение |\n|---|---|\n")
        f.write(f"| Accuracy | {acc:.1%} |\n")
        f.write(f"| Macro F1 | {macro_f1:.3f} |\n")
        f.write(f"| Within-1-grade | {within_1:.1%} |\n\n")
        f.write(f"## Per-class report\n\n```\n{report}```\n\n")
        f.write(f"## Confusion matrix\n\n```\n{cm_df.to_string()}\n```\n\n")
        f.write(f"## Важность признаков\n\n")
        for feat, imp in imp_pairs:
            f.write(f"- `{feat}`: {imp:.1f}\n")
    print(f"  Сохранено: {summary_path}")

    # model_card.json
    card = {
        "model_type": "CatBoostClassifier",
        "target": f"grade in {TARGET_SUBJECT}, semester 2",
        "features_used": "ONLY from Математический анализ semester 1",
        "classes": TARGET_CLASSES,
        "features": list(X.columns),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "best_iteration": best_iter,
        "metrics": {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "within_1_grade": round(within_1, 4),
        },
        "random_seed": 42,
        "date": datetime.now().isoformat(),
    }
    card_path = output_dir / "model_card.json"
    with open(card_path, "w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=False, indent=2)
    print(f"  Сохранено: {card_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Baseline: MA grade prediction (MA data only)"
    )
    parser.add_argument("--data", type=str, default=None,
                        help="Путь к xlsx-файлу с данными")
    parser.add_argument("--output", type=str, default="outputs/baseline",
                        help="Папка для результатов (default: outputs/baseline/)")
    args = parser.parse_args()

    df = load_data(args.data)
    merged = prepare_ma_data(df)

    if len(merged) < 50:
        print("ОШИБКА: слишком мало данных для обучения.")
        return

    train_and_evaluate(merged, Path(args.output))


if __name__ == "__main__":
    main()
