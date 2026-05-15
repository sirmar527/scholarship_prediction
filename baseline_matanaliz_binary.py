"""
Бинарный baseline: Мат анализ, семестр 1 → 2.
Проходной (4, 5, Зачтено) vs Непроходной (3, 2, Неуд, НеЗач, Неявка).

Модель смотрит ТОЛЬКО на данные по Мат анализу — никаких других предметов.

Запуск:
    python baseline_matanaliz_binary.py
    python baseline_matanaliz_binary.py --data path/to/file.xlsx
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
PASS_GRADES = {"Отлично", "Хорошо", "Зачтено"}
FAIL_GRADES = {"Удовлетворительно", "Неудовлетворительно", "Не зачтено", "Неявка"}

SEMESTER_MAP = {
    "Первый семестр": 1, "Второй семестр": 2, "Третий семестр": 3,
    "Четвертый семестр": 4, "Пятый семестр": 5, "Шестой семестр": 6,
    "Седьмой семестр": 7, "Восьмой семестр": 8, "Девятый семестр": 9,
    "Десятый семестр": 10,
}

DATA_FILE_PATTERNS = [
    "ГОСТ_Р_70946-2023-Приложение-8_sorted.xlsx",
    "ГОСТ Р 70946-2023-Приложение-8_sorted.xlsx",
    "ГОСТ*Приложение*.xlsx",
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
        f"Не нашёл xlsx с данными в {directory}.\n"
        f"Положите файл рядом со скриптом или укажите --data путь/к/файлу.xlsx"
    )


# ---------------------------------------------------------------------------
# Загрузка и подготовка
# ---------------------------------------------------------------------------
def load_data(path: str | Path | None = None) -> pd.DataFrame:
    if path is None:
        path = find_data_file(Path(__file__).parent)
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

    # Дедупликация: Экзамен > Зачет с оценкой, при равном — худшая оценка
    priority = {"Экзамен": 0, "Зачет с оценкой": 1}
    ma["_prio"] = ma["ВидКонтроля"].map(priority).fillna(2)
    ma = ma.sort_values(["ЗачетнаяКнижка", "sem_num", "_prio", "grade_num"])
    ma = ma.drop_duplicates(subset=["ЗачетнаяКнижка", "sem_num"], keep="first")
    ma = ma.drop(columns=["_prio"])

    sem1 = ma[ma["sem_num"] == 1].copy()
    sem2 = ma[ma["sem_num"] == 2][["ЗачетнаяКнижка", "ИтоговаяОтметка"]].copy()

    merged = sem1.merge(sem2, on="ЗачетнаяКнижка", how="inner",
                        suffixes=("_s1", "_s2"))

    # Бинарная целевая: 1 = проходной (4-5), 0 = непроходной (3 и ниже)
    merged["target"] = merged["ИтоговаяОтметка_s2"].apply(
        lambda g: 1 if g in PASS_GRADES else 0
    )
    print(f"  → {len(merged)} студентов с MA в обоих семестрах")
    print(f"  → Проходных: {merged['target'].sum()} ({merged['target'].mean():.1%})")
    print(f"  → Непроходных: {(1 - merged['target']).sum().astype(int)} ({1 - merged['target'].mean():.1%})")

    return merged


# ---------------------------------------------------------------------------
# Обучение и оценка
# ---------------------------------------------------------------------------
def train_and_evaluate(merged: pd.DataFrame, output_dir: Path):

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
    print(f"Целевая: 1 = проходной (4-5), 0 = непроходной (3 и ниже)")

    cat_idx = [1, 4]
    X_train, X_test, y_train, y_test, _, students_test = train_test_split(
        X, y, students, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)} / Test: {len(X_test)}")

    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.05,
        depth=4,
        loss_function="Logloss",
        eval_metric="AUC",
        cat_features=cat_idx,
        random_seed=42,
        verbose=0,
        early_stopping_rounds=100,
        use_best_model=True,
        auto_class_weights="Balanced",
    )

    print("Обучение CatBoost...")
    model.fit(
        Pool(X_train, y_train, cat_features=cat_idx),
        eval_set=Pool(X_test, y_test, cat_features=cat_idx),
    )
    best_iter = model.get_best_iteration()
    print(f"  Лучшая итерация: {best_iter}")

    y_pred = model.predict(X_test).flatten().astype(int)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Метрики
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n{'='*60}")
    print(f"  РЕЗУЛЬТАТЫ ({len(X_test)} студентов)")
    print(f"{'='*60}")
    print(f"  Accuracy:    {acc:.1%}")
    print(f"  F1 (pass):   {f1:.3f}")
    print(f"  F1 (macro):  {f1_macro:.3f}")
    print(f"  ROC-AUC:     {auc:.3f}")
    print()

    report = classification_report(
        y_test, y_pred,
        target_names=["непроходной (3-)", "проходной (4+)"],
        digits=3,
    )
    print(report)

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["факт_непрох", "факт_прох"],
        columns=["пред_непрох", "пред_прох"],
    )
    print("Confusion matrix:")
    print(cm_df.to_string())

    # Важность
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
        t = "прох" if y_test.iloc[i] == 1 else "непрох"
        p = "прох" if y_pred[i] == 1 else "непрох"
        prob = y_proba[i]
        mark = "✓" if y_test.iloc[i] == y_pred[i] else "✗"
        print(f"  Студент {sid}: MA сем.1={g1:.0f} → пред={p} (p={prob:.2f}), факт={t} {mark}")

    # --- Сохранение ---
    output_dir.mkdir(parents=True, exist_ok=True)

    # predictions.csv
    pred_df = pd.DataFrame({
        "ЗачетнаяКнижка": students_test,
        "ma_grade_sem1": X_test["ma_grade_sem1"].values,
        "ma_retake": X_test["ma_retake"].values,
        "plan": X_test["plan"].values,
        "prob_pass": y_proba,
        "predicted": y_pred,
        "actual": y_test.values,
        "correct": (y_pred == y_test.values).astype(int),
    })
    pred_csv = output_dir / "predictions_binary.csv"
    pred_df.to_csv(pred_csv, index=False, encoding="utf-8-sig")
    print(f"\n  Сохранено: {pred_csv}")

    # model
    model_path = output_dir / "model_binary.cbm"
    model.save_model(str(model_path))
    print(f"  Сохранено: {model_path}")

    # summary
    summary_path = output_dir / "summary_binary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"# Бинарный baseline: {TARGET_SUBJECT}\n\n")
        f.write(f"**Задача:** проходной (4+) vs непроходной (3-) по MA, сем.1 → 2\n\n")
        f.write(f"**Дата:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"**Модель:** CatBoostClassifier (binary), {best_iter} итераций\n\n")
        f.write(f"## Признаки (5 штук, только MA)\n\n")
        f.write(f"1. `ma_grade_sem1` — оценка MA в сем.1 (2–5)\n")
        f.write(f"2. `ma_control_type` — Экзамен / Зачет с оценкой\n")
        f.write(f"3. `ma_retake` — была ли пересдача (0/1)\n")
        f.write(f"4. `ma_commission` — была ли комиссия (0/1)\n")
        f.write(f"5. `plan` — учебный план\n\n")
        f.write(f"## Данные\n\n")
        f.write(f"- Студентов: {len(merged)}\n")
        f.write(f"- Train: {len(X_train)} / Test: {len(X_test)}\n")
        f.write(f"- Проходных: {merged['target'].sum()} ({merged['target'].mean():.1%})\n\n")
        f.write(f"## Метрики\n\n")
        f.write(f"| Метрика | Значение |\n|---|---|\n")
        f.write(f"| Accuracy | {acc:.1%} |\n")
        f.write(f"| F1 (pass) | {f1:.3f} |\n")
        f.write(f"| F1 (macro) | {f1_macro:.3f} |\n")
        f.write(f"| ROC-AUC | {auc:.3f} |\n\n")
        f.write(f"## Classification report\n\n```\n{report}```\n\n")
        f.write(f"## Confusion matrix\n\n```\n{cm_df.to_string()}\n```\n\n")
        f.write(f"## Важность признаков\n\n")
        for feat, imp in imp_pairs:
            f.write(f"- `{feat}`: {imp:.1f}\n")
    print(f"  Сохранено: {summary_path}")

    # model_card
    card = {
        "model_type": "CatBoostClassifier (binary)",
        "target": f"pass (4+) vs fail (3-) in {TARGET_SUBJECT}, semester 2",
        "features_used": "ONLY from Математический анализ semester 1",
        "features": list(X.columns),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "best_iteration": best_iter,
        "metrics": {
            "accuracy": round(acc, 4),
            "f1_pass": round(f1, 4),
            "f1_macro": round(f1_macro, 4),
            "roc_auc": round(auc, 4),
        },
        "random_seed": 42,
        "date": datetime.now().isoformat(),
    }
    card_path = output_dir / "model_card_binary.json"
    with open(card_path, "w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=False, indent=2)
    print(f"  Сохранено: {card_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Binary baseline: MA pass/fail prediction"
    )
    parser.add_argument("--data", type=str, default=None,
                        help="Путь к xlsx-файлу с данными")
    parser.add_argument("--output", type=str, default="output",
                        help="Папка для результатов (default: output/)")
    args = parser.parse_args()

    df = load_data(args.data)
    merged = prepare_ma_data(df)

    if len(merged) < 50:
        print("ОШИБКА: слишком мало данных.")
        return

    train_and_evaluate(merged, Path(args.output))

    print(f"\n{'='*60}")
    print(f"  Готово!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
