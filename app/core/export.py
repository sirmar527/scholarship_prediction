"""
Выгрузка результата в xlsx с листами «О модели» и «Диагностика».

Файл, найденный через полгода, должен сам объяснять, какая модель и какие
правила его породили и что из исходной выгрузки не было учтено.
"""

from __future__ import annotations

import io
from datetime import datetime

import pandas as pd

from .constants import DECISION_THRESHOLD, RISK_BANDS, VERDICT_LABELS
from .features import Diagnostics


def model_sheet(predictor, semester: int | None) -> list[tuple[str, str]]:
    card, m = predictor.card, predictor.metrics
    persistence = predictor.baselines.get("persistence", {}).get("accuracy")
    rows = [
        ("Модель", str(card.get("model_type", ""))),
        ("Дата обучения", str(card.get("date", ""))[:10]),
        ("Итераций бустинга", str(card.get("n_iter", ""))),
        ("Признаков", str(card.get("n_features", len(predictor.feature_order)))),
        ("Обучающих пар (семестр N -> N+1)", str(card.get("n_train", ""))),
        ("Тестовых пар", str(card.get("n_test", ""))),
        ("Точность на тесте", f"{m.get('accuracy', 0):.4f}"),
        ("Точность правила «следующий семестр как текущий»", f"{persistence:.4f}" if persistence else ""),
        ("ROC-AUC на тесте", f"{m.get('roc_auc', 0):.4f}"),
        ("Что оценивается", "закроет ли студент семестр N+1 без троек, долгов и пересдач; это определяет стипендию в N+2"),
        ("Порог класса", f"{DECISION_THRESHOLD:.2f}"),
    ]
    for lower, key, label in RISK_BANDS:
        rows.append((f"Зона «{label}»", f"оценка >= {lower:.2f}"))
    rows.append(("Персональные подписи", "; ".join(f"{k[0]}/{'стипендия есть' if k[1] else 'стипендии нет'}: {v}" for k, v in VERDICT_LABELS.items())))
    rows.append(("Текущий семестр", f"{semester}-й" if semester else "последний семестр каждого студента"))
    rows.append(("Файл модели", predictor.model_path.name))
    rows.append(("Сформировано", datetime.now().strftime("%Y-%m-%d %H:%M")))
    rows.append(("Замечание", "оценка модели, обученной с балансировкой классов; в среднем она немного оптимистична"))
    return rows


def build_export(table: pd.DataFrame, diag: Diagnostics | None, predictor, semester: int | None) -> bytes:
    """Три листа: Прогноз, О модели, Диагностика."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        table.to_excel(xw, index=False, sheet_name="Прогноз")
        pd.DataFrame(model_sheet(predictor, semester), columns=["Параметр", "Значение"]).to_excel(
            xw, index=False, sheet_name="О модели"
        )
        if diag is not None:
            counts = pd.DataFrame(diag.rows_table(), columns=["Показатель", "Значение"])
            labels = pd.DataFrame(diag.labels_table(), columns=["Где", "Значение", "Строк"])
            counts.to_excel(xw, index=False, sheet_name="Диагностика")
            if len(labels):
                labels.to_excel(xw, index=False, sheet_name="Диагностика", startrow=len(counts) + 2)
        for sheet in xw.sheets.values():
            for col in sheet.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                sheet.column_dimensions[col[0].column_letter].width = min(max(12, width + 2), 80)
    return buf.getvalue()
