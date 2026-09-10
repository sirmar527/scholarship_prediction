"""Сводки по группе для пакетного режима: гистограмма оценок и разрез по учебным планам."""

from __future__ import annotations

import numpy as np
import pandas as pd

HIST_EDGES = np.linspace(0, 1, 11)


def probability_histogram(result: pd.DataFrame) -> pd.DataFrame:
    """Число студентов в десяти интервалах оценки; индекс - подпись интервала."""
    counts, _ = np.histogram(result["prob_clean_next_sem"].clip(0, 1 - 1e-9), bins=HIST_EDGES)
    labels = [f"{int(lo * 100)}-{int(hi * 100)}%" for lo, hi in zip(HIST_EDGES[:-1], HIST_EDGES[1:], strict=True)]
    return pd.DataFrame({"Студентов": counts}, index=pd.Index(labels, name="Оценка"))


def by_plan(result: pd.DataFrame, min_students: int = 5) -> pd.DataFrame:
    """Разрез по учебным планам: сколько студентов, средняя оценка, доли по зонам."""
    if "УчебныйПлан" not in result.columns or result.empty:
        return pd.DataFrame()
    g = result.groupby("УчебныйПлан")
    table = pd.DataFrame(
        {
            "Студентов": g.size(),
            "Средняя оценка": g["prob_clean_next_sem"].mean(),
            "Со стипендией": g["risk_key"].apply(lambda s: (s == "high").mean()),
            "Под вопросом": g["risk_key"].apply(lambda s: (s == "mid").mean()),
            "Без стипендии": g["risk_key"].apply(lambda s: (s == "low").mean()),
        }
    )
    table = table[table["Студентов"] >= min_students].sort_values("Средняя оценка")
    table.index.name = "Учебный план"
    return table.reset_index()


def apply_filters(result: pd.DataFrame, levels: list | None, plans: list | None) -> pd.DataFrame:
    out = result
    if levels and "УровеньПодготовки" in out.columns:
        out = out[out["УровеньПодготовки"].isin(levels)]
    if plans and "УчебныйПлан" in out.columns:
        out = out[out["УчебныйПлан"].isin(plans)]
    return out
