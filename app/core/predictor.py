"""
Обертка над обученной моделью.

Загружает scholarship_model.joblib и model_card.json (порядок признаков
берется только из карточки: модель обучена на numpy-массиве и имен
признаков не хранит).
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd

from .constants import DECISION_THRESHOLD, DEFAULT_LAST_SEMESTER, PROGRAM_LAST_SEMESTER, risk_band, verdict_label
from .features import Diagnostics, features_from_raw, features_from_raw_with_diagnostics, range_notes, to_matrix
from .manual import WHAT_IF_SCENARIOS_RAW, Subject, subjects_to_raw

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_MODEL = MODELS_DIR / "scholarship_model.joblib"
DEFAULT_CARD = MODELS_DIR / "model_card.json"

# Человекочитаемые названия признаков для интерфейса.
FEATURE_LABELS = {
    "sem_num": "Семестр",
    "n_subjects": "Дисциплин в семестре",
    "n_blocks": "Троек и долгов",
    "n_retakes": "Пересдач",
    "any_block": "Есть тройки или долги",
    "any_retake": "Есть пересдача",
    "gpa_overall": "Средний балл",
    "min_grade": "Минимальная оценка",
    "std_grade": "Разброс оценок",
    "share_5": "Доля пятерок",
    "share_3": "Доля троек",
    "share_zachet": "Доля зачетов в нагрузке",
    "had_clean_current_sem": "Текущий семестр без троек, долгов и пересдач",
}


@dataclass
class Prediction:
    sem_num: int
    prob_clean_next: float
    features: dict = field(default_factory=dict)
    incomplete: bool = False  # семестр из 1-2 дисциплин: модель училась на полных
    range_notes: list[str] = field(default_factory=list)  # чем семестр не похож на обучающие данные

    @property
    def out_of_range(self) -> bool:
        return bool(self.range_notes)

    @property
    def target_sem(self) -> int:
        return self.sem_num + 1

    @property
    def stipend_sem(self) -> int:
        return self.sem_num + 2

    @property
    def pred_clean_next(self) -> int:
        return int(self.prob_clean_next >= DECISION_THRESHOLD)

    @property
    def band(self) -> tuple[str, str]:
        return risk_band(self.prob_clean_next)

    @property
    def verdict(self) -> str:
        """Персональная формулировка: сохранит / потеряет / получит / останется без."""
        return verdict_label(self.band[0], self.current_clean)

    @property
    def current_clean(self) -> bool:
        return bool(self.features.get("had_clean_current_sem", 0))


class ScholarshipPredictor:
    def __init__(self, model_path: Path = DEFAULT_MODEL, card_path: Path = DEFAULT_CARD):
        self.model_path = Path(model_path)
        self.card_path = Path(card_path)
        with open(self.card_path, encoding="utf-8") as f:
            self.card = json.load(f)
        self.feature_order: list[str] = list(self.card["features"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.model = joblib.load(self.model_path)
        # предупреждение о несовпадении версии sklearn показываем в UI
        self.load_warnings = [str(w.message) for w in caught]

    # --- низкий уровень ------------------------------------------------------

    def predict_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Признаки (одна строка на студент-семестр) -> вероятности и классы."""
        out = features_df.copy()
        if len(out) == 0:
            out["prob_clean_next_sem"] = []
            out["pred_clean_next_sem"] = []
            return out
        X = to_matrix(out, self.feature_order)
        proba = self.model.predict_proba(X)[:, 1]
        out["prob_clean_next_sem"] = proba
        out["pred_clean_next_sem"] = (proba >= DECISION_THRESHOLD).astype(int)
        out["target_sem"] = out["sem_num"] + 1
        bands = [risk_band(p) for p in proba]
        out["risk_key"] = [b[0] for b in bands]
        out["risk_label"] = [b[1] for b in bands]
        out["verdict_label"] = [
            verdict_label(k, bool(c)) for k, c in zip(out["risk_key"], out["had_clean_current_sem"], strict=True)
        ]
        return out

    # --- пакетный режим (xlsx) -----------------------------------------------

    @staticmethod
    def select_semester(
        feats: pd.DataFrame, semester: int | None = None, diag: Diagnostics | None = None
    ) -> pd.DataFrame:
        """
        Оставить по одной строке на студента.

        semester=None: последний семестр каждого студента. Иначе - только указанный.
        Последний семестр программы исключается: следующего, который надо предсказывать, не бывает.
        Если передан diag, в него записывается, кто и почему выпал.
        """
        students_all = int(feats["ЗачетнаяКнижка"].nunique()) if len(feats) else 0
        if semester is not None:
            feats = feats[feats["sem_num"] == int(semester)]
            if diag is not None:
                diag.students_without_semester = students_all - int(feats["ЗачетнаяКнижка"].nunique())
        elif len(feats):
            idx = feats.groupby("ЗачетнаяКнижка")["sem_num"].idxmax()
            feats = feats.loc[idx]
        before = int(feats["ЗачетнаяКнижка"].nunique()) if len(feats) else 0
        # после последнего семестра программы (8-й у бакалавров, 10-й у специалистов)
        # следующего семестра нет - предсказывать нечего
        if "УровеньПодготовки" in feats.columns:
            last = feats["УровеньПодготовки"].map(PROGRAM_LAST_SEMESTER).fillna(DEFAULT_LAST_SEMESTER)
        else:
            last = pd.Series(DEFAULT_LAST_SEMESTER, index=feats.index)
        feats = feats[feats["sem_num"] < last]
        if diag is not None:
            diag.students_program_finished = before - (int(feats["ЗачетнаяКнижка"].nunique()) if len(feats) else 0)
            diag.students_in_result = len(feats)
            if "maybe_incomplete" in feats.columns:
                diag.students_incomplete = int(feats["maybe_incomplete"].sum())
            if "out_of_range" in feats.columns:
                diag.students_out_of_range = int(feats["out_of_range"].sum())
        return feats.reset_index(drop=True)

    def predict_raw(self, raw_df: pd.DataFrame, semester: int | None = None) -> pd.DataFrame:
        """Сырые строки оценок -> прогноз по студентам (см. select_semester)."""
        return self.predict_features(self.select_semester(features_from_raw(raw_df), semester))

    def predict_raw_with_diagnostics(
        self, raw_df: pd.DataFrame, semester: int | None = None
    ) -> tuple[pd.DataFrame, Diagnostics]:
        """То же, плюс отчет о том, что не попало в результат."""
        feats, diag = features_from_raw_with_diagnostics(raw_df)
        return self.predict_features(self.select_semester(feats, semester, diag)), diag

    def all_semesters(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Прогноз для всех (студент, семестр) - нужен для golden-теста."""
        feats = features_from_raw(raw_df)
        return self.predict_features(feats)

    # --- ручной ввод ---------------------------------------------------------

    def prediction_from_row(self, row: pd.Series) -> Prediction:
        """Строка результата (predict_features) -> Prediction для экрана одного студента."""
        feats = {k: (None if pd.isna(row[k]) else float(row[k])) for k in self.feature_order}
        return Prediction(
            sem_num=int(row["sem_num"]),
            prob_clean_next=float(row["prob_clean_next_sem"]),
            features=feats,
            incomplete=bool(row.get("maybe_incomplete", False)),
            range_notes=range_notes(feats),
        )

    def predict_rows(self, rows: pd.DataFrame, sem_num: int) -> Prediction:
        """Строки исходной схемы одного студента за один семестр -> Prediction."""
        result = self.predict_features(features_from_raw(rows))
        if result.empty:
            raise ValueError("после фильтров конвейера не осталось ни одной строки")
        return self.prediction_from_row(result.iloc[0])

    def predict_subjects(self, subjects: list[Subject], sem_num: int) -> Prediction:
        return self.predict_rows(subjects_to_raw(subjects, sem_num), sem_num)

    def what_if_rows(self, rows: pd.DataFrame, sem_num: int, base: Prediction | None = None) -> list[dict]:
        """Сценарии над строками исходной схемы; возвращаются только те, что что-то меняют."""
        if base is None:
            base = self.predict_rows(rows, sem_num)
        out = []
        for label, transform in WHAT_IF_SCENARIOS_RAW:
            changed = transform(rows)
            if changed.equals(rows):
                continue
            alt = self.predict_rows(changed, sem_num)
            out.append(
                {
                    "scenario": label,
                    "prob": alt.prob_clean_next,
                    "delta": alt.prob_clean_next - base.prob_clean_next,
                    "band": alt.band[1],
                }
            )
        return out

    def what_if(self, subjects: list[Subject], sem_num: int, base: Prediction | None = None) -> list[dict]:
        """Форма идет тем же путем: Subject -> строки исходной схемы -> сценарии."""
        return self.what_if_rows(subjects_to_raw(subjects, sem_num), sem_num, base)

    # --- справка -------------------------------------------------------------

    @property
    def metrics(self) -> dict:
        return self.card.get("metrics", {})

    @property
    def n_train(self) -> int:
        return int(self.card.get("n_train", 0))

    @property
    def baselines(self) -> dict:
        """Точность простых правил из карточки модели, например
        baselines["persistence"]["accuracy"] - «следующий семестр как текущий»."""
        return self.card.get("baselines", {})
