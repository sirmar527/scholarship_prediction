"""
Конвейер признаков.

`clean_rows` и `build_features` повторяют логику `load_data` и
`build_features` из scholarship_predict_histgb.py шаг в шаг (убраны только
чтение файла и печать в консоль). Любое изменение здесь должно дублироваться
в обучающем скрипте, иначе признаки на инференсе разойдутся с обучением.
Тест tests/test_golden.py проверяет совпадение с predictions.csv.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .constants import (
    DEDUP_PRIORITY,
    DEDUP_PRIORITY_DEFAULT,
    GRADE_MAP,
    KEPT_VEDOMOST_TYPES,
    PASS_FAIL_CONTROL_TYPES,
    PASS_FAIL_GRADES,
    REQUIRED_RAW_COLUMNS,
    RETAKE_VEDOMOST_TYPES,
    SCHOLARSHIP_BLOCKING_GRADES,
    SEMESTER_MAP,
    TRAINING_RANGE,
)


class SchemaError(ValueError):
    """В исходной таблице нет обязательных колонок."""


@dataclass
class Diagnostics:
    """
    Что конвейер сделал с файлом: сколько строк и студентов отфильтровано на
    каждом шаге и какие подписи он не смог прочитать. Признаки от этого не
    зависят - это только отчет.
    """

    rows_read: int = 0
    rows_period_unmapped: int = 0     # ПериодКонтроля не из списка семестров (сессии заочников и т.п.)
    rows_not_full_time: int = 0       # ФормаОбучения != Очная
    rows_vedomost_dropped: int = 0    # тип ведомости вне учитываемых
    rows_retake_duplicates: int = 0   # строки пересдач, дублирующие основную ведомость
    rows_kept: int = 0
    students_in_file: int = 0
    students_kept: int = 0
    periods_unmapped: dict[str, int] = field(default_factory=dict)
    forms: dict[str, int] = field(default_factory=dict)
    vedomost_dropped: dict[str, int] = field(default_factory=dict)
    grades_unmapped: dict[str, int] = field(default_factory=dict)  # в оставшихся строках
    # заполняется при выборе семестра
    students_program_finished: int = 0  # последний семестр программы: следующего нет
    students_without_semester: int = 0
    students_incomplete: int = 0
    students_out_of_range: int = 0   # семестр за пределами обучающих данных
    students_in_result: int = 0

    @property
    def rows_dropped(self) -> int:
        return self.rows_read - self.rows_kept

    def rows_table(self) -> list[tuple[str, int]]:
        """Строки для таблицы «что осталось за кадром»."""
        return [
            ("Строк в файле", self.rows_read),
            ("Период не распознан как семестр", self.rows_period_unmapped),
            ("Не очная форма обучения", self.rows_not_full_time),
            ("Тип ведомости не учитывается", self.rows_vedomost_dropped),
            ("Дубликаты пересдач", self.rows_retake_duplicates),
            ("Строк учтено", self.rows_kept),
            ("Студентов в файле", self.students_in_file),
            ("Студентов после фильтров", self.students_kept),
            ("Исключены: программа окончена", self.students_program_finished),
            ("Исключены: нет выбранного семестра", self.students_without_semester),
            ("В результате", self.students_in_result),
            ("Из них с неполным семестром", self.students_incomplete),
            ("Из них за пределами обучающих данных", self.students_out_of_range),
        ]

    def labels_table(self) -> list[tuple[str, str, int]]:
        """Нераспознанные подписи: (где, значение, сколько строк)."""
        rows = [("Период контроля", k, v) for k, v in self.periods_unmapped.items()]
        rows += [("Форма обучения", k, v) for k, v in self.forms.items() if k != "Очная"]
        rows += [("Тип ведомости", k, v) for k, v in self.vedomost_dropped.items()]
        rows += [("Итоговая отметка", k, v) for k, v in self.grades_unmapped.items()]
        return rows


def validate_raw_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(
            "В таблице нет обязательных колонок: " + ", ".join(missing)
            + ". Ожидается формат ГОСТ Р 70946-2023, Приложение 8."
        )


def read_raw_excel(path_or_buffer, sheet_name="Sheet1") -> pd.DataFrame:
    """Прочитать исходный xlsx. Если листа Sheet1 нет, берется первый лист."""
    try:
        return pd.read_excel(path_or_buffer, sheet_name=sheet_name)
    except ValueError:
        if hasattr(path_or_buffer, "seek"):
            path_or_buffer.seek(0)
        return pd.read_excel(path_or_buffer, sheet_name=0)


def _counts(series: pd.Series) -> dict[str, int]:
    vc = series.astype("string").fillna("(пусто)").value_counts()
    return {str(k): int(v) for k, v in vc.items()}


def clean_rows_with_diagnostics(df: pd.DataFrame) -> tuple[pd.DataFrame, Diagnostics]:
    """Отфильтровать и разметить строки оценок (= load_data без чтения файла) и посчитать, что отфильтровано."""
    validate_raw_columns(df)
    diag = Diagnostics(rows_read=len(df), students_in_file=int(df["ЗачетнаяКнижка"].nunique()))
    diag.forms = _counts(df["ФормаОбучения"])
    df = df.copy()

    df["sem_num"] = df["ПериодКонтроля"].map(SEMESTER_MAP)
    unmapped = df["sem_num"].isna()
    diag.rows_period_unmapped = int(unmapped.sum())
    diag.periods_unmapped = _counts(df.loc[unmapped, "ПериодКонтроля"])
    df = df[~unmapped].copy()
    df["sem_num"] = df["sem_num"].astype(int)

    not_full_time = df["ФормаОбучения"] != "Очная"
    diag.rows_not_full_time = int(not_full_time.sum())
    df = df[~not_full_time]

    bad_type = ~df["ТипВедомости"].isin(KEPT_VEDOMOST_TYPES)
    diag.rows_vedomost_dropped = int(bad_type.sum())
    diag.vedomost_dropped = _counts(df.loc[bad_type, "ТипВедомости"])
    df = df[~bad_type].copy()

    # пересдачи чаще всего дублируют строку Основная для той же
    # (студент, семестр, дисциплина) - такие дубликаты убираем
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
    diag.rows_retake_duplicates = int(drop_mask.sum())
    if drop_mask.any():
        df = df[~drop_mask].copy()

    # В выгрузках эти колонки бывают текстовыми; для чисел результат тот же
    # (NaN > 0 было False, теперь 0 > 0 - тоже False).
    for col in ("Пересдача", "Комиссия"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["grade_num"] = df["ИтоговаяОтметка"].map(GRADE_MAP)
    unknown_grade = df["grade_num"].isna()
    diag.grades_unmapped = _counts(df.loc[unknown_grade, "ИтоговаяОтметка"])
    df["is_scholarship_blocking"] = (
        df["ИтоговаяОтметка"].isin(SCHOLARSHIP_BLOCKING_GRADES).astype(int)
    )
    df["has_retake"] = (
        (df["Пересдача"] > 0)
        | (df["Комиссия"] > 0)
        | (df["ИтоговаяОтметка"] == "Неявка")
    ).astype(int)
    df.loc[df["ТипВедомости"].isin(RETAKE_VEDOMOST_TYPES), "has_retake"] = 1

    # Перезачет: засчитывается в нагрузку и GPA, но не блокирует стипендию
    perezachet_mask = df["ТипВедомости"] == "Перезачет"
    df.loc[perezachet_mask, "is_scholarship_blocking"] = 0
    df.loc[perezachet_mask, "has_retake"] = 0

    diag.rows_kept = len(df)
    diag.students_kept = int(df["ЗачетнаяКнижка"].nunique())
    return df, diag


def clean_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Отфильтровать и разметить строки оценок (= load_data без чтения файла)."""
    return clean_rows_with_diagnostics(df)[0]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Агрегировать оценки в одну строку на (студент, семестр)."""
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

    return overall


def finalize_for_inference(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавить `had_clean_current_sem`.

    В обучающем скрипте этот признак появляется только в build_pairs и равен
    clean_record текущего семестра. На инференсе следующего семестра еще нет,
    поэтому берем тот же clean_record напрямую.
    """
    out = features_df.copy()
    out["had_clean_current_sem"] = out["clean_record"].astype(int)
    return out


def to_matrix(features_df: pd.DataFrame, feature_order: list[str]) -> np.ndarray:
    """Собрать матрицу признаков строго в порядке model_card.json."""
    missing = [c for c in feature_order if c not in features_df.columns]
    if missing:
        raise KeyError(f"Нет признаков: {missing}")
    X = features_df[feature_order].copy()
    # один оцениваемый предмет -> std не определено -> 0 (как при обучении)
    if "std_grade" in X.columns:
        X["std_grade"] = X["std_grade"].fillna(0)
    return X.to_numpy(dtype=float)


INCOMPLETE_RATIO = 0.5   # семестр «неполный», если дисциплин меньше половины медианы прошлых семестров
INCOMPLETE_MIN_SUBJECTS = 2  # ...или не больше двух, когда прошлых семестров нет


def add_completeness_flag(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Пометить семестры, похожие на незавершенные (выгрузка посреди сессии).

    Сравниваем число дисциплин с медианой по более ранним семестрам того же
    студента; без истории считаем подозрительными семестры из 1-2 дисциплин.
    На признаки модели колонка не влияет.
    """
    out = features_df.sort_values(["ЗачетнаяКнижка", "sem_num"]).copy()
    prev_median = out.groupby("ЗачетнаяКнижка")["n_subjects"].transform(
        lambda s: s.expanding().median().shift(1)
    )
    has_history = prev_median.notna()
    out["maybe_incomplete"] = (
        (has_history & (out["n_subjects"] < INCOMPLETE_RATIO * prev_median))
        | (~has_history & (out["n_subjects"] <= INCOMPLETE_MIN_SUBJECTS))
    )
    return out.sort_index()


METADATA_COLUMNS = ["УровеньПодготовки", "УчебныйПлан"]


def attach_metadata(features_df: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame:
    """Добавить уровень подготовки и учебный план (метаданные, не признаки):
    первый нужен, чтобы понять, есть ли у студента следующий семестр вообще,
    второй - для фильтров по группе."""
    present = [c for c in METADATA_COLUMNS if c in cleaned.columns]
    if not present:
        return features_df
    meta = cleaned.groupby(["ЗачетнаяКнижка", "sem_num"])[present].first().reset_index()
    return features_df.merge(meta, on=["ЗачетнаяКнижка", "sem_num"], how="left")


attach_level = attach_metadata  # прежнее имя


def range_notes(features: dict) -> list[str]:
    """
    Чем семестр не похож на обучающие данные. Пустой список - все в норме.

    Замечания нужны, потому что за пределами данных число на экране - не оценка
    вероятности, а результат экстраполяции по последнему листу деревьев.
    """
    r = TRAINING_RANGE
    notes: list[str] = []
    n = features.get("n_subjects")
    blocks = features.get("n_blocks")
    gpa = features.get("gpa_overall")
    std = features.get("std_grade")
    share_zachet = features.get("share_zachet") or 0.0
    if n is not None:
        n = int(n)
        if n > r["n_subjects_max"]:
            notes.append(
                f"В семестре {n} дисциплин - в обучающих данных не было семестров больше {r['n_subjects_max']}: "
                f"число получено экстраполяцией и не является оценкой вероятности."
            )
        elif n > r["n_subjects_p99"]:
            notes.append(f"В семестре {n} дисциплин - таких семестров в обучающих данных меньше 1%: оценка ненадежна.")
    if blocks is not None:
        blocks = int(blocks)
        if blocks > r["n_blocks_max"]:
            notes.append(
                f"Троек и долгов ({blocks}) больше, чем в любом семестре обучающих данных (максимум {r['n_blocks_max']}): "
                f"число получено экстраполяцией."
            )
        elif blocks > r["n_blocks_p99"]:
            notes.append(f"Троек и долгов больше {r['n_blocks_p99']} - таких семестров в обучающих данных меньше 1%: оценка ненадежна.")
    graded = round((n or 0) * (1 - share_zachet))
    if gpa is not None and std is not None and graded >= 3 and float(gpa) == 2.0 and float(std) == 0.0:
        notes.append(
            f"Все оценки - одинаковые двойки: таких семестров в обучающих данных около "
            f"{r['identical_fail_share'] * 100:.0f}%, оценка ненадежна."
        )
    return notes


def add_range_flags(features_df: pd.DataFrame) -> pd.DataFrame:
    """Колонка out_of_range: семестр за пределами (или на редком краю) обучающих данных."""
    r = TRAINING_RANGE
    out = features_df.copy()
    graded = (out["n_subjects"] * (1 - out["share_zachet"].fillna(0))).round()
    identical_fail = (out["gpa_overall"] == 2.0) & (out["std_grade"].fillna(0) == 0.0) & (graded >= 3)
    out["out_of_range"] = (
        (out["n_subjects"] > r["n_subjects_p99"]) | (out["n_blocks"] > r["n_blocks_p99"]) | identical_fail
    )
    return out


def features_from_raw_with_diagnostics(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, Diagnostics]:
    """Полный путь с отчетом: сырые строки -> признаки + что было отфильтровано."""
    cleaned, diag = clean_rows_with_diagnostics(raw_df)
    feats = attach_metadata(finalize_for_inference(build_features(cleaned)), cleaned)
    feats = add_range_flags(add_completeness_flag(feats))
    return feats, diag


def features_from_raw(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Полный путь: сырые строки оценок -> признаки, готовые для модели."""
    return features_from_raw_with_diagnostics(raw_df)[0]
