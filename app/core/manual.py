"""
Ручной ввод: превращаем строки формы в таблицу исходной схемы.

Форма не пересчитывает признаки сама. Каждая дисциплина становится строкой
в формате ГОСТ Р 70946-2023 (Приложение 8) и проходит через тот же
конвейер, что и данные при обучении.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

import json

from .constants import (
    ALL_GRADE_LABELS,
    CONTROL_TYPES,
    DEDUP_PRIORITY,
    DEDUP_PRIORITY_DEFAULT,
    GRADE_MAP,
    MAX_INPUT_SEMESTER,
    MIN_SEMESTER,
    PASS_FAIL_GRADES,
    RAW_COLUMNS,
    RETAKE_VEDOMOST_TYPES,
    SEMESTER_MAP,
    SEMESTER_NAMES,
)

MANUAL_STUDENT_ID = "manual"


@dataclass
class Subject:
    name: str
    control_type: str
    grade: str
    retake: bool = False


class ManualInputError(ValueError):
    """Форма заполнена некорректно."""


def validate_subjects(subjects: list[Subject], sem_num: int) -> None:
    if not (MIN_SEMESTER <= sem_num <= MAX_INPUT_SEMESTER):
        raise ManualInputError(
            f"Семестр должен быть от {MIN_SEMESTER} до {MAX_INPUT_SEMESTER}: "
            f"модель предсказывает следующий семестр, а одиннадцатого не бывает."
        )
    if not subjects:
        raise ManualInputError("Добавьте хотя бы одну дисциплину.")
    seen = set()
    for s in subjects:
        if not s.name or not str(s.name).strip():
            raise ManualInputError("У каждой дисциплины должно быть название.")
        if s.control_type not in CONTROL_TYPES:
            raise ManualInputError(f"Неизвестный вид контроля: {s.control_type}")
        if s.grade not in ALL_GRADE_LABELS:
            raise ManualInputError(f"Неизвестная оценка: {s.grade}")
        key = str(s.name).strip().lower()
        if key in seen:
            raise ManualInputError(
                f"Дисциплина «{s.name}» указана дважды. Оставьте одну строку "
                f"с итоговой оценкой."
            )
        seen.add(key)


def subjects_to_raw(
    subjects: list[Subject], sem_num: int, student_id: str = MANUAL_STUDENT_ID
) -> pd.DataFrame:
    """Собрать DataFrame в исходной схеме из строк формы."""
    validate_subjects(subjects, sem_num)
    rows = []
    for s in subjects:
        rows.append(
            {
                "ЗачетнаяКнижка": student_id,
                "УчебныйПлан": 0,
                "УровеньПодготовки": "Бакалавриат",
                "ФормаОбучения": "Очная",
                "ПериодКонтроля": SEMESTER_NAMES[sem_num],
                "ВидКонтроля": s.control_type,
                "Дисциплина": str(s.name).strip(),
                "ТипВедомости": "Основная",
                "КодПреподавателя": 0,
                "ПерваяАттестация": 0,
                "ВтораяАттестация": 0,
                "ТретьяАттестация": 0,
                "Экзамен": 0,
                # положительное значение = была пересдача (так размечены данные)
                "Пересдача": 1 if s.retake else 0,
                "Комиссия": 0,
                "ПересдачаДляДиплома": 0,
                "ИтоговаяОтметка": s.grade,
            }
        )
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def subjects_from_records(records: list[dict]) -> list[Subject]:
    """Строки редактора таблицы -> Subject (пустые строки пропускаем)."""
    out = []
    for r in records:
        name = r.get("Дисциплина")
        if name is None or (isinstance(name, float)) or not str(name).strip():
            continue
        out.append(
            Subject(
                name=str(name),
                control_type=r.get("Вид контроля") or "Экзамен",
                grade=r.get("Оценка") or "Хорошо",
                retake=bool(r.get("Пересдача") or False),
            )
        )
    return out


# --- семестр, заданный количествами оценок -----------------------------------

@dataclass
class GradeCounts:
    """
    Семестр без названий дисциплин: сколько каких оценок получено.

    Из этих чисел генерируются строки Subject с условными названиями, и дальше
    работает обычный конвейер - поэтому признаки всегда согласованы между собой.
    """

    excellent: int = 0      # Отлично (экзамен / зачет с оценкой)
    good: int = 0           # Хорошо
    satisfactory: int = 0   # Удовлетворительно
    fail: int = 0           # Неудовлетворительно
    absent_exam: int = 0    # Неявка на экзамен (входит в средний балл как 2)
    passed: int = 0         # Зачтено
    not_passed: int = 0     # Не зачтено
    absent_pass: int = 0    # Неявка на зачет (в средний балл не входит)
    extra_retakes: int = 0  # дисциплин с пересдачей помимо неявок

    @property
    def total(self) -> int:
        return (
            self.excellent + self.good + self.satisfactory + self.fail + self.absent_exam
            + self.passed + self.not_passed + self.absent_pass
        )

    @property
    def absences(self) -> int:
        return self.absent_exam + self.absent_pass


# Порядок генерации: от худших к лучшим. Дополнительные пересдачи получают
# первые дисциплины после неявок. На признаки модели это распределение не влияет
# (считаются только n_retakes и any_retake), оно нужно лишь для правдоподобия.
_COUNT_ORDER = [
    ("absent_exam", "Экзамен", "Неявка"),
    ("absent_pass", "Зачет", "Неявка"),
    ("fail", "Экзамен", "Неудовлетворительно"),
    ("not_passed", "Зачет", "Не зачтено"),
    ("satisfactory", "Экзамен", "Удовлетворительно"),
    ("good", "Экзамен", "Хорошо"),
    ("passed", "Зачет", "Зачтено"),
    ("excellent", "Экзамен", "Отлично"),
]


def validate_counts(c: GradeCounts) -> None:
    fields = [
        c.excellent, c.good, c.satisfactory, c.fail, c.absent_exam,
        c.passed, c.not_passed, c.absent_pass, c.extra_retakes,
    ]
    if any(v < 0 for v in fields):
        raise ManualInputError("Количества не могут быть отрицательными.")
    if c.total == 0:
        raise ManualInputError("Укажите хотя бы одну оценку.")
    available = c.total - c.absences
    if c.extra_retakes > available:
        raise ManualInputError(
            f"Пересдач помимо неявок не может быть больше {available}: "
            f"неявка уже считается пересдачей, а других дисциплин в семестре {available}."
        )


def subjects_from_counts(c: GradeCounts) -> list[Subject]:
    """Количества оценок -> список Subject с условными названиями."""
    validate_counts(c)
    subjects: list[Subject] = []
    for field, control_type, grade in _COUNT_ORDER:
        for _ in range(getattr(c, field)):
            subjects.append(Subject(f"Дисциплина {len(subjects) + 1}", control_type, grade))
    # неявки уже идут первыми и считаются пересдачами в конвейере;
    # дополнительные пересдачи - следующим по порядку дисциплинам
    for s in subjects[c.absences : c.absences + c.extra_retakes]:
        s.retake = True
    return subjects


# Сценарии «что если» для панели объяснения.
def clear_retakes(subjects: list[Subject]) -> list[Subject]:
    return [Subject(s.name, s.control_type, s.grade, False) for s in subjects]


def threes_to_fours(subjects: list[Subject]) -> list[Subject]:
    return [
        Subject(
            s.name,
            s.control_type,
            "Хорошо" if s.grade == "Удовлетворительно" else s.grade,
            s.retake,
        )
        for s in subjects
    ]


def fix_failures(subjects: list[Subject]) -> list[Subject]:
    """
    Незачеты, двойки и неявки -> минимальный положительный результат.

    Долг закрывается пересдачей, поэтому у исправленных дисциплин остается
    флаг пересдачи (для неявки конвейер ставил его сам, пока оценка была
    «Неявка»; после исправления флаг нужно сохранить явно).
    """
    fixed = []
    for s in subjects:
        grade = s.grade
        if grade in ("Не зачтено",):
            grade = "Зачтено"
        elif grade in ("Неудовлетворительно", "Неявка"):
            grade = "Зачтено" if s.control_type == "Зачет" else "Удовлетворительно"
        changed = grade != s.grade
        fixed.append(Subject(s.name, s.control_type, grade, s.retake or changed))
    return fixed


WHAT_IF_SCENARIOS = [
    ("Без пересдач", clear_retakes),
    ("Тройки исправлены на четверки", threes_to_fours),
    ("Долги закрыты пересдачей", fix_failures),
]


# --- те же сценарии над строками исходной схемы --------------------------------
# Нужны, чтобы «что если» работало для реального студента из выгрузки, а не
# только для формы. Форма тоже идет этим путем: Subject -> raw -> сценарий.

def clear_retakes_raw(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    out["Пересдача"] = 0
    out["Комиссия"] = 0
    out.loc[out["ТипВедомости"].isin(RETAKE_VEDOMOST_TYPES), "ТипВедомости"] = "Основная"
    return out


def threes_to_fours_raw(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    out.loc[out["ИтоговаяОтметка"] == "Удовлетворительно", "ИтоговаяОтметка"] = "Хорошо"
    return out


def fix_failures_raw(rows: pd.DataFrame) -> pd.DataFrame:
    """Незачеты, двойки и неявки закрыты пересдачей на минимальный положительный результат."""
    out = rows.copy()
    not_passed = out["ИтоговаяОтметка"] == "Не зачтено"
    failed = out["ИтоговаяОтметка"].isin(["Неудовлетворительно", "Неявка"])
    is_pass_fail = out["ВидКонтроля"] == "Зачет"
    out.loc[not_passed, "ИтоговаяОтметка"] = "Зачтено"
    out.loc[failed & is_pass_fail, "ИтоговаяОтметка"] = "Зачтено"
    out.loc[failed & ~is_pass_fail, "ИтоговаяОтметка"] = "Удовлетворительно"
    changed = not_passed | failed
    retakes = pd.to_numeric(out["Пересдача"], errors="coerce").fillna(0)
    out["Пересдача"] = retakes.where(~changed, retakes.clip(lower=1))
    return out


WHAT_IF_SCENARIOS_RAW = [
    ("Без пересдач", clear_retakes_raw),
    ("Тройки исправлены на четверки", threes_to_fours_raw),
    ("Долги закрыты пересдачей", fix_failures_raw),
]


# --- реальный студент -> строки редактора --------------------------------------

def subjects_from_raw(rows: pd.DataFrame) -> tuple[list[Subject], list[str]]:
    """
    Строки выгрузки одного студента за один семестр -> строки редактора.

    Дисциплина с несколькими ведомостями (экзамен и курсовая) сводится к одной
    строке с худшей оценкой - так же поступает конвейер. Виды контроля вне списка
    редактора (ГАК, Защита ВКР) заменяются на «Экзамен» или «Зачет» по типу оценки;
    все такие замены перечисляются в notes.
    """
    notes: list[str] = []
    df = rows.copy()
    df = df[df["ИтоговаяОтметка"].isin(ALL_GRADE_LABELS)]
    skipped = len(rows) - len(df)
    if skipped:
        notes.append(f"{skipped} строк с нераспознанной оценкой пропущено")
    if df.empty:
        return [], notes

    retake_num = pd.to_numeric(df["Пересдача"], errors="coerce").fillna(0)
    commission = pd.to_numeric(df["Комиссия"], errors="coerce").fillna(0)
    df = df.assign(
        _grade=df["ИтоговаяОтметка"].map(GRADE_MAP),
        _prio=df["ВидКонтроля"].map(DEDUP_PRIORITY).fillna(DEDUP_PRIORITY_DEFAULT),
        _retake=(retake_num > 0) | (commission > 0) | df["ТипВедомости"].isin(RETAKE_VEDOMOST_TYPES),
    )
    merged = 0
    subjects: list[Subject] = []
    graded_label = {5: "Отлично", 4: "Хорошо", 3: "Удовлетворительно", 2: "Неудовлетворительно"}
    for name, grp in df.groupby("Дисциплина", sort=False):
        if len(grp) > 1:
            merged += len(grp) - 1
        # как конвейер: представитель - ведомость с высшим приоритетом, оценка - худшая по дисциплине
        rep = grp.sort_values(["_prio", "_grade"]).iloc[0]
        min_grade = int(grp["_grade"].min())
        any_absent = bool((grp["ИтоговаяОтметка"] == "Неявка").any())
        control = rep["ВидКонтроля"]
        rep_pass_fail = control == "Зачет" or rep["ИтоговаяОтметка"] in PASS_FAIL_GRADES
        if rep_pass_fail:
            grade = "Зачтено" if min_grade >= 4 else ("Неявка" if any_absent else "Не зачтено")
        else:
            grade = "Неявка" if (min_grade == 2 and any_absent) else graded_label[min_grade]
        if control not in CONTROL_TYPES:
            replacement = "Зачет" if rep_pass_fail else "Экзамен"
            notes.append(f"«{name}»: вид контроля «{control}» показан как «{replacement}»")
            control = replacement
        elif rep_pass_fail and control != "Зачет":
            control = "Зачет"  # зачтено/не зачтено в ведомости с оценкой - для редактора это зачет
        subjects.append(Subject(str(name), control, grade, bool(grp["_retake"].any())))
    if merged:
        notes.append(f"{merged} строк объединено с другими ведомостями тех же дисциплин (взята худшая оценка)")
    transferred = int((df["ТипВедомости"] == "Перезачет").sum())
    if transferred:
        notes.append(
            f"{transferred} дисциплин перезачтены: в редакторе перезачет не отличить от обычной оценки, "
            f"поэтому тройки в них станут блокирующими и оценка может оказаться ниже, чем в таблице"
        )
    return subjects, notes


def semester_of_rows(rows: pd.DataFrame) -> int | None:
    sems = rows["ПериодКонтроля"].map(SEMESTER_MAP).dropna().unique()
    return int(sems[0]) if len(sems) == 1 else None


# --- сохранение семестра в JSON ----------------------------------------------------

def subjects_to_json(subjects: list[Subject], sem_num: int) -> str:
    return json.dumps(
        {"semester": int(sem_num), "subjects": [s.__dict__ for s in subjects]},
        ensure_ascii=False,
        indent=2,
    )


def subjects_from_json(text: str) -> tuple[list[Subject], int]:
    try:
        data = json.loads(text)
        sem_num = int(data["semester"])
        subjects = [Subject(str(d["name"]), str(d["control_type"]), str(d["grade"]), bool(d.get("retake", False))) for d in data["subjects"]]
    except (ValueError, KeyError, TypeError) as e:
        raise ManualInputError(f"Файл не похож на сохраненный семестр: {e}") from e
    validate_subjects(subjects, sem_num)
    return subjects, sem_num


# --- готовые примеры для редактора -------------------------------------------------

PRESETS = {
    "Без троек и пересдач": [
        Subject("Математический анализ", "Экзамен", "Отлично"),
        Subject("Программирование", "Экзамен", "Отлично"),
        Subject("Дискретная математика", "Экзамен", "Хорошо"),
        Subject("История России", "Зачет с оценкой", "Хорошо"),
        Subject("Иностранный язык", "Зачет", "Зачтено"),
        Subject("Физическая культура", "Зачет", "Зачтено"),
    ],
    "Пограничный": [
        Subject("Математический анализ", "Экзамен", "Хорошо"),
        Subject("Программирование", "Экзамен", "Хорошо"),
        Subject("Дискретная математика", "Экзамен", "Хорошо", retake=True),
        Subject("История России", "Зачет с оценкой", "Хорошо"),
        Subject("Иностранный язык", "Зачет", "Зачтено"),
        Subject("Физическая культура", "Зачет", "Зачтено"),
    ],
    "С долгами": [
        Subject("Математический анализ", "Экзамен", "Неудовлетворительно"),
        Subject("Программирование", "Экзамен", "Удовлетворительно"),
        Subject("Дискретная математика", "Экзамен", "Хорошо"),
        Subject("История России", "Зачет с оценкой", "Хорошо"),
        Subject("Иностранный язык", "Зачет", "Не зачтено"),
        Subject("Физическая культура", "Зачет", "Зачтено"),
    ],
}


def subjects_to_records(subjects: list[Subject]) -> list[dict]:
    """Subject -> строки редактора (обратно к subjects_from_records)."""
    return [
        {"Дисциплина": s.name, "Вид контроля": s.control_type, "Оценка": s.grade, "Пересдача": bool(s.retake)}
        for s in subjects
    ]
