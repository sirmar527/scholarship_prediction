"""
Подготовка данных для прогноза стипендии.

Загружает сырые записи по оценкам, нормализует периоды контроля и предоставляет
утилиты для построения аналитических подвыборок:
    - Модель A (раннее предупреждение): все очники с нужными парами семестров.
    - Модель B (переносимость): только студенты, завершившие программу.

ВАЖНО про периоды контроля:
    В датасете встречаются два набора названий:
        * "Первый/Второй/... семестр" — очники и специалисты (стандартная сетка).
        * "Первая зимняя/летняя сессия" и т.п. — СЕССИИ ЗАОЧНИКОВ, а не пересдачи.
          Проверка по данным: пересечений с основной ведомостью по дисциплинам
          практически нет (7 случаев на 37 тыс.), 98% оценок проходные.
    Академическая стипендия в РФ — только для бюджетников очной формы, поэтому
    заочников исключаем явным фильтром.

Использование:
    Положите .xlsx с данными в ту же папку, что этот скрипт, либо передайте
    путь к файлу явно:  df = load_raw(path='путь/к/файлу.xlsx')
"""
from pathlib import Path
import pandas as pd


# Шаблоны имени файла — покрывают варианты с подчёркиваниями и пробелами
DATA_FILE_PATTERNS = [
    'ГОСТ_Р_70946-2023-Приложение-8_sorted.xlsx',
    'ГОСТ Р 70946-2023-Приложение-8_sorted.xlsx',
    'ГОСТ*Приложение*.xlsx',  # fallback через glob
]


def find_data_file(directory: Path | None = None) -> Path:
    """Ищет файл с данными рядом со скриптом (или в указанной папке).

    Пробует сначала точные имена, потом glob-шаблон. Это нужно, чтобы скрипт
    работал вне зависимости от того, с какими символами сохранилось имя
    (подчёркивания vs пробелы) и кроссплатформенно (Windows/Linux/macOS).
    """
    if directory is None:
        directory = Path(__file__).parent
    directory = Path(directory)

    # Точные имена
    for name in DATA_FILE_PATTERNS:
        if '*' in name:
            continue
        candidate = directory / name
        if candidate.exists():
            return candidate

    # Glob-fallback
    for pattern in DATA_FILE_PATTERNS:
        if '*' not in pattern:
            continue
        matches = list(directory.glob(pattern))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f'Не нашёл .xlsx с данными в {directory}.\n'
        f'Положите файл рядом со скриптом или передайте путь явно:\n'
        f'    df = load_raw(path="путь/к/файлу.xlsx")'
    )


# Стандартные семестры очной/специалитет сетки (1..10)
SEMESTER_MAP = {
    'Первый семестр': 1,
    'Второй семестр': 2,
    'Третий семестр': 3,
    'Четвертый семестр': 4,
    'Пятый семестр': 5,
    'Шестой семестр': 6,
    'Седьмой семестр': 7,
    'Восьмой семестр': 8,
    'Девятый семестр': 9,
    'Десятый семестр': 10,
}

# Сессии заочной формы обучения (не пересдачи — проверено по данным)
PARTTIME_SESSIONS = {
    'Первая зимняя сессия', 'Первая летняя сессия',
    'Вторая зимняя сессия', 'Вторая летняя сессия',
    'Третья зимняя сессия', 'Третья летняя сессия',
    'Четвертая зимняя сессия', 'Четвертая летняя сессия',
    'Пятая зимняя сессия', 'Пятая летняя сессия',
    'Пятая установочная сессия',
}

PROGRAM_LENGTH = {'Бакалавриат': 8, 'Специалитет': 10}


def load_raw(path: Path | str | None = None) -> pd.DataFrame:
    """Загружает .xlsx в DataFrame без модификаций.

    Если path не указан — ищет файл с данными рядом со скриптом.
    """
    if path is None:
        path = find_data_file()
    path = Path(path)
    df = pd.read_excel(path, sheet_name='Sheet1')
    df['ЗачетнаяКнижка'] = df['ЗачетнаяКнижка'].astype(int)
    df['УчебныйПлан'] = df['УчебныйПлан'].astype(int)
    return df


def add_semester_info(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет sem_num (1..10 или NaN) и period_type ('regular'/'parttime'/'other')."""
    df = df.copy()
    df['sem_num'] = df['ПериодКонтроля'].map(SEMESTER_MAP).astype('Int64')
    df['period_type'] = 'other'
    df.loc[df['ПериодКонтроля'].isin(SEMESTER_MAP), 'period_type'] = 'regular'
    df.loc[df['ПериодКонтроля'].isin(PARTTIME_SESSIONS), 'period_type'] = 'parttime'
    return df


def filter_main_sheets(df: pd.DataFrame) -> pd.DataFrame:
    """Оставляет только строки основной ведомости (ТипВедомости == 'Основная')."""
    return df[df['ТипВедомости'] == 'Основная'].copy()


def filter_fulltime(df: pd.DataFrame) -> pd.DataFrame:
    """Оставляет только очников — единственная форма, получающая академическую стипендию."""
    return df[df['ФормаОбучения'] == 'Очная'].copy()


def prepare(path: Path | str | None = None) -> pd.DataFrame:
    """Стандартный пайплайн: загрузка → разметка периодов → очная → основная ведомость."""
    df = load_raw(path=path)
    df = add_semester_info(df)
    df = filter_fulltime(df)
    df = filter_main_sheets(df)
    return df


def get_completed_students(df: pd.DataFrame) -> set:
    """Студенты, у которых есть записи в последнем семестре своей программы."""
    reg = df[df['period_type'] == 'regular']
    per_student = reg.groupby('ЗачетнаяКнижка').agg(
        max_sem=('sem_num', 'max'),
        level=('УровеньПодготовки', 'first'),
    )
    completed = set()
    for sid, row in per_student.iterrows():
        req = PROGRAM_LENGTH.get(row['level'])
        if req is not None and row['max_sem'] is not pd.NA and row['max_sem'] >= req:
            completed.add(sid)
    return completed


def get_transition_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Для каждого студента возвращает доступные пары (from_sem, to_sem = N, N+1)."""
    reg = df[df['period_type'] == 'regular']
    student_sems = (
        reg.groupby('ЗачетнаяКнижка')['sem_num']
        .apply(lambda s: set(int(x) for x in s.dropna()))
    )
    rows = []
    for sid, sems in student_sems.items():
        for n in range(1, 10):
            if n in sems and (n + 1) in sems:
                rows.append((sid, n, n + 1))
    return pd.DataFrame(rows, columns=['ЗачетнаяКнижка', 'from_sem', 'to_sem'])


def summarize(df: pd.DataFrame) -> None:
    """Краткий отчёт о составе датасета после подготовки."""
    reg = df[df['period_type'] == 'regular']
    print(f"Всего строк:                  {len(df):>10,}")
    print(f"  в т.ч. regular-семестры:    {len(reg):>10,}")
    print(f"Уникальных студентов:         {df['ЗачетнаяКнижка'].nunique():>10,}")
    print(f"Уникальных учебных планов:    {df['УчебныйПлан'].nunique():>10,}")
    print()

    completed = get_completed_students(df)
    print(f"Завершивших программу:        {len(completed):>10,}")
    by_level = reg[reg['ЗачетнаяКнижка'].isin(completed)].groupby('УровеньПодготовки')['ЗачетнаяКнижка'].nunique()
    for level, cnt in by_level.items():
        print(f"  {level:<28}{cnt:>10,}")
    print()

    pairs = get_transition_pairs(df)
    print(f"Всего пар (сем N, сем N+1):   {len(pairs):>10,}")
    print("  распределение по from_sem → to_sem:")
    print(pairs.groupby(['from_sem', 'to_sem']).size().to_string())


if __name__ == '__main__':
    df = prepare()
    summarize(df)
