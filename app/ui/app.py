"""
Прогноз стипендии - интерфейс.

Запуск из корня проекта:  streamlit run ui/app.py
"""

from __future__ import annotations

import copy
import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.cohort import apply_filters, by_plan, probability_histogram  # noqa: E402
from core.constants import (  # noqa: E402
    ALL_GRADE_LABELS,
    CONTROL_TYPES,
    MAX_INPUT_SEMESTER,
    MIN_SEMESTER,
    SEMESTER_NAMES,
)
from core.export import build_export  # noqa: E402
from core.features import Diagnostics, SchemaError, features_from_raw_with_diagnostics, read_raw_excel  # noqa: E402
from core.manual import (  # noqa: E402
    PRESETS,
    GradeCounts,
    ManualInputError,
    Subject,
    subjects_from_counts,
    subjects_from_json,
    subjects_from_raw,
    subjects_from_records,
    subjects_to_json,
    subjects_to_records,
)
from core.predictor import FEATURE_LABELS, Prediction, ScholarshipPredictor  # noqa: E402

st.set_page_config(
    page_title="Прогноз стипендии",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Токены темы «обложка зачетки» (дублируют .streamlit/config.toml для собственной разметки).
COVER, ENDPAPER, BRASS, PAPER, PENCIL, RULE = "#182440", "#223154", "#D4B26A", "#EDE7D8", "#9AA5B8", "#33456B"
BAND_COLORS = {"high": "#8ED08F", "mid": "#E7BE5C", "low": "#EB8A7E"}  # тушь штампов

st.markdown(
    f"""
<style>
h1 {{ color: {BRASS}; }}

/* результат - строка ведомости: линейки сверху и снизу, без карточки */
.verdict {{
  display: grid;
  grid-template-columns: 1fr;
  row-gap: .2rem;
  border-top: 1px solid {RULE};
  border-bottom: 1px solid {RULE};
  padding: 1.1rem 0 1.2rem;
  margin: .4rem 0 1.2rem;
}}
.verdict .num {{
  font-family: "PT Serif", Georgia, serif;
  font-size: 3.6rem;
  font-weight: 700;
  line-height: 1;
  color: {PAPER};
  letter-spacing: -.02em;
}}
.verdict .lead {{ font-size: 1.05rem; margin: .5rem 0 0; color: {PAPER}; max-width: 62ch; }}
.verdict .sub {{ color: {PENCIL}; margin: .6rem 0 0; font-size: .92rem; max-width: 62ch; }}

/* итог - спокойная плашка по центру, цвет туши только в тексте и рамке */
.verdict .tag {{
  justify-self: center;
  margin-top: 1.1rem;
  font-family: "PT Serif", Georgia, serif;
  font-size: 1rem;
  line-height: 1.3;
  text-align: center;
  color: var(--band);
  border: 1px solid var(--band);
  background: color-mix(in srgb, var(--band) 10%, transparent);
  border-radius: 4px;
  padding: .45rem 1rem;
}}

/* три семестра - одна строка таблицы, разделенная вертикальными линейками */
.timeline {{
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  margin: .6rem 0 .2rem;
  border-top: 1px solid {RULE};
  border-bottom: 1px solid {RULE};
}}
.timeline .cell {{ padding: .75rem 1rem .85rem; border-left: 1px solid {RULE}; }}
.timeline .cell:first-child {{ border-left: none; padding-left: 0; }}
.timeline .cell .sem {{ font-family: "PT Serif", Georgia, serif; font-weight: 700; font-size: 1.05rem; }}
.timeline .cell .role {{ font-size: .8rem; color: {PENCIL}; margin-top: .1rem; }}
.timeline .cell .val {{ font-size: .95rem; margin-top: .5rem; }}
.timeline .cell.now {{ background: {BRASS}; color: {COVER}; padding-left: 1rem; }}
.timeline .cell.now .role {{ color: {COVER}; opacity: .75; }}

.whatif {{ margin: .3rem 0; }}
.whatif .d-up {{ color: {BAND_COLORS["high"]}; font-weight: 600; }}
.whatif .d-down {{ color: {BAND_COLORS["low"]}; font-weight: 600; }}

@media (max-width: 640px) {{
  .timeline {{ grid-template-columns: 1fr; }}
  .timeline .cell {{ border-left: none; border-top: 1px solid {RULE}; padding-left: 0; }}
  .timeline .cell:first-child {{ border-top: none; }}
  .timeline .cell.now {{ padding-left: .75rem; }}
}}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Загружаю модель...")
def load_predictor() -> ScholarshipPredictor:
    return ScholarshipPredictor()


def pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def sem_ordinal(n: int) -> str:
    return f"{n}-й семестр"


def sem_loc(n: int) -> str:
    """Предложный падеж без предлога: «4-м семестре»."""
    return f"{n}-м семестре"


def v_sem(n: int) -> str:
    """«в 3-м семестре», но «во 2-м семестре»."""
    return ("во " if n == 2 else "в ") + sem_loc(n)


def sem_gen(n: int) -> str:
    """Родительный падеж: «с 4-го семестра»."""
    return f"{n}-го семестра"


# --- компоненты ---------------------------------------------------------------

def render_verdict(p: Prediction) -> None:
    key, _ = p.band
    label = p.verdict
    color = BAND_COLORS[key]
    target = sem_ordinal(p.target_sem)

    if p.current_clean:
        outcome = f"сохранит стипендию на {sem_ordinal(p.stipend_sem)}"
        status = (
            f"Сейчас: {sem_ordinal(p.sem_num)} закрыт без троек, долгов и пересдач, "
            f"стипендия {v_sem(p.target_sem)} есть."
        )
        cell_now = f"без троек, долгов и пересдач: стипендия {v_sem(p.target_sem)} есть"
    else:
        outcome = f"получит стипендию с {sem_gen(p.stipend_sem)}"
        status = (
            f"Сейчас: {v_sem(p.sem_num)} есть тройки, долги или пересдачи, "
            f"стипендии {v_sem(p.target_sem)} не будет."
        )
        cell_now = f"есть тройки, долги или пересдачи: стипендии {v_sem(p.target_sem)} нет"

    st.markdown(
        f"""
<div class="verdict" style="--band:{color}">
  <div>
    <div class="num">{pct(p.prob_clean_next)}</div>
    <p class="lead">с такой вероятностью студент закроет {target} без троек, долгов и пересдач
    и {outcome}.</p>
    <p class="sub">{status} Стипендия {v_sem(p.stipend_sem)} зависит от того, как пройдет {target}.</p>
  </div>
  <div class="tag">{label}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
<div class="timeline">
  <div class="cell">
    <div class="sem">{sem_ordinal(p.sem_num)}</div>
    <div class="role">факт</div>
    <div class="val">{cell_now}</div>
  </div>
  <div class="cell now">
    <div class="sem">{target}</div>
    <div class="role">оценка модели</div>
    <div class="val">без троек, долгов и пересдач - {pct(p.prob_clean_next)}</div>
  </div>
  <div class="cell">
    <div class="sem">{sem_ordinal(p.stipend_sem)}</div>
    <div class="role">следствие</div>
    <div class="val">стипендия - {pct(p.prob_clean_next)}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_prediction_notes(p: Prediction) -> None:
    """Оговорки, при которых оценке стоит доверять меньше."""
    for note in p.range_notes:
        st.warning(note)
    if p.incomplete:
        st.info(
            "В семестре всего одна-две дисциплины. Модель обучалась на полных семестрах, "
            "поэтому к оценке стоит относиться осторожно."
        )
    if p.sem_num == 8:
        st.info(
            "Для бакалавриата 8-й семестр - последний: следующего семестра и стипендии в нем не будет. "
            "Оценка имеет смысл, только если обучение продолжается (специалитет)."
        )


def render_summary(p: Prediction, full: bool = False) -> None:
    f = p.features

    def fmt(name: str):
        v = f.get(name)
        if v is None:
            return "—"
        if name in ("any_block", "any_retake", "had_clean_current_sem"):
            return "да" if v else "нет"
        if name in ("share_5", "share_3", "share_zachet"):
            return pct(v)
        if name in ("gpa_overall", "std_grade"):
            return f"{v:.2f}"
        return f"{int(v)}"

    names = (
        list(FEATURE_LABELS)
        if full
        else ["gpa_overall", "n_subjects", "n_blocks", "n_retakes", "share_5", "share_3", "share_zachet"]
    )
    rows = [(FEATURE_LABELS[n], fmt(n)) for n in names]
    st.dataframe(
        pd.DataFrame(rows, columns=["Показатель", "Значение"]),
        hide_index=True,
        width="stretch",
    )


def render_what_if(rows: list[dict], base: float) -> None:
    if not rows:
        st.caption("Исправлять нечего: в семестре нет пересдач, троек и долгов.")
        return
    for r in rows:
        cls = "d-up" if r["delta"] >= 0 else "d-down"
        sign = "+" if r["delta"] >= 0 else "−"
        st.markdown(
            f'<div class="whatif">{r["scenario"]}: {pct(base)} → <b>{pct(r["prob"])}</b> '
            f'<span class="{cls}">({sign}{abs(r["delta"]) * 100:.0f} п.п.)</span></div>',
            unsafe_allow_html=True,
        )


# --- режимы -------------------------------------------------------------------

# --- редактор оценок: состояние, переживающее смену страницы ----------------------
# Streamlit сбрасывает виджеты, которых нет на текущей странице. Поэтому содержимое
# редактора хранится отдельно (manual_saved), а сам виджет получает новый ключ
# каждый раз, когда данные подменяются извне (пример, JSON, студент из файла).

EDITOR_COLUMNS = ["Дисциплина", "Вид контроля", "Оценка", "Пересдача"]
DEFAULT_EXAMPLE = [
    Subject("Математический анализ", "Экзамен", "Хорошо"),
    Subject("Программирование", "Экзамен", "Отлично"),
    Subject("История России", "Зачет с оценкой", "Удовлетворительно"),
    Subject("Иностранный язык", "Зачет", "Зачтено"),
    Subject("Физическая культура", "Зачет", "Зачтено"),
]


def rows_frame(subjects: list[Subject]) -> pd.DataFrame:
    return pd.DataFrame(subjects_to_records(subjects), columns=EDITOR_COLUMNS)


def replace_editor_rows(subjects: list[Subject], sem_num: int | None = None, notes: list[str] | None = None) -> None:
    """Подменить содержимое редактора. Новый ключ виджета - чтобы старые правки не наложились."""
    st.session_state.manual_saved = rows_frame(subjects)
    st.session_state.editor_version = st.session_state.get("editor_version", 0) + 1
    if sem_num is not None:
        st.session_state.manual_sem_saved = int(sem_num)
    st.session_state.editor_notes = notes or []


def editor_key_and_base() -> tuple[str, pd.DataFrame]:
    version = st.session_state.setdefault("editor_version", 0)
    key = f"manual_editor_{version}"
    if key not in st.session_state:  # виджет создается заново: первый визит, возврат на страницу или подмена
        st.session_state.manual_base = st.session_state.get("manual_saved", rows_frame(DEFAULT_EXAMPLE))
    return key, st.session_state.manual_base


# --- страница «Оценки» ----------------------------------------------------------------

def grades_view(predictor: ScholarshipPredictor) -> None:
    st.title("Будет ли стипендия через семестр")
    st.subheader("Оценки за текущий семестр")
    st.caption(
        "Заполните таблицу так, как это выглядит в зачетке. Строки добавляются кнопкой под таблицей, "
        "пустые строки не учитываются."
    )

    options = list(range(MIN_SEMESTER, MAX_INPUT_SEMESTER + 1))
    saved_sem = int(st.session_state.get("manual_sem_saved", 2))
    sem_num = st.selectbox("Какой семестр закончился", options=options, index=options.index(saved_sem), format_func=sem_ordinal)
    st.session_state.manual_sem_saved = int(sem_num)

    for note in st.session_state.pop("editor_notes", []):
        st.caption(note)

    key, base = editor_key_and_base()
    edited = st.data_editor(
        base,
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        column_config={
            "Дисциплина": st.column_config.TextColumn("Дисциплина", required=True, width="large"),
            "Вид контроля": st.column_config.SelectboxColumn("Вид контроля", options=CONTROL_TYPES, required=True),
            "Оценка": st.column_config.SelectboxColumn("Оценка", options=ALL_GRADE_LABELS, required=True),
            "Пересдача": st.column_config.CheckboxColumn("Была пересдача", default=False),
        },
        key=key,
    )
    st.session_state.manual_saved = edited

    with st.expander("Примеры, сохранение и загрузка"):
        cols = st.columns(len(PRESETS))
        for col, (name, subjects) in zip(cols, PRESETS.items(), strict=True):
            if col.button(name, key=f"preset_{name}"):
                replace_editor_rows(subjects)
                st.rerun()
        left, right = st.columns(2)
        with left:
            current = subjects_from_records(edited.to_dict("records"))
            st.download_button(
                "Сохранить семестр (json)",
                data=subjects_to_json(current, sem_num).encode("utf-8"),
                file_name=f"semester_{sem_num}.json",
                mime="application/json",
                on_click="ignore",
                disabled=not current,
            )
        with right:
            loaded = st.file_uploader("Загрузить семестр (json)", type=["json"], key="json_upload", label_visibility="collapsed")
            if loaded is not None:
                token = (loaded.name, loaded.size)
                if st.session_state.get("json_loaded") != token:
                    try:
                        subjects, loaded_sem = subjects_from_json(loaded.getvalue().decode("utf-8"))
                    except ManualInputError as e:
                        st.warning(str(e))
                    else:
                        st.session_state.json_loaded = token
                        replace_editor_rows(subjects, loaded_sem, [f"Загружен семестр из {loaded.name}"])
                        st.rerun()

    if st.button("Рассчитать прогноз", type="primary"):
        try:
            subjects = subjects_from_records(edited.to_dict("records"))
            prediction = predictor.predict_subjects(subjects, sem_num)
            what_if = predictor.what_if(subjects, sem_num, base=prediction)
        except ManualInputError as e:
            st.warning(str(e))
            return
        render_prediction(prediction, what_if, "Как модель увидела семестр")


def render_prediction(prediction: Prediction, what_if: list[dict], summary_title: str, full: bool = False) -> None:
    st.subheader("Прогноз")
    render_verdict(prediction)
    render_prediction_notes(prediction)
    left, right = st.columns(2)
    with left:
        st.markdown(f"**{summary_title}**")
        render_summary(prediction, full=full)
    with right:
        st.markdown("**Что изменило бы прогноз**")
        render_what_if(what_if, prediction.prob_clean_next)


# --- страница «Файл» -------------------------------------------------------------------

@st.cache_data(show_spinner="Читаю файл...")
def raw_from_upload(data: bytes) -> pd.DataFrame:
    return read_raw_excel(io.BytesIO(data))


@st.cache_data(show_spinner="Считаю признаки...")
def features_from_upload(data: bytes) -> tuple[pd.DataFrame, Diagnostics]:
    """Файл разбирается один раз на содержимое; смена семестра его не перечитывает."""
    return features_from_raw_with_diagnostics(raw_from_upload(data))


def render_diagnostics(diag: Diagnostics) -> None:
    with st.expander("Что осталось за кадром", expanded=bool(diag.grades_unmapped)):
        st.caption(
            "Модель обучалась на очной форме и на семестрах с 1-го по 10-й, поэтому остальные строки "
            "выгрузки не учитываются. Ниже - сколько и чего отфильтровано."
        )
        st.dataframe(pd.DataFrame(diag.rows_table(), columns=["Показатель", "Значение"]), hide_index=True, width="stretch")
        labels = diag.labels_table()
        if labels:
            st.markdown("**Значения, которые не были распознаны или не учитываются**")
            st.dataframe(pd.DataFrame(labels, columns=["Где", "Значение", "Строк"]), hide_index=True, width="stretch")


def result_table(result: pd.DataFrame) -> pd.DataFrame:
    """Строки результата -> таблица для экрана и экспорта."""
    columns = {
        "ЗачетнаяКнижка": "Зачетная книжка",
        "sem_num": "Семестр",
        "УровеньПодготовки": "Уровень",
        "УчебныйПлан": "Учебный план",
        "gpa_overall": "Средний балл",
        "n_blocks": "Троек и долгов",
        "n_retakes": "Пересдач",
        "had_clean_current_sem": "Стипендия сейчас",
        "prob_clean_next_sem": "Шанс на стипендию через семестр",
        "verdict_label": "Итог",
        "maybe_incomplete": "Семестр неполный?",
        "out_of_range": "За пределами данных?",
    }
    present = [c for c in columns if c in result.columns]
    sort_by = [c for c in ("maybe_incomplete", "prob_clean_next_sem") if c in result.columns]
    table = result.sort_values(sort_by)[present].rename(columns=columns)
    table["Стипендия сейчас"] = table["Стипендия сейчас"].map({1: "есть", 0: "нет"})
    if "Семестр неполный?" in table.columns:
        table["Семестр неполный?"] = table["Семестр неполный?"].map({True: "похоже", False: ""})
    if "За пределами данных?" in table.columns:
        table["За пределами данных?"] = table["За пределами данных?"].map({True: "да", False: ""})
    return table


def render_student_detail(
    predictor: ScholarshipPredictor, result: pd.DataFrame, raw: pd.DataFrame | None, student_id, sem_num: int
) -> None:
    """Экран одного студента из таблицы: тот же результат, что в строке, плюс «что если» по его строкам."""
    row = result[(result["ЗачетнаяКнижка"] == student_id) & (result["sem_num"] == sem_num)]
    if row.empty:
        st.warning("Строка не найдена - таблица изменилась, выберите студента еще раз.")
        return
    prediction = predictor.prediction_from_row(row.iloc[0])
    st.subheader(f"Студент {student_id}, {sem_ordinal(sem_num)}")

    rows = None
    if raw is not None:
        rows = raw[(raw["ЗачетнаяКнижка"] == student_id) & (raw["ПериодКонтроля"] == SEMESTER_NAMES[sem_num])]
    what_if = predictor.what_if_rows(rows, sem_num, base=prediction) if rows is not None and len(rows) else []

    render_verdict(prediction)
    render_prediction_notes(prediction)
    left, right = st.columns(2)
    with left:
        st.markdown("**Как модель увидела семестр**")
        render_summary(prediction)
    with right:
        st.markdown("**Что изменило бы прогноз**")
        render_what_if(what_if, prediction.prob_clean_next)
        if rows is not None and len(rows) and st.button("Открыть в редакторе оценок"):
            subjects, notes = subjects_from_raw(rows)
            if not subjects:
                st.warning("Не удалось перенести ни одной дисциплины: " + "; ".join(notes))
            else:
                replace_editor_rows(subjects, sem_num, [f"Студент {student_id}, {sem_ordinal(sem_num)} - перенесено из файла", *notes])
                st.switch_page(GRADES_PAGE)


def render_batch_result(
    predictor: ScholarshipPredictor,
    result: pd.DataFrame,
    diag: Diagnostics | None = None,
    semester: int | None = None,
    raw: pd.DataFrame | None = None,
) -> None:
    st.subheader("Результат")

    # фильтры по группе
    levels = plans = None
    if "УровеньПодготовки" in result.columns or "УчебныйПлан" in result.columns:
        f1, f2 = st.columns([1, 2])
        if "УровеньПодготовки" in result.columns:
            levels = f1.multiselect("Уровень подготовки", sorted(result["УровеньПодготовки"].dropna().unique()), placeholder="все")
        if "УчебныйПлан" in result.columns:
            plans = f2.multiselect("Учебный план", sorted(result["УчебныйПлан"].dropna().unique()), placeholder="все планы")
    shown = apply_filters(result, levels, plans)
    if shown.empty:
        st.warning("Под выбранные фильтры не попал ни один студент.")
        return

    counts = shown["risk_key"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Студентов", f"{len(shown):,}".replace(",", " "))
    c2.metric("Скорее всего со стипендией", int(counts.get("high", 0)))
    c3.metric("Под вопросом", int(counts.get("mid", 0)))
    c4.metric("Скорее всего без стипендии", int(counts.get("low", 0)))
    st.caption("Речь о стипендии через семестр: она зависит от того, как студент закроет следующий.")

    if diag is not None:
        if diag.grades_unmapped:
            n = sum(diag.grades_unmapped.values())
            st.warning(
                f"В {n} строках итоговая отметка не распознана и не учтена ни как оценка, ни как долг - "
                f"для этих студентов оценка модели может быть завышена. Список значений - ниже, в разделе "
                f"«Что осталось за кадром»."
            )
        dropped_students = diag.students_in_file - diag.students_kept
        if dropped_students:
            st.info(
                f"{dropped_students} студентов из файла не учтены: не очная форма обучения или период не "
                f"распознан как семестр. Подробности - в разделе «Что осталось за кадром»."
            )
    out_of_range = int(shown["out_of_range"].sum()) if "out_of_range" in shown.columns else 0
    if out_of_range:
        st.warning(
            f"У {out_of_range} студентов семестр не похож на обучающие данные (слишком много дисциплин или долгов, "
            f"либо все оценки - одинаковые двойки): для них число в таблице получено экстраполяцией. "
            f"Они помечены в колонке «За пределами данных?»."
        )
    incomplete = int(shown["maybe_incomplete"].sum()) if "maybe_incomplete" in shown.columns else 0
    if incomplete:
        st.info(
            f"У {incomplete} студентов в последнем семестре заметно меньше дисциплин, чем обычно - похоже на "
            f"незакрытую сессию. Их строки помечены и показаны в конце таблицы."
        )

    st.bar_chart(probability_histogram(shown), height=180)
    st.caption("Сколько студентов приходится на каждый интервал оценки.")

    table = result_table(shown)
    event = st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key="result_table",
        column_config={
            "Зачетная книжка": st.column_config.NumberColumn(format="%d"),
            "Учебный план": st.column_config.NumberColumn(format="%d"),
            "Средний балл": st.column_config.NumberColumn(format="%.2f"),
            "Шанс на стипендию через семестр": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
        },
    )
    st.caption("Щелкните по строке, чтобы открыть студента.")

    plans_table = by_plan(shown)
    if len(plans_table):
        with st.expander("По учебным планам"):
            st.dataframe(
                plans_table,
                hide_index=True,
                width="stretch",
                column_config={
                    "Учебный план": st.column_config.NumberColumn(format="%d"),
                    "Средняя оценка": st.column_config.NumberColumn(format="%.2f"),
                    "Со стипендией": st.column_config.NumberColumn(format="percent"),
                    "Под вопросом": st.column_config.NumberColumn(format="percent"),
                    "Без стипендии": st.column_config.NumberColumn(format="percent"),
                },
            )
            st.caption("Планы, в которых не меньше пяти студентов; отсортированы от самой низкой средней оценки.")

    if diag is not None:
        render_diagnostics(diag)

    export = build_export(table, diag, predictor, semester)
    st.download_button(
        "Скачать таблицу (xlsx: прогноз, о модели, диагностика)",
        data=export,
        file_name="scholarship_forecast.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        on_click="ignore",
    )

    selected = event.selection.rows if event is not None and hasattr(event, "selection") else []
    if selected:
        picked = table.iloc[selected[0]]
        st.divider()
        render_student_detail(predictor, result, raw, picked["Зачетная книжка"], int(picked["Семестр"]))


def current_upload() -> dict | None:
    """Файл из виджета или из памяти сессии: виджет сбрасывается при смене страницы, память - нет."""
    upload = st.file_uploader("Выберите xlsx", type=["xlsx"], key="upload", label_visibility="collapsed")
    if upload is not None:
        store = {"name": upload.name, "size": upload.size, "data": upload.getvalue()}
        if st.session_state.get("upload_store", {}).get("name") != store["name"] or st.session_state.get("upload_store", {}).get("size") != store["size"]:
            st.session_state.pop("batch", None)
        st.session_state.upload_store = store
        return store
    store = st.session_state.get("upload_store")
    if store:
        c1, c2 = st.columns([4, 1])
        c1.caption(f"Используется загруженный ранее файл: {store['name']} ({store['size'] / 1e6:.1f} МБ). Выберите другой, чтобы заменить.")
        if c2.button("Убрать файл"):
            st.session_state.pop("upload_store", None)
            st.session_state.pop("batch", None)
            st.rerun()
    return store


def file_view(predictor: ScholarshipPredictor) -> None:
    st.title("Прогноз по выгрузке ведомостей")
    st.caption("Формат ГОСТ Р 70946-2023, Приложение 8 (лист Sheet1).")
    store = current_upload()
    if store is None:
        st.info("Загрузите выгрузку ведомостей, чтобы получить прогноз по всем студентам.")
        return

    try:
        feats, diag = features_from_upload(store["data"])
        raw = raw_from_upload(store["data"])
    except SchemaError as e:
        st.error(str(e))
        return
    except Exception as e:  # noqa: BLE001
        st.error(f"Не удалось прочитать файл: {e}")
        return

    scope = st.radio(
        "Какой семестр брать за текущий",
        ["Последний семестр каждого студента", "Конкретный семестр"],
        horizontal=True,
    )
    semester = None
    if scope == "Конкретный семестр":
        semester = st.selectbox(
            "Семестр", options=list(range(MIN_SEMESTER, MAX_INPUT_SEMESTER + 1)), index=1, format_func=sem_ordinal
        )

    # результат живет в session_state, чтобы пережить перезапуск скрипта,
    # но показывается только для того же файла и того же выбора семестра
    key = (store["name"], store["size"], semester)
    if st.button("Рассчитать прогноз", type="primary"):
        run_diag = copy.deepcopy(diag)  # выбор семестра дописывает свои счетчики
        result = predictor.predict_features(predictor.select_semester(feats, semester, run_diag))
        st.session_state.batch = (key, result, run_diag)

    stored = st.session_state.get("batch")
    if not stored or stored[0] != key:
        return
    _, result, run_diag = stored
    if result.empty:
        st.warning(
            "После фильтров не осталось ни одного студента. Проверьте форму обучения и выбранный семестр - "
            "подробности ниже."
        )
        render_diagnostics(run_diag)
        return
    render_batch_result(predictor, result, run_diag, semester, raw)


# --- страница «Количества» ------------------------------------------------------------

def counts_view(predictor: ScholarshipPredictor) -> None:
    st.title("Семестр по количествам оценок")
    st.caption(
        "Без названий дисциплин: укажите, сколько каких оценок получено. Из этих чисел собирается семестр "
        "и проходит тот же конвейер признаков, что и файл."
    )

    with st.form("counts"):
        sem_num = st.selectbox(
            "Какой семестр закончился",
            options=list(range(MIN_SEMESTER, MAX_INPUT_SEMESTER + 1)),
            index=1,
            format_func=sem_ordinal,
        )
        st.markdown("**Экзамены и зачеты с оценкой**")
        e = st.columns(5)
        excellent = e[0].number_input("Отлично", 0, 30, 3)
        good = e[1].number_input("Хорошо", 0, 30, 3)
        satisfactory = e[2].number_input("Удовлетворительно", 0, 30, 0)
        fail = e[3].number_input("Неудовлетворительно", 0, 30, 0)
        absent_exam = e[4].number_input("Неявка", 0, 30, 0, key="absent_exam", help="Входит в средний балл как двойка.")
        st.markdown("**Зачеты**")
        z = st.columns(5)
        passed = z[0].number_input("Зачтено", 0, 30, 2)
        not_passed = z[1].number_input("Не зачтено", 0, 30, 0)
        absent_pass = z[2].number_input("Неявка", 0, 30, 0, key="absent_pass", help="В средний балл не входит.")
        st.markdown("**Пересдачи**")
        extra_retakes = st.number_input(
            "Дисциплин с пересдачей, кроме неявок",
            0,
            60,
            0,
            help="Неявка считается пересдачей автоматически. Дополнительные пересдачи получают дисциплины "
            "с худшими оценками; на признаки модели это распределение не влияет.",
        )
        submitted = st.form_submit_button("Рассчитать прогноз", type="primary")

    if not submitted:
        return

    counts = GradeCounts(
        excellent=int(excellent), good=int(good), satisfactory=int(satisfactory), fail=int(fail),
        absent_exam=int(absent_exam), passed=int(passed), not_passed=int(not_passed),
        absent_pass=int(absent_pass), extra_retakes=int(extra_retakes),
    )
    try:
        subjects = subjects_from_counts(counts)
        prediction = predictor.predict_subjects(subjects, sem_num)
        what_if = predictor.what_if(subjects, sem_num, base=prediction)
    except ManualInputError as e:
        st.warning(str(e))
        return
    render_prediction(prediction, what_if, "Признаки, посчитанные из количеств", full=True)
    st.caption(f"Собрано дисциплин: {counts.total}, из них с оценкой {counts.total - counts.passed - counts.not_passed - counts.absent_pass}.")


# --- общая боковая панель и страницы ------------------------------------------------------

def render_sidebar(predictor: ScholarshipPredictor) -> None:
    with st.sidebar:
        m = predictor.metrics
        persistence = predictor.baselines.get("persistence", {}).get("accuracy")
        n_train = f"{predictor.n_train:,}".replace(",", " ")
        st.markdown("**О модели**")
        baseline_txt = (
            f" против {persistence * 100:.1f}% у правила «следующий семестр как текущий»" if persistence else ""
        )
        st.markdown(
            f"HistGradientBoosting, обучена на {n_train} парах семестров. "
            f"Точность {m.get('accuracy', 0) * 100:.1f}%{baseline_txt}; ROC-AUC {m.get('roc_auc', 0):.3f} - "
            f"основная ценность модели в ранжировании риска, а не в бинарном ответе."
        )
        st.caption(
            "Модель смотрит на оценки семестра N и оценивает, закроет ли студент семестр N+1 "
            "без троек, долгов и пересдач - то есть будет ли у него стипендия в N+2. "
            "Стипендия в N+1 уже известна по правилам и не предсказывается. "
            "Число на экране - оценка модели, обученной с балансировкой классов; в среднем она немного оптимистична."
        )
        for w in predictor.load_warnings:
            st.warning(w)


def page_grades() -> None:
    predictor = load_predictor()
    render_sidebar(predictor)
    grades_view(predictor)


def page_file() -> None:
    predictor = load_predictor()
    render_sidebar(predictor)
    file_view(predictor)


def page_counts() -> None:
    predictor = load_predictor()
    render_sidebar(predictor)
    counts_view(predictor)


GRADES_PAGE = st.Page(page_grades, title="Оценки", url_path="grades", default=True)
FILE_PAGE = st.Page(page_file, title="Файл", url_path="file")
COUNTS_PAGE = st.Page(page_counts, title="Количества", url_path="counts")


def main() -> None:
    st.navigation([GRADES_PAGE, FILE_PAGE, COUNTS_PAGE]).run()


if __name__ == "__main__":
    main()
