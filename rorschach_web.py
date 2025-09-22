# -*- coding: utf-8 -*-
"""
Streamlit Web App — Rorschach Test (Нейросеть + Байес + Анкета + Диагностика)
Запуск:
    streamlit run rorschach_web.py
"""

import os
import csv
import joblib
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
import time

# Пути
RESULTS_CSV = "results.csv"
CARDS_DIR = "cards"
TOTAL_SLIDES = 10

# -----------------------------
# Загрузка моделей
# -----------------------------
@st.cache_resource
def load_models():
    nn_model = load_model("nn_animal_model.h5")
    nn_vec = joblib.load("nn_vectorizer.pkl")
    bayes_model = joblib.load("bayes_model.pkl")
    bayes_vec = joblib.load("bayes_vectorizer.pkl")
    return nn_model, nn_vec, bayes_model, bayes_vec

nn_model, nn_vec, bayes_model, bayes_vec = load_models()

# -----------------------------
# Функции предсказания
# -----------------------------
def predict_is_animal(text: str):
    x_input = nn_vec.transform([text]).toarray()
    pred = nn_model.predict(x_input, verbose=0)[0][0]
    is_animal = int(pred > 0.5)
    return is_animal, float(pred)

def predict_popular(text: str):
    x_bayes = bayes_vec.transform([text])
    label = bayes_model.predict(x_bayes)[0]
    conf = float(np.max(bayes_model.predict_proba(x_bayes)))
    if conf < 0.5:
        label = "other"
    return label, conf

# -----------------------------
# CSV
# -----------------------------
def ensure_results_csv(path: str):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "username","slide","text",
                "is_animal","animal_conf",
                "popular_label","popular_confidence",
                "localization","movement","color_choice","form_clear"
            ])

ensure_results_csv(RESULTS_CSV)

# -----------------------------
# Диагностика
# -----------------------------
def get_diagnosis(scores):
    scoreA = scores["animals"]
    scoreM = scores["movement"]
    scoreC = scores["color"]
    scoreF = scores["form"]

    if scoreA > 30 and scoreM < 10 and scoreC < 20 and scoreF > 70:
        return "Здоровый тестируемый"
    elif scoreM > 40 and scoreC > 50:
        return "Шизофрения"
    elif scoreC > 60 and scoreF < 50:
        return "Алкогольный делирий"
    elif scoreA < 20 and scoreF < 40:
        return "Органические поражения мозга"
    else:
        return "Диагноз не определён"

def update_scores(is_animal, movement, color_choice, form_clear):
    st.session_state.answers["total"] += 1
    if is_animal:
        st.session_state.answers["animals"] += 1
    if movement:
        st.session_state.answers["movement"] += 1
    # Для color_choice: учитываем как "цвет важен", если выбраны 2 или 3 вариант
    if color_choice in ["Важнее цвет, форма вторична", "Важен только цвет"]:
        st.session_state.answers["color"] += 1
    if form_clear:
        st.session_state.answers["form"] += 1

def calc_percentages():
    total = st.session_state.answers["total"]
    if total == 0:
        return {k: 0 for k in st.session_state.answers}
    return {
        k: (v / total) * 100 if k != "total" else v
        for k, v in st.session_state.answers.items()
    }

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config("Rorschach Test", layout="centered")
st.title("Тест Роршаха (Web)")

# Имя пользователя
if "username" not in st.session_state:
    st.session_state.username = st.text_input("Введите имя пользователя:", value="anonymous")

# Слайд
if "slide" not in st.session_state:
    st.session_state.slide = 1

# Счётчики
if "answers" not in st.session_state:
    st.session_state.answers = {
        "total": 0,
        "animals": 0,
        "movement": 0,
        "color": 0,
        "form": 0,
    }

slide = st.session_state.slide
st.subheader(f"Карточка {slide} из {TOTAL_SLIDES}")

# Показ изображения
img_path = os.path.join(CARDS_DIR, f"{slide}.jpg")
if os.path.exists(img_path):
    st.image(img_path, caption=f"Карточка {slide}", use_container_width=True)
else:
    st.warning(f"Нет изображения {img_path}")

# Ввод ответа
text = st.text_area("Что вы видите?", "")

# Вопросы анкеты (как на скрине)
form_clear = st.radio("Увиденное имеет чёткую форму?", ["Да", "Нет"]) == "Да"
movement = st.radio("Изображение движется или искажается?", ["Да", "Нет"]) == "Да"

localization = st.radio(
    "В какой части таблицы Вы увидели образ?",
    [
        "Всё изображение или большая его часть",
        "Часть большого пятна",
        "Отдельные пятна или мелкие детали",
        "Белый фон",
        "Другое"
    ]
)

color_choice = st.radio(
    "На ваш ответ повлияли цвет или форма?",
    [
        "Важнее форма, цвет вторичен",
        "Важнее цвет, форма вторична",
        "Важен только цвет"
    ]
)

col1, col2 = st.columns(2)
with col1:
    skip = st.button("Отказаться от ответа")
with col2:
    submit = st.button("Отправить ответ")

# -----------------------------
# Обработка ответа
# -----------------------------
if submit or skip:
    if not text.strip() and not skip:
        st.error("Введите ответ или нажмите «Отказаться от ответа»")
    else:
        if skip:
            text = ""
            is_animal, animal_conf = 0, 0.0
            label, label_conf = "skipped", 0.0
        else:
            is_animal, animal_conf = predict_is_animal(text)
            label, label_conf = predict_popular(text)

            st.success(
                f"Нейросеть: {'Животное' if is_animal else 'Не животное'} "
                f"(conf={animal_conf:.2f}) | "
                f"Популярный ответ: {label} (conf={label_conf:.2f})"
            )

        with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                st.session_state.username, slide, text,
                is_animal, f"{animal_conf:.3f}",
                label, f"{label_conf:.3f}",
                localization, int(movement), color_choice, int(form_clear)
            ])

        update_scores(is_animal, movement, color_choice, form_clear)

        # Следующий слайд
        st.session_state.slide += 1
        if st.session_state.slide > TOTAL_SLIDES:
            st.success("✅ Тест завершён! Спасибо за участие.")

            scores = calc_percentages()
            diagnosis = get_diagnosis(scores)

            st.subheader("📊 Результаты:")
            st.write(f"Животные: {scores['animals']:.1f}%")
            st.write(f"Движение: {scores['movement']:.1f}%")
            st.write(f"Цвет: {scores['color']:.1f}%")
            st.write(f"Форма: {scores['form']:.1f}%")

            st.subheader("🧠 Предварительный диагноз:")
            st.info(diagnosis)

            st.session_state.slide = TOTAL_SLIDES
        else:
            time.sleep(2)
            st.rerun()
