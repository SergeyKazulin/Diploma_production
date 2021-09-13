import numpy as np
import pandas as pd
from datetime import date
from tensorflow.keras.models import load_model
import streamlit as st
from streamlit import *
from sklearn.preprocessing import normalize
from tensorflow.python.keras.backend import reverse


st.write('Дата: ', date.today())

st.title('Возможное заболевание сердца')

st.sidebar.header('Введите параметры')

#heart = pd.read_csv('C:/programming/samples/heart.csv')
heart = pd.read_csv('e:/diploma/heart_diploma.csv')
new_columns = [1, 0, 3, 7, 6, 10, 4, 5, 2, 8, 9, 11]
heart = heart[heart.columns[new_columns]]


def user_input_features():    
    sex = st.sidebar.selectbox('Выбирете пол: ', heart['sex'].unique())
    if sex == 1:
        st.sidebar.markdown('Пол: мужской')
    if sex == 0:
        st.sidebar.markdown('Пол: женский')
    
    age = st.sidebar.slider('Возраст: ', min_value=0, max_value=100, value=1 , step=1, format=None)
    st.sidebar.markdown(f'Возраст: {age} лет')

    resting_bp_s = st.sidebar.slider('Давление SYS: ', min_value=50, max_value=250, value=None , step=1, format=None)
    st.sidebar.markdown(f'SYS давление: {resting_bp_s} mmHg')

    max_heart_rate = st.sidebar.slider('Пульс: ', min_value=0, max_value=220, value=None , step=1, format=None)
    st.sidebar.markdown(f'Максимальный Пульс: {max_heart_rate} b/min')
    
    resting_ecg = st.sidebar.selectbox('Кардиограмма в покое: ', heart['resting ecg'].unique())
    if resting_ecg == 0:
        st.sidebar.markdown('Кардиограмма хорошая')
    if resting_ecg == 1:
        st.sidebar.markdown('Кардиогамма нормальная')
    if resting_ecg == 2:
        st.sidebar.markdown('Кардиограмма плохая')
    
    st_slope = st.sidebar.selectbox('Наклон сегмента кардиограммы ST: ', heart['ST slope'].unique())
    if st_slope == 1:
        st.sidebar.markdown('Наклон восходящий')
    if st_slope == 2:
        st.sidebar.markdown('Наклон плоский')
    if st_slope == 3:
        st.sidebar.markdown('Наклон низходящий')
    if st_slope == 0:
        st.sidebar.markdown('Наклон очень большой')

    fasting_blood_sugar = st.sidebar.selectbox('Уровень сахара : ', heart['fasting blood sugar'].unique())
    if fasting_blood_sugar == 0:
        st.sidebar.markdown('Содержание сахара меньше 120 mg/dl')
    if fasting_blood_sugar == 1:
        st.sidebar.markdown('Содержание сахара больше 120 mg/dl')

    cholesterol = st.sidebar.slider('Холестерин: ', min_value=50, max_value=500, value=None , step=1, format=None)
    st.sidebar.markdown(f'Холестерин: {cholesterol} mg/dl')

    chest_pain_type = st.sidebar.selectbox('Боль в груди: ', heart['chest pain type'].unique())
    if chest_pain_type == 1:
        st.sidebar.markdown('Боль: отсутствует')
    if chest_pain_type == 2:
        st.sidebar.markdown('Боль: слабая')
    if chest_pain_type == 3:
        st.sidebar.markdown('Боль: средняя')
    if chest_pain_type == 4:
        st.sidebar.markdown('Боль: безсимптомная')

    exercise_angina = st.sidebar.selectbox('Заболевание ангиной : ', heart['exercise angina'].unique())
    if exercise_angina == 1:
        st.sidebar.markdown('Переболел')
    if exercise_angina == 0:
        st.sidebar.markdown('Не болел')

    oldpeak = st.sidebar.slider('Депрессия: ', min_value=0.0, max_value=10.0, value=None , step=0.1, format=None)
    st.sidebar.markdown(f'Уровень депрессии: {oldpeak}')
 
    data = {'sex': sex,
            'age': age,
            'resting bp s': resting_bp_s,
            'max heart rate': max_heart_rate,
            'resting ecg': resting_ecg,
            'ST slope': st_slope,
            'cholesterol': cholesterol,
            'fasting blood sugar': fasting_blood_sugar,
            'chest pain type': chest_pain_type,
            'exercise angina': exercise_angina,
            'oldpeak': oldpeak}

    features = pd.DataFrame(data, index=[0])
    return features
df = user_input_features()

# Загружаем сохраненуую модель
save_model = load_model('diploma_heart.hdf5')

proba = np.array(df)
proba = normalize(proba)

test_proba = save_model.predict(proba)
st.subheader(f'Прогноз заболеваемости: {100*test_proba[0][1]:.2f} %')

#if test_proba[0][1] < 0.5:
#    st.balloons()
