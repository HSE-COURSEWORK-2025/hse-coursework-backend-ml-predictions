import json
import pickle
import numpy as np
from pydantic import BaseModel
from typing import Optional
from models import SleepDisorderInput, SleepDisorderOutput


def predict_hypertension(country, age, bmi, physical_activity_level, sleep_duration, heart_rate, gender):
    """
    Предсказание наличия гипертонии по неэнкодированным входным данным.
    Возвращает вероятности классов в формате JSON.
    """
    # Загружаем модель и энкодеры из файла
    with open('./ml_models_files/hypertension.pkl', 'rb') as f:
        data = pickle.load(f)
        model = data['model']
        hypertension_encoder = data['hypertension_encoder']
        physical_activity_level_encoder = data['physical_activity_level_encoder']
        gender_encoder = data['gender_encoder']
        country_encoder = data['country_encoder']

    # Кодируем входные данные
    country_enc = country_encoder.transform([country])[0]
    physical_activity_enc = physical_activity_level_encoder.transform([physical_activity_level])[0]
    gender_enc = gender_encoder.transform([gender])[0]

    features = [country_enc, age, bmi, physical_activity_enc, sleep_duration, heart_rate, gender_enc]
    sample = np.array(features).reshape(1, -1)

    probas = model.named_steps['clf'].predict_proba(sample)[0]
    class_labels = model.named_steps['clf'].classes_
    diagnosis_names = hypertension_encoder.inverse_transform(class_labels)
    result = {diagnosis: float(prob) for diagnosis, prob in zip(diagnosis_names, probas)}
    return json.dumps(result, ensure_ascii=False)
