import json
import pickle
import numpy as np
from pydantic import BaseModel
from typing import Optional
from models import SleepDisorderInput, SleepDisorderOutput


def predict_sleep_disorder(
    input_data: SleepDisorderInput
) -> SleepDisorderOutput:
    """
    Predict sleep disorder probabilities using a pre-trained model.

    :param input_data: SleepDisorderInput Pydantic model instance
    :return: SleepDisorderOutput Pydantic model instance with probabilities
    """
    # Load model and encoders
    with open('./ml_models_files/insomnia_apnea.pkl', 'rb') as f:
        data = pickle.load(f)
        pipeline = data['model']
        sleep_encoder = data['sleep_encoder']
        gender_encoder = data.get('gender_encoder')
        bmi_encoder = data.get('bmi_encoder')

    # Encode categorical features
    gender_code = gender_encoder.transform([input_data.gender])[0]
    bmi_code = bmi_encoder.transform([input_data.bmi_category])[0]

    # Prepare numerical features
    age = input_data.age
    sleep_duration = input_data.sleep_duration_hours
    try:
        physical_activity = float(input_data.physical_activity_mins_daily)
    except (TypeError, ValueError):
        raise ValueError("physical_activity_mins_daily must be a number or numeric string")
    heart_rate = input_data.heart_rate
    daily_steps = input_data.daily_steps

    # Build feature vector
    features = [
        gender_code,
        age,
        sleep_duration,
        physical_activity,
        bmi_code,
        heart_rate,
        daily_steps,
    ]
    sample = np.array(features).reshape(1, -1)

    # Perform prediction
    classifier = pipeline.named_steps['clf']
    probabilities = classifier.predict_proba(sample)[0]
    class_labels = classifier.classes_
    diagnosis_names = sleep_encoder.inverse_transform(class_labels)

    # Map results to dict
    result_dict = {str(name).replace(' ', '_'): float(prob) for name, prob in zip(diagnosis_names, probabilities)}
    # Return Pydantic output model
    return SleepDisorderOutput.model_validate(result_dict)
