import json
import pickle
import numpy as np


def predict_depression(heart_rate, sleep_duration, physical_activity_steps):
    """
    Предсказание наличия депрессии по входным данным:
    - heart_rate: пульс (int)
    - sleep_duration: продолжительность сна (float)
    - physical_activity_steps: количество шагов (int)
    Возвращает вероятности классов в формате JSON.
    """
    with open("./ml_models_files/depression.pkl", "rb") as f:
        data = pickle.load(f)
        model = data["model"]

    features = [heart_rate, sleep_duration, physical_activity_steps]
    sample = np.array(features).reshape(1, -1)

    probas = model.named_steps["clf"].predict_proba(sample)[0]
    class_labels = model.named_steps["clf"].classes_
    result = {int(label): float(prob) for label, prob in zip(class_labels, probas)}
    return json.dumps(result, ensure_ascii=False)
