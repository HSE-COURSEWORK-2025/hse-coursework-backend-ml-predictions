import asyncio
import argparse
import logging
import re
import pickle
import sys
from datetime import datetime, date, timedelta

from sqlalchemy.future import select
from sqlalchemy import func, cast, Numeric

from notifications import notifications_api
from records_db.schemas import MLPredictionsRecords, RawRecords, ProcessedRecords
from records_db.db_session import get_records_db_session
from users_db.db_session import get_users_db_session
from users_db.schemas import Users

from settings import Settings

from ml_models.insomnia_apnea import predict_sleep_disorder
from ml_models.hypertension import predict_hypertension
from models import SleepDisorderInput


logger = logging.getLogger(__name__)


def get_bmi_category(weight_kg: int | float, height_meters: int | float):
    if weight_kg and height_meters:
        bmi = weight_kg / (height_meters**2)

        if bmi < 18.5:
            return "Normal"
        elif bmi < 25:
            return "Normal Weight"
        elif bmi < 30:
            return "Overweight"
        else:
            return "Obese"
    else:
        raise Exception("not enough date for bmi")


# Функция для вычисления среднего за последние 30 дней или крайние 30 записей
async def compute_avg_raw_records(session, data_type, email):
    today = date.today()
    last_30_days = today - timedelta(days=30)

    # Попытка среднего за последние 30 дней
    result = session.execute(
        select(func.avg(cast(RawRecords.value, Numeric))).where(
            RawRecords.email == email,
            RawRecords.data_type == data_type,
            RawRecords.time >= last_30_days,
        )
    )
    avg = result.scalar()
    if avg is not None:
        return float(avg)

    # Если нет данных за последние 30 дней — берём крайние 30 записей
    result = session.execute(
        select(RawRecords.value)
        .where(RawRecords.email == email, RawRecords.data_type == data_type)
        .order_by(RawRecords.time.desc())
        .limit(30)
    )
    values = [float(row[0]) for row in result.fetchall()]
    return float(sum(values) / len(values)) if values else 0.0


async def compute_avg_processed_records(session, data_type, email):
    today = date.today()
    last_30_days = today - timedelta(days=30)

    # Попытка среднего за последние 30 дней
    result = session.execute(
        select(func.avg(cast(ProcessedRecords.value, Numeric))).where(
            ProcessedRecords.email == email,
            ProcessedRecords.data_type == data_type,
            ProcessedRecords.time >= last_30_days,
        )
    )
    avg = result.scalar()
    if avg is not None:
        return float(avg)

    # Если нет данных за последние 30 дней — берём крайние 30 записей
    result = session.execute(
        select(ProcessedRecords.value)
        .where(ProcessedRecords.email == email, ProcessedRecords.data_type == data_type)
        .order_by(ProcessedRecords.time.desc())
        .limit(30)
    )
    values = [float(row[0]) for row in result.fetchall()]
    return float(sum(values) / len(values)) if values else 0.0


async def make_insomnia_apnea_predictions(
    records_db_session, users_db_session, email: str, iteration: int
):
    # Получаем данные пользователя
    result = users_db_session.execute(select(Users).where(Users.email == email))
    user: Users | None = result.scalar_one_or_none()
    if user is None:
        logger.error(f"User with email '{email}' not found in users database")
        return

    # Извлекаем необходимые поля
    gender = user.gender
    birth_date: date = user.birth_date.date()
    today = date.today()
    age = (
        today.year
        - birth_date.year
        - ((today.month, today.day) < (birth_date.month, birth_date.day))
    )

    # Вычисляем параметры
    sleep_duration_hours = await compute_avg_processed_records(
        records_db_session, "SleepSessionTimeData", email
    )
    if sleep_duration_hours:
        sleep_duration_hours = sleep_duration_hours / 60
    physical_activity_mins_daily = await compute_avg_processed_records(
        records_db_session, "ActiveMinutesRecord", email
    )
    heart_rate = await compute_avg_raw_records(
        records_db_session, "HeartRateRecord", email
    )
    daily_steps = await compute_avg_processed_records(
        records_db_session, "StepsRecord", email
    )

    last_weight_record = records_db_session.execute(
        select(RawRecords.value)
        .where(RawRecords.email == email, RawRecords.data_type == "WeightRecord")
        .order_by(RawRecords.time.desc())
        .limit(1)
    )
    weight = last_weight_record.scalar_one_or_none()

    last_height_record = records_db_session.execute(
        select(RawRecords.value)
        .where(RawRecords.email == email, RawRecords.data_type == "HeightRecord")
        .order_by(RawRecords.time.desc())
        .limit(1)
    )
    height = last_height_record.scalar_one_or_none()

    try:
        weight = float(weight)
        height = float(height)
        bmi_category = get_bmi_category(weight, height)
    except Exception as e:
        logger.error(f"Error during bmi calc: {e}")
        return

    # Проверка наличия всех необходимых данных
    required_fields = {
        "gender": gender,
        "age": age,
        "sleep_duration_hours": sleep_duration_hours,
        "bmi_category": bmi_category,
        "physical_activity_mins_daily": physical_activity_mins_daily,
        "heart_rate": heart_rate,
        "daily_steps": daily_steps,
    }
    missing = [name for name, value in required_fields.items() if value is None]
    if missing:
        logger.error(f"Missing required fields for ML input: {', '.join(missing)}")
        return

    gender = gender.capitalize()
    # Готовим данные для ML-модели
    input_data = SleepDisorderInput(
        gender=gender,
        age=age,
        sleep_duration_hours=sleep_duration_hours,
        physical_activity_mins_daily=int(physical_activity_mins_daily),
        bmi_category=bmi_category,
        heart_rate=int(heart_rate),
        daily_steps=int(daily_steps),
    )
    try:
        predictions = predict_sleep_disorder(input_data)
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return

    logger.info(f"predicted: {predictions}")

    now = datetime.utcnow()
    records = []

    name = "insomnia_apnea"
    result_value = predictions.model_dump_json()
    rec = MLPredictionsRecords(
        email=email,
        result_value=result_value,
        diagnosis_name=name,
        iteration_num=iteration,
        iteration_datetime=now,
    )
    records.append(rec)
    logger.info(f"Generated {name}: result_value")

    records_db_session.add_all(records)
    records_db_session.commit()
    logger.info(f"Committed {len(records)} ML predictions (insomnia/apnea) to DB")


def categorize_physical_activity(minutes_per_day: float) -> str:
    """
    Категоризирует уровень физической активности (минуты в день) в:
      - 'Low'      : < 30 мин
      - 'Moderate' : 30–60 мин включительно
      - 'High'     : > 60 мин
    """
    if minutes_per_day < 30:
        return "Low"
    elif minutes_per_day <= 60:
        return "Moderate"
    else:
        return "High"
    

async def make_hypertension_predictions(records_db_session, users_db_session, email: str, iteration: int):
    # Получаем данные пользователя
    result = users_db_session.execute(select(Users).where(Users.email == email))
    user: Users | None = result.scalar_one_or_none()
    if user is None:
        logger.error(f"User with email '{email}' not found in users database")
        return

    # Извлекаем необходимые поля
    gender = user.gender
    birth_date: date = user.birth_date.date()
    today = date.today()
    age = (
        today.year
        - birth_date.year
        - ((today.month, today.day) < (birth_date.month, birth_date.day))
    )

    # Получаем необходимые параметры
    heart_rate = await compute_avg_raw_records(records_db_session, "HeartRateRecord", email)
    sleep_duration_minutes = await compute_avg_processed_records(records_db_session, "SleepSessionTimeData", email)
    
    sleep_duration_hours = sleep_duration_minutes / 60 if sleep_duration_minutes is not None else None
    physical_activity_mins_daily = await compute_avg_processed_records(
        records_db_session, "ActiveMinutesRecord", email
    )

    last_weight_record = records_db_session.execute(
        select(RawRecords.value)
        .where(RawRecords.email == email, RawRecords.data_type == "WeightRecord")
        .order_by(RawRecords.time.desc())
        .limit(1)
    )
    weight = last_weight_record.scalar_one_or_none()

    last_height_record = records_db_session.execute(
        select(RawRecords.value)
        .where(RawRecords.email == email, RawRecords.data_type == "HeightRecord")
        .order_by(RawRecords.time.desc())
        .limit(1)
    )
    height = last_height_record.scalar_one_or_none()

    try:
        weight = float(weight)
        height = float(height)
        bmi = weight / (height ** 2)
    except Exception as e:
        logger.error(f"Error during bmi calc: {e}")
        return

    # Собираем только нужные параметры для модели
    country = 'Russia'
    required_fields = {
        "country": country,
        "age": age,
        "bmi": bmi,
        "physical_activity_level": physical_activity_mins_daily,
        "sleep_duration": sleep_duration_hours,
        "heart_rate": heart_rate,
        "gender": gender,
    }
    missing = [name for name, value in required_fields.items() if value is None]
    if missing:
        logger.error(f"Missing required fields for ML input: {', '.join(missing)}")
        return

    gender = gender.capitalize()
    input_data = {
        "country": required_fields["country"],
        "age": required_fields["age"],
        "bmi": required_fields["bmi"],
        "physical_activity_level": categorize_physical_activity(int(required_fields["physical_activity_level"])),
        "sleep_duration": float(required_fields["sleep_duration"]),
        "heart_rate": int(required_fields["heart_rate"]),
        "gender": gender,
    }

    try:
        predictions = predict_hypertension(**input_data)
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return

    logger.info(f"predicted: {predictions}")

    now = datetime.utcnow()
    records = []

    name = "hypertension"
    # predictions is already a JSON string
    result_value = predictions
    rec = MLPredictionsRecords(
        email=email,
        result_value=result_value,
        diagnosis_name=name,
        iteration_num=iteration,
        iteration_datetime=now,
    )
    records.append(rec)

    records_db_session.commit()
    logger.info(f"Committed {len(records)} ML predictions (hypertension) to DB")
