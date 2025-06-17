import asyncio
import argparse
import logging
import re
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
from models import SleepDisorderInput

# Настройки
settings = Settings()
EMAIL_REGEX = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")

# Логирование
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


async def send_ml_start_notification(
    email: str, iteration_number: int, start_time: str
):
    subject = f"[Iteration #{iteration_number}] Запуск ML-анализа"
    body = f"""
    <html><body>
      <h2>🚀 ML Анализ — Запуск итерации #{iteration_number}</h2>
      <p><strong>Пользователь:</strong> {email}</p>
      <p><strong>Время старта:</strong> {start_time}</p>
      <p>Начинаем генерацию прогнозов на основе данных пользователя.</p>
    </body></html>
    """
    await notifications_api.send_email(email, subject, body)
    logger.info("Sent ML start notification email")


async def send_ml_completion_notification(
    email: str, iteration_number: int, start_time: str, finish_time: str
):
    subject = f"[Iteration #{iteration_number}] Завершение ML-анализа"
    body = f"""
    <html><body>
      <h2>✅ ML Анализ — Итерация #{iteration_number} завершена</h2>
      <p><strong>Пользователь:</strong> {email}</p>
      <p><strong>Время старта:</strong> {start_time}</p>
      <p><strong>Время окончания:</strong> {finish_time}</p>
      <p>Прогнозы успешно сохранены.</p>
    </body></html>
    """
    await notifications_api.send_email(email, subject, body)
    logger.info("Sent ML completion notification email")


async def make_predictions(
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

    # Функция для вычисления среднего за последние 30 дней или крайние 30 записей
    async def compute_avg_raw_records(session, data_type):
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
        result = await session.execute(
            select(RawRecords.value)
            .where(RawRecords.email == email, RawRecords.data_type == data_type)
            .order_by(RawRecords.time.desc())
            .limit(30)
        )
        values = [float(row[0]) for row in result.fetchall()]
        return float(sum(values) / len(values)) if values else 0.0

    async def compute_avg_processed_records(session, data_type):
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
        result = await session.execute(
            select(ProcessedRecords.value)
            .where(ProcessedRecords.email == email, ProcessedRecords.data_type == data_type)
            .order_by(ProcessedRecords.time.desc())
            .limit(30)
        )
        values = [float(row[0]) for row in result.fetchall()]
        return float(sum(values) / len(values)) if values else 0.0

    # Вычисляем параметры
    sleep_duration_hours = await compute_avg_processed_records(records_db_session, "SleepSessionTimeData")
    if sleep_duration_hours:
        sleep_duration_hours = sleep_duration_hours / 60
    physical_activity_mins_daily = await compute_avg_processed_records(
        records_db_session, "ActiveMinutesRecord"
    )
    heart_rate = await compute_avg_raw_records(records_db_session, "HeartRateRecord")
    daily_steps = await compute_avg_processed_records(records_db_session, "StepsRecord")


    bmi_category = "normal"

    # Проверка наличия всех необходимых данных
    required_fields = {
        'gender': gender,
        'age': age,
        'sleep_duration_hours': sleep_duration_hours,
        'bmi_category': bmi_category,
        'physical_activity_mins_daily': physical_activity_mins_daily,
        'heart_rate': heart_rate,
        'daily_steps': daily_steps,
    }
    missing = [name for name, value in required_fields.items() if value is None]
    if missing:
        logger.error(f"Missing required fields for ML input: {', '.join(missing)}")
        return

    gender = gender.capitalize()
    bmi_category = bmi_category.capitalize()
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

    logger.info(f'predicted: {predictions}')

    now = datetime.utcnow()
    records = []

    name = 'insomnia_apnea'
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
    logger.info(f"Committed {len(records)} ML predictions to DB")


async def main(email: str):
    # Подключаемся к БД
    records_db_session = await get_records_db_session().__anext__()
    users_db_session = await get_users_db_session().__anext__()

    # Определяем итерацию
    result = records_db_session.execute(
        select(func.max(MLPredictionsRecords.iteration_num)).where(
            MLPredictionsRecords.email == email
        )
    )
    max_iter = result.scalar() or 0
    iteration_number = max_iter + 1

    start_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    try:
        await send_ml_start_notification(email, iteration_number, start_time)
    except Exception as e:
        logger.error(f"failed to send notification: {e}")

    # Выполняем предсказания
    await make_predictions(
        records_db_session, users_db_session, email, iteration_number
    )

    finish_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    try:
        await send_ml_completion_notification(
            email, iteration_number, start_time, finish_time
        )
    except Exception as e:
        logger.error(f"failed to send notification: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ML predictions for user.")
    parser.add_argument(
        "-e", "--email", required=True, help="Email address of the user"
    )
    args = parser.parse_args()

    if not EMAIL_REGEX.fullmatch(args.email):
        logger.error(f"Invalid email format: {args.email}")
        sys.exit(1)

    asyncio.run(main(args.email))
