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
from make_predictions_funcs import make_insomnia_apnea_predictions, make_hypertension_predictions

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



async def main(email: str):
    logger.info(f'launch for user {email}')
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
    try:
        await make_insomnia_apnea_predictions(
            records_db_session, users_db_session, email, iteration_number
        )
    except Exception as e:
        logger.error(f'error during make_insomnia_apnea_predictions: {e}')
    
    try:
        await make_hypertension_predictions(records_db_session, users_db_session, email, iteration_number)
    except Exception as e:
        logger.error(f'error during make_hypertension_predictions: {e}')
    

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
