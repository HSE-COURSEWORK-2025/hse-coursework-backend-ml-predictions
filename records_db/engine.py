import logging
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text

from .settings import settings

logger = logging.getLogger("database")


class DbEngine:
    def __init__(self):
        self.url = f"{settings.RECORDS_DB_ENGINE}://{settings.RECORDS_DB_USER}:{settings.RECORDS_DB_PASSWORD}@{settings.RECORDS_DB_HOST}:{settings.RECORDS_DB_PORT}/{settings.RECORDS_DB_NAME}"
        self.engine = create_engine(self.url, pool_pre_ping=True)
        self.session = sessionmaker(bind=self.engine)

    def create_session(self):
        return self.session(bind=self.engine)

    def request(self, db_request: str | Any) -> list:
        with self.create_session() as session:
            session.begin()
            try:
                result = session.execute(db_request)
            except:
                session.rollback()
                raise
            else:
                session.commit()
                return result


records_db_engine = DbEngine()


async def db_engine_check():
    logger.info(
        f"connecting to database {settings.RECORDS_DB_HOST}:{settings.RECORDS_DB_PORT}"
    )
    try:
        version_info = records_db_engine.request(text("SELECT version();")).fetchone()
    except Exception as e:
        logger.error(f"error connecting to database: {e}")
        raise e
    logger.info(f"database version: {version_info[0]}")
