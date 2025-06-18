from users_db.engine import users_db_engine


async def get_users_db_session():
    session = users_db_engine.create_session()
    try:
        yield session
    finally:
        session.close()
