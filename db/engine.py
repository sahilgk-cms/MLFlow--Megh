import sqlalchemy
from sqlalchemy import create_engine



def get_engine(db_user: str, db_password: str, db_host: str,
                db_port: str, db_name: str) -> sqlalchemy.engine.base.Engine:
    engine = create_engine(
        f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    )
    return engine