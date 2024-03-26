import os
from typing import Dict, List
from app import ROOT_DIR

import dotenv
from sqlalchemy import JSON, Column, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

dotenv.load_dotenv()

MONITOR_DB_URI = os.environ.get('SQLALCHEMY_DATABASE_URI') or 'sqlite:///sample.db'
LOG_LEVEL = os.environ.get('LOG_LEVEL') or 'INFO'

echo = False
if LOG_LEVEL == 'DEBUG':
    echo = True
    
print('Creating database for metrics: ', MONITOR_DB_URI)
engine = create_engine(MONITOR_DB_URI, echo=echo)  # Change the database name if needed

Base = declarative_base()

class Eval(Base):
    """Define data model for the table 'eval', to store metrics for later evaluation."""
    __tablename__ = 'eval'
    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(String)
    log = Column(JSON)

Base.metadata.create_all(engine)

def write_data(event_id, log):
    """
    Write data to the database using the provided event ID and log. 
    """

    Session = sessionmaker(bind=engine)
    with Session() as session:
        new_eval = Eval(event_id=event_id, log=log)
        session.add(new_eval)
        session.commit()


def get_data(event_id: str | None = None) -> List[Dict]:
    """
    Function to retrieve data from the evals database based on the provided event ID.
    """

    Session = sessionmaker(bind=engine)
    with Session() as session:
        if event_id:
            evals = session.query(Eval).filter_by(event_id=event_id).all()
        else:
            evals = session.query(Eval).all()

        results = []
        for eval in evals:
            results.append({'event_id': eval.event_id, 'log': eval.log})
    return results


def delete_rows(event_id: str | None = None, table_name: str = 'eval') -> None:
    """Delete rows from a table. If no event_id is given, delete all rows."""
    session = sessionmaker(bind=engine)
    with session() as db_session:
        if event_id:
            table_query = db_session.query(eval(table_name)).filter_by(event_id=event_id)
        else:
            table_query = db_session.query(eval(table_name))
        table_query.delete()
        db_session.commit()
