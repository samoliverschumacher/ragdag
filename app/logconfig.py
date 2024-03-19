import logging

from app.database import write_data

# Define a custom log level
EVAL = 25
logging.addLevelName(EVAL, 'EVAL')


class CustomLogger(logging.Logger):

    def eval(self, event_id: str, log: dict) -> None:
        assert type(log) == dict
        write_data(event_id=event_id, log=log)


def setup_logger(name) -> CustomLogger:
    """A logger implementation of a system monitoring service.

    EVAL level logs are sent to a database.

    Other levels are sent to std out.
    """
    logging.setLoggerClass(CustomLogger)

    logger = CustomLogger(name)

    # Alternative to logging to database, is to use .log( level=EVAL )
    file_handler = logging.FileHandler('eval_logs.log', mode='a')
    file_handler.setLevel(EVAL)
    file_handler.addFilter(lambda record: record.levelno == EVAL)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)8s] %(filename)s:%(lineno)3s - %(funcName)20s() [%(module)s] %(message)s"))
    stream_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    return logger
