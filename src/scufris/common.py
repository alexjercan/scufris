import functools
import logging
import os
import time

import psutil
from rich.logging import RichHandler


def setup_logger(run_id: str) -> logging.Logger:
    """Setup a logger with a unique run ID."""

    class RunIDFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            record.run_id = run_id
            return True

    FORMAT = "[RUN %(run_id)s] %(message)s"

    logger = get_logger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(FORMAT, datefmt="[%X]")
    console_handler = RichHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addFilter(RunIDFilter())

    return logger


def get_logger() -> logging.Logger:
    """Get the logger instance."""

    return logging.getLogger("scufris")


def with_stats(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger()
        process = psutil.Process(os.getpid())

        start_memory = process.memory_info().rss
        start_time = time.time()

        result = func(*args, **kwargs)

        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.debug(
            f"Function '{func.__name__}' executed in {elapsed_time:.2f} seconds."
        )

        end_memory = process.memory_info().rss
        memory_used = end_memory - start_memory
        memory_used = memory_used / (1024 * 1024)  # Convert to MB
        logger.debug(f"Function '{func.__name__}' used {memory_used:.2f} MB of memory.")

        return result

    return wrapper
