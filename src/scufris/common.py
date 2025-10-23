import functools
import logging
import os
import time

import psutil
from rich.logging import RichHandler
from collections import OrderedDict
import time
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from proglog import ProgressBarLogger


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


class RichProgressBarLogger(ProgressBarLogger):
    """Use Rich Progress bars instead of tqdm."""

    def __init__(
        self,
        init_state=None,
        bars=None,
        ignored_bars=None,
        logged_bars="all",
        min_time_interval=0,
        ignore_bars_under=0,
        refresh_per_second=5,
    ):
        super().__init__(
            init_state=init_state,
            bars=bars,
            ignored_bars=ignored_bars,
            logged_bars=logged_bars,
            min_time_interval=min_time_interval,
            ignore_bars_under=ignore_bars_under,
        )
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[{task.completed}/{task.total}]"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            refresh_per_second=refresh_per_second,
        )
        self.task_map = {}  # bar_name -> rich task_id
        self.progress.__enter__()

    def new_rich_task(self, bar_name):
        info = self.bars[bar_name]
        total = info.get("total")
        desc = info.get("title", bar_name)
        task_id = self.progress.add_task(desc, total=total)
        self.task_map[bar_name] = task_id

    def bars_callback(self, bar, attr, value, old_value):
        # Called when a bar attribute changes (index, total, message)
        # Ensure we have a corresponding rich task
        if bar not in self.task_map:
            self.new_rich_task(bar)

        task_id = self.task_map[bar]
        if attr == "total":
            self.progress.update(task_id, total=value)
        elif attr == "index":
            # Advance by difference
            advance = value - (old_value if old_value is not None else 0)
            if advance < 0:
                # reset scenario: recreate task
                self.progress.remove_task(task_id)
                self.new_rich_task(bar)
                task_id = self.task_map[bar]
                advance = value
            self.progress.update(task_id, advance=advance)
        elif attr == "message":
            # we might update a description or extra field
            self.progress.update(
                task_id, description=f"{self.bars[bar]['title']}: {value}"
            )

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.progress.__exit__(exc_type, exc_val, exc_tb)
